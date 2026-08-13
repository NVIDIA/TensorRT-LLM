/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "mamba2MTPSSMCacheKernel.cuh"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Prevent instantiation in this translation unit — provided by vec4/vec8/vec16 .cu files
extern template void launchMamba2MTPSSMCacheKernel<4>(
    Mamba2MTPSSMCacheParams const& params, dim3 grid, dim3 block, cudaStream_t stream);
extern template void launchMamba2MTPSSMCacheKernel<8>(
    Mamba2MTPSSMCacheParams const& params, dim3 grid, dim3 block, cudaStream_t stream);
extern template void launchMamba2MTPSSMCacheKernel<16>(
    Mamba2MTPSSMCacheParams const& params, dim3 grid, dim3 block, cudaStream_t stream);

#define MTP_DISPATCH_VEC_SIZE(SSM_DIM, NAME, ...)                                                                      \
    [&]                                                                                                                \
    {                                                                                                                  \
        if (SSM_DIM == 128)                                                                                            \
        {                                                                                                              \
            constexpr int NAME = 4;                                                                                    \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else if (SSM_DIM == 256)                                                                                       \
        {                                                                                                              \
            constexpr int NAME = 8;                                                                                    \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else if (SSM_DIM == 512)                                                                                       \
        {                                                                                                              \
            constexpr int NAME = 16;                                                                                   \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            TLLM_CHECK_WITH_INFO(false, "Unsupported ssm_dim. Supported: 128, 256, 512");                              \
        }                                                                                                              \
    }()

void invokeMamba2MTPSSMCacheUpdate(Mamba2MTPSSMCacheParams const& params, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(
        (params.head_dim % (MTP_NUM_WARPS * MTP_HDIMS_PER_WARP)) == 0, "head_dim should be a multiple of 8");

    dim3 block(MTP_NUM_BLOCK_THREADS);
    dim3 grid(params.head_dim / (MTP_NUM_WARPS * MTP_HDIMS_PER_WARP), params.nheads, params.bs);

    MTP_DISPATCH_VEC_SIZE(
        params.ssm_dim, DISPATCH_VS, [&] { launchMamba2MTPSSMCacheKernel<DISPATCH_VS>(params, grid, block, stream); });
}

} // namespace kernels

TRTLLM_NAMESPACE_END
