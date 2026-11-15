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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/moe/communication/moeAlltoAllLamportKernels.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::moe_comm
{

// The top-k values the fence path instantiates (moeAlltoAllKernels.cu SWITCH_TOP_K).
#define MOE_A2A_LAMPORT_SWITCH_TOP_K(topK, TOP_K, ...)                                                                \
    switch (topK)                                                                                                      \
    {                                                                                                                  \
    case 22: { constexpr int TOP_K = 22; __VA_ARGS__; break; }                                                         \
    case 18: { constexpr int TOP_K = 18; __VA_ARGS__; break; }                                                         \
    case 16: { constexpr int TOP_K = 16; __VA_ARGS__; break; }                                                         \
    case 14: { constexpr int TOP_K = 14; __VA_ARGS__; break; }                                                         \
    case 12: { constexpr int TOP_K = 12; __VA_ARGS__; break; }                                                         \
    case 10: { constexpr int TOP_K = 10; __VA_ARGS__; break; }                                                         \
    case 8: { constexpr int TOP_K = 8; __VA_ARGS__; break; }                                                           \
    case 6: { constexpr int TOP_K = 6; __VA_ARGS__; break; }                                                           \
    case 4: { constexpr int TOP_K = 4; __VA_ARGS__; break; }                                                           \
    case 2: { constexpr int TOP_K = 2; __VA_ARGS__; break; }                                                           \
    case 1: { constexpr int TOP_K = 1; __VA_ARGS__; break; }                                                           \
    default: TLLM_CHECK_WITH_INFO(false, "MoE all-to-all (sentinel-polled): unsupported top_k %d", topK);             \
    }

void moe_a2a_lamport_dispatch_launch(moe_a2a_lamport::DispatchParams const& params, int topK, cudaStream_t stream)
{
    bool const pdl = tensorrt_llm::common::getEnvEnablePDL();
    cudaError_t err = cudaErrorInvalidValue;
    MOE_A2A_LAMPORT_SWITCH_TOP_K(topK, TOP_K, err = moe_a2a_lamport::launchDispatch<TOP_K>(params, stream, pdl));
    TLLM_CHECK_WITH_INFO(
        err == cudaSuccess, "MoE all-to-all (sentinel-polled) dispatch launch failed: %s", cudaGetErrorString(err));
}

void moe_a2a_lamport_combine_launch(
    moe_a2a_lamport::CombineParams const& params, int topK, bool isBf16, cudaStream_t stream)
{
    bool const pdl = tensorrt_llm::common::getEnvEnablePDL();
    cudaError_t err = cudaErrorInvalidValue;
    if (isBf16)
    {
        MOE_A2A_LAMPORT_SWITCH_TOP_K(
            topK, TOP_K, err = moe_a2a_lamport::launchCombine<__nv_bfloat16, TOP_K>(params, stream, pdl));
    }
    else
    {
        MOE_A2A_LAMPORT_SWITCH_TOP_K(topK, TOP_K, err = moe_a2a_lamport::launchCombine<half, TOP_K>(params, stream, pdl));
    }
    TLLM_CHECK_WITH_INFO(
        err == cudaSuccess, "MoE all-to-all (sentinel-polled) combine launch failed: %s", cudaGetErrorString(err));
}

#undef MOE_A2A_LAMPORT_SWITCH_TOP_K

} // namespace kernels::moe_comm

TRTLLM_NAMESPACE_END
