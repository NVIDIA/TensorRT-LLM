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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/cutlass_kernels/w4a16_nvfp4_gemm/w4a16_nvfp4_gemm_sm120.cuh"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/cutlassGemmW4A16NVFP4.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace cutlass_gemm_w4a16_nvfp4
{
namespace
{

bool isSupported(Params const& params)
{
    int const smVersion = tensorrt_llm::common::getSMVersion();
    return (smVersion == 120 || smVersion == 121) && params.inputType == CUDA_R_16BF && params.outputType == CUDA_R_16BF
        && params.m > 16 && params.n > 0 && params.k > 0 && params.n % 32 == 0 && params.k % 32 == 0
        && params.weightScale != nullptr && params.weightGlobalScale != nullptr;
}

} // namespace

bool cutlassGemmDispatcher(Params const& params, cudaStream_t stream)
{
    if (!isSupported(params))
    {
        int const smVersion = tensorrt_llm::common::getSMVersion();
        TLLM_LOG_WARNING(
            "tensorrt_llm::kernels::cutlass_gemm_w4a16_nvfp4::cutlassGemmDispatcher [NOT DISPATCHED], "
            "inputType=%d, outputType=%d, m=%d, n=%d, k=%d, sm=%d",
            params.inputType, params.outputType, params.m, params.n, params.k, smVersion);
        return false;
    }

    return sm120::dispatch(params, stream);
}

} // namespace cutlass_gemm_w4a16_nvfp4
} // namespace kernels

TRTLLM_NAMESPACE_END
