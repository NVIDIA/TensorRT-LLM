/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorrt_llm/kernels/kdaDecode/kdaDecode.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/kdaDecode/kdaDecodeInternal.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::kdaDecode
{

namespace
{

enum class KdaDecodeKernel
{
    kLegacyCompactHeads,
    kLegacyManyHeads,
};

KdaDecodeKernel selectKdaDecodeKernel(KdaDecodeParams const& params)
{
    bool const useCompactHeads = shouldUseCompactHeads(params.batchSize, params.numHeads, params.numValueHeads);
    return useCompactHeads ? KdaDecodeKernel::kLegacyCompactHeads : KdaDecodeKernel::kLegacyManyHeads;
}

} // namespace

void invokeKdaDecode(KdaDecodeParams const& params, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(params.numHeads == params.numValueHeads, "KDA decode requires numHeads == numValueHeads");
    switch (selectKdaDecodeKernel(params))
    {
    case KdaDecodeKernel::kLegacyCompactHeads: launchKdaDecodeLegacyCompactHeads(params, stream); break;
    case KdaDecodeKernel::kLegacyManyHeads: launchKdaDecodeLegacyManyHeads(params, stream); break;
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels::kdaDecode

TRTLLM_NAMESPACE_END
