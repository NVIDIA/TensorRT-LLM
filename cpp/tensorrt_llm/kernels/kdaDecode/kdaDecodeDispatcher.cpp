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

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::kdaDecode
{

namespace
{

enum class KdaDecodeKernel
{
    kLegacyCompactHeads,
    kLegacyManyHeads,
    kOptimizedSingleCta,
    kOptimizedTwoStageBulk,
    kOptimizedFourStageBulk,
    kOptimizedFourCtaCluster,
};

KdaDecodeKernel selectLegacyKdaDecodeKernel(KdaDecodeParams const& params)
{
    bool const useCompactHeads = shouldUseCompactHeads(params.batchSize, params.numHeads, params.numValueHeads);
    return useCompactHeads ? KdaDecodeKernel::kLegacyCompactHeads : KdaDecodeKernel::kLegacyManyHeads;
}

KdaDecodeKernel selectOptimizedKdaDecodeKernel(int smVersion, KdaDecodeParams const& params)
{
    int64_t const workload = static_cast<int64_t>(params.batchSize) * params.numHeads;
    if (workload <= 32)
    {
        return KdaDecodeKernel::kOptimizedFourCtaCluster;
    }
    if (workload <= 144)
    {
        return KdaDecodeKernel::kLegacyCompactHeads;
    }

    if (smVersion == 100)
    {
        if (workload >= 672 && workload <= 720)
        {
            return KdaDecodeKernel::kLegacyManyHeads;
        }
        if ((workload >= 624 && workload <= 865) || (workload >= 1368 && workload <= 2496))
        {
            return KdaDecodeKernel::kOptimizedSingleCta;
        }
        return KdaDecodeKernel::kOptimizedTwoStageBulk;
    }

    if (smVersion == 103)
    {
        if ((params.numHeads % 6 == 0 && params.batchSize >= 512) || (workload >= 672 && workload <= 720))
        {
            return KdaDecodeKernel::kLegacyManyHeads;
        }
        if ((workload >= 624 && workload < 672) || (workload >= 816 && workload <= 867))
        {
            return KdaDecodeKernel::kOptimizedSingleCta;
        }
        return KdaDecodeKernel::kOptimizedTwoStageBulk;
    }

    TLLM_THROW("Optimized KDA decode selector requires SM100 or SM103");
}

KdaDecodeKernel selectKdaDecodeKernel(KdaDecodeParams const& params)
{
    static int const smVersion = tensorrt_llm::common::getSMVersion();
    if (smVersion == 100 || smVersion == 103)
    {
        return selectOptimizedKdaDecodeKernel(smVersion, params);
    }
    return selectLegacyKdaDecodeKernel(params);
}

} // namespace

void invokeKdaDecode(KdaDecodeParams const& params, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(params.numHeads == params.numValueHeads, "KDA decode requires numHeads == numValueHeads");
    switch (selectKdaDecodeKernel(params))
    {
    case KdaDecodeKernel::kLegacyCompactHeads: launchKdaDecodeLegacyCompactHeads(params, stream); break;
    case KdaDecodeKernel::kLegacyManyHeads: launchKdaDecodeLegacyManyHeads(params, stream); break;
    case KdaDecodeKernel::kOptimizedSingleCta: launchKdaDecodeOptimizedSingleCta(params, stream); break;
    case KdaDecodeKernel::kOptimizedTwoStageBulk: launchKdaDecodeOptimizedTwoStageBulk(params, stream); break;
    case KdaDecodeKernel::kOptimizedFourStageBulk: launchKdaDecodeOptimizedFourStageBulk(params, stream); break;
    case KdaDecodeKernel::kOptimizedFourCtaCluster: launchKdaDecodeOptimizedFourCtaCluster(params, stream); break;
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels::kdaDecode

TRTLLM_NAMESPACE_END
