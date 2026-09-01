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
    kBlackwellSingleCta,
    kBlackwellTwoStageBulk,
    kBlackwellFourStageBulk,
    kBlackwellFourCtaCluster,
};

KdaDecodeKernel selectLegacyKdaDecodeKernel(KdaDecodeParams const& params)
{
    bool const useCompactHeads = shouldUseCompactHeads(params.batchSize, params.numHeads, params.numValueHeads);
    return useCompactHeads ? KdaDecodeKernel::kLegacyCompactHeads : KdaDecodeKernel::kLegacyManyHeads;
}

KdaDecodeKernel selectBlackwellKdaDecodeKernel(int smVersion, int64_t workload)
{
    if (smVersion == 100)
    {
        if (workload <= 48)
        {
            return KdaDecodeKernel::kBlackwellFourCtaCluster;
        }
        if ((workload >= 320 && workload <= 864) || (workload >= 960 && workload <= 1152)
            || (workload >= 1440 && workload <= 3072))
        {
            return KdaDecodeKernel::kBlackwellSingleCta;
        }
        return KdaDecodeKernel::kBlackwellTwoStageBulk;
    }
    else
    {
        if (workload <= 48)
        {
            return KdaDecodeKernel::kBlackwellFourCtaCluster;
        }
        if ((workload >= 512 && workload <= 864) || (workload >= 1440 && workload <= 6144))
        {
            return KdaDecodeKernel::kBlackwellSingleCta;
        }
        return KdaDecodeKernel::kBlackwellTwoStageBulk;
    }
}

KdaDecodeKernel selectKdaDecodeKernel(KdaDecodeParams const& params)
{
    static int const smVersion = tensorrt_llm::common::getSMVersion();
    if (smVersion == 100 || smVersion == 103)
    {
        int64_t const workload = static_cast<int64_t>(params.batchSize) * params.numHeads;
        return selectBlackwellKdaDecodeKernel(smVersion, workload);
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
    case KdaDecodeKernel::kBlackwellSingleCta: launchKdaDecodeBlackwellSingleCta(params, stream); break;
    case KdaDecodeKernel::kBlackwellTwoStageBulk: launchKdaDecodeBlackwellTwoStageBulk(params, stream); break;
    case KdaDecodeKernel::kBlackwellFourStageBulk: launchKdaDecodeBlackwellFourStageBulk(params, stream); break;
    case KdaDecodeKernel::kBlackwellFourCtaCluster: launchKdaDecodeBlackwellFourCtaCluster(params, stream); break;
    }
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels::kdaDecode

TRTLLM_NAMESPACE_END
