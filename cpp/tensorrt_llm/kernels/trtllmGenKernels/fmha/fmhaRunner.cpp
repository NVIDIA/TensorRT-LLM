/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "fmhaRunner.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tensorrt_llm
{
namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

TllmGenFmhaRunner::TllmGenFmhaRunner(Data_type dtypeQ, Data_type dtypeKv, Data_type dtypeOut)
    : mSM(tensorrt_llm::common::getSMVersion())
    , mDtypeQ(dtypeQ)
    , mDtypeKv(dtypeKv)
    , mDtypeOut(dtypeOut)
{
    TLLM_CHECK_WITH_INFO(mSM == kSM_100 || mSM == kSM_103, "Unsupported architecture");
    TLLM_CHECK_WITH_INFO(
        mDtypeQ == DATA_TYPE_E4M3 || mDtypeQ == DATA_TYPE_FP16 || mDtypeQ == DATA_TYPE_BF16, "Unsupported Q data type");
    TLLM_CHECK_WITH_INFO(mDtypeKv == DATA_TYPE_E2M1 || mDtypeKv == DATA_TYPE_E4M3 || mDtypeKv == DATA_TYPE_FP16
            || mDtypeKv == DATA_TYPE_BF16,
        "Unsupported Kv data type");
    TLLM_CHECK_WITH_INFO(mDtypeOut == DATA_TYPE_E2M1 || mDtypeOut == DATA_TYPE_E4M3 || mDtypeOut == DATA_TYPE_FP16
            || mDtypeOut == DATA_TYPE_BF16,
        "Unsupported Output data type");
    auto const [freeMemory, totalMemory] = tensorrt_llm::common::getDeviceMemoryInfo(false);
    mTotalDeviceMemory = totalMemory;
    TLLM_CHECK_WITH_INFO(mTotalDeviceMemory > 0, "Total device memory is invalid");
    mKernel = getTllmFmhaKernels(mDtypeQ, mDtypeKv, mDtypeOut, mSM);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TllmGenFmhaRunner::run(TllmGenFmhaRunnerParams const& runnerParams)
{
    mKernel->run(runnerParams);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TllmGenFmhaRunner::isSupported(TllmGenFmhaRunnerParams const& runnerParams) const
{
    return mKernel->checkIfKernelExist(runnerParams).first;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::pair<bool, std::string> TllmGenFmhaRunner::isSupportedWithInfo(TllmGenFmhaRunnerParams const& runnerParams) const
{
    return mKernel->checkIfKernelExist(runnerParams);
}

size_t TllmGenFmhaRunner::getTotalDeviceMemory() const
{
    return mTotalDeviceMemory;
}

} // namespace kernels
} // namespace tensorrt_llm
