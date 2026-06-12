/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <cuda_runtime.h>
#include <string>

#include "fmhaKernels.h"
#include "fmhaRunnerParams.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// TllmGenFmhaSelectedKernel is declared in fmhaKernels.h so the kernel-side
// probe helper can return it without introducing a fmhaRunner.h -> fmhaKernels.h
// circular include.

class TllmGenFmhaRunner
{
public:
    // Constructor.
    explicit TllmGenFmhaRunner(Data_type dtypeQ, Data_type dtypeK, Data_type dtypeV, Data_type dtypeOut,
        int numEltsPerSageAttnBlkQ = 0, int numEltsPerSageAttnBlkK = 0, int numEltsPerSageAttnBlkP = 0,
        int numEltsPerSageAttnBlkV = 0);

    TllmGenFmhaRunner() = default;

    // Check if fmha is supported.
    bool isSupported(TllmGenFmhaRunnerParams const& runnerParams) const;

    // Check if fmha is supported with additional info.
    std::pair<bool, std::string> isSupportedWithInfo(TllmGenFmhaRunnerParams const& runnerParams) const;

    // Get the total device memory.
    size_t getTotalDeviceMemory() const;

    // Run the fmha kernel.
    void run(TllmGenFmhaRunnerParams const&);

#if defined(TLLM_FMHA_TEST_HOOKS)
    // Test-only: probe which cubin (if any) the static-lib autotuner would
    // select for these params, WITHOUT launching. Lets a unit test assert the
    // resolved cubin function name + grouped flags from hashFromFmhaOptions /
    // mFunctions.find.
    //
    // Defined inline so the symbol does not need to be exported from the
    // tensorrt_llm shared library. The test target defines
    // TLLM_FMHA_TEST_HOOKS in its CMakeLists; production builds compile this
    // method out entirely.
    TllmGenFmhaSelectedKernel probeKernelSelectionForTesting(TllmGenFmhaRunnerParams const& runnerParams) const
    {
        return mKernel->probeKernelSelectionForTesting(runnerParams);
    }
#endif // TLLM_FMHA_TEST_HOOKS

private:
    // The input/output datatype.
    Data_type mDtypeQ, mDtypeK, mDtypeV, mDtypeOut;
    // The SM version.
    int mSM;
    // The total device memory.
    size_t mTotalDeviceMemory;
    // The class that stores all the kernels.
    TllmGenFmhaKernel* mKernel;
    // SageAttention extensions.
    int mNumEltsPerSageAttnBlkQ;
    int mNumEltsPerSageAttnBlkK;
    int mNumEltsPerSageAttnBlkP;
    int mNumEltsPerSageAttnBlkV;
};

} // namespace kernels

TRTLLM_NAMESPACE_END
