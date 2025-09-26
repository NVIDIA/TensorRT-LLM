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

#pragma once

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "fused_multihead_attention_common.h"
#include "fused_multihead_attention_v2.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tmaDescriptor.h"

namespace tensorrt_llm
{
namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

// Workflow of fmha runner:
// 1. check if FMHA kernels are supported statically.
// 2. construct FMHA runner object with the fixed params.
// 3. run the kernel (with all needed device pointers).
class FusedMHARunnerV2
{
public:
    // Constructor.
    FusedMHARunnerV2(MHARunnerFixedParams fixedParams);

    // Deconstructor.
    ~FusedMHARunnerV2() = default; // for pimpl

    // Check if any fmha kernel meets the requirements.
    bool isFmhaSupported();

    // Does FMHA need a separate Q and Kv input ?
    bool isSeparateQAndKvInput() const
    {
        return mFixedParams.attentionInputLayout != AttentionInputLayout::PACKED_QKV;
    }

    // Run the fmha kernel.
    void run(MHARunnerParams runnerParams);

private:
    // Set the kernel params.
    void setupKernelParams(MHARunnerParams runnerParams);

    // Set the launch params to select kernels.
    void setupLaunchParams(MHARunnerParams runnerParams);

    // Set the tma descriptors.
    void setTmaDescriptors(MHARunnerParams runnerParams);

    // Check if it is a valid sequence length (only used by non-flash-attention kernels).
    bool isValidS(int s) const;

    // Get the kernel sequence that support the max sequence length (only used by non-flash-attention kernels).
    int getSFromMaxSeqLen(int const max_seq_len) const;

private:
    // The attention fixed params (mostly related to the attention structure).
    MHARunnerFixedParams mFixedParams;
    // The attention input params (runtime-known parameters).
    MHARunnerParams mRunnerParams;
    // The launch params to select the specific fmha kernel.
    Launch_params mLaunchParams;
    // The kernel params.
    Fused_multihead_attention_params_v2 mKernelParams;
    // The SM version.
    int mSM = tensorrt_llm::common::getSMVersion();
    // The multiple processor count.
    int mMultiProcessorCount;
    // The L2 cache size.
    int mDeviceL2CacheSize;
    // The total device memory.
    size_t mTotalDeviceMemory;
    // The class that stores all the kernels.
    FusedMultiHeadAttentionXMMAKernelV2 const* xmmaKernel;
};

} // namespace kernels
} // namespace tensorrt_llm
