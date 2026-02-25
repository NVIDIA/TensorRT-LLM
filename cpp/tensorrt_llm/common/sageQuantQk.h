/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#pragma once

#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

namespace tensorrt_llm::common
{

struct SageQuantQkParams
{
    int sumSeqLensQk{};
    int numHeads{};
    int headDim{};
    int tokenBlockSize{};
    bool kSmooth{false};
    void const* ptrQk{nullptr};
    void* ptrQkQuant{nullptr};
    kernels::Data_type inputType{kernels::DATA_TYPE_FP16};
    kernels::Data_type quantType{kernels::DATA_TYPE_E4M3};
    float* ptrQkScale{nullptr};
    float* ptrKMean{nullptr};
    int smCount{};
    cudaStream_t stream{};
};

void invokeSageQuantQk(SageQuantQkParams const& params);

} // namespace tensorrt_llm::common
