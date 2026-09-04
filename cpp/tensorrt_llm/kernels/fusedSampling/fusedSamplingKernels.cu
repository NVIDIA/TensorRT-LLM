/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "fusedSamplingKernels.h"
#include "fusedSamplingKernelsInternal.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
void invokeFusedSampling(FusedSamplingParams const& params, cudaStream_t stream)
{
    if (fusedSampling::shouldUseMultiCta(params))
    {
        fusedSampling::launchFusedSamplingMultiCta<T>(params, stream);
        return;
    }

    fusedSampling::launchFusedSamplingSingleCta<T>(params, stream);
}

template void invokeFusedSampling<float>(FusedSamplingParams const&, cudaStream_t);
template void invokeFusedSampling<__half>(FusedSamplingParams const&, cudaStream_t);
template void invokeFusedSampling<__nv_bfloat16>(FusedSamplingParams const&, cudaStream_t);

} // namespace kernels
} // namespace tensorrt_llm
