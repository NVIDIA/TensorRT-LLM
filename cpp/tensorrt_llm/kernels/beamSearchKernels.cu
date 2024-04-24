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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/beamSearchKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T, int PAD_K>
void topKSoftMaxKernelLauncher(
    T const* logits, T const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream);

#define CASE_K(PAD_K)                                                                                                  \
    topKSoftMaxKernelLauncher<T, PAD_K>(logits, bias, workspace, bh, stream);                                          \
    break;

template <typename T>
void invokeTopkSoftMax(T const* logits, T const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream)
{
    switch (padToNextPowerOfTwo(bh.nBeamWidth)) // PAD_K must be a compilation-time constant
    {
    case 1:
    case 2:
    case 4:        // 0 < beam_width <= 4
        CASE_K(4)
    case 8:        // 4 < beam_width <= 8
        CASE_K(8)
#ifndef FAST_BUILD // For fast build, skip case 3, 4, 5
    case 16:       // 9 < beam_width <= 16
        CASE_K(16)
    case 32:       // 16 < beam_width <= 32
        CASE_K(32)
    case 64:       // 32 < beam_width <= 64
        CASE_K(64)
#endif             // FAST_BUILD
    default:
        throw std::runtime_error(
            fmtstr("%s:%d Maximum beam width supported for beam search (%d) is larger than beam_width now use (%d)",
                __FILE__, __LINE__, nMaxBeamWidth, bh.nBeamWidth));
    }
}

#undef CASE_K

template void invokeTopkSoftMax<float>(
    float const* logits, float const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream);

template void invokeTopkSoftMax<half>(
    half const* logits, half const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
