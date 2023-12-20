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
#include "tensorrt_llm/kernels/onlineSoftmaxBeamsearchKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T, int MAX_K>
void topK_softMax_kernelLauncher(const T* log_probs, const T* bias, const FinishedState* finished,
    const int* sequence_lengths, float* cum_log_probs, float* output_log_probs, int** output_ids_ptr,
    void* temp_storage, const int temp_storage_size, BeamHypotheses* beam_hyps, const int batch_size,
    const int beam_width, const int vocab_size, const int* end_ids, const float* diversity_rates,
    const float* length_penalties, cudaStream_t stream);

#define CASE_K(MAX_K)                                                                                                  \
    topK_softMax_kernelLauncher<T, MAX_K>(log_probs, bias, finished, sequence_lengths, cum_log_probs,                  \
        output_log_probs, output_ids_ptr, temp_storage, temp_storage_size, beam_hyps, batch_size, beam_width,          \
        vocab_size, end_ids, diversity_rates, length_penalties, stream);                                               \
    break;

template <typename T>
void invokeTopkSoftMax(const T* log_probs, const T* bias, const FinishedState* finished, const int* sequence_lengths,
    float* cum_log_probs, float* output_log_probs, int** output_ids_ptr, void* temp_storage,
    const int temp_storage_size, BeamHypotheses* beam_hyps, const int batch_size, const int beam_width,
    const int vocab_size, const int* end_ids, const float* diversity_rates, const float* length_penalties,
    cudaStream_t stream)
{
    int log_beam_width(0);
    int recursor(beam_width - 1);
    while (recursor >>= 1)
        ++log_beam_width;

    switch (log_beam_width)
    {
    // 0 < beam_width <= 4
    case 0:        // 1, 2
    case 1:        // 3, 4
        CASE_K(4)
    case 2:        // 4 < beam_width <= 8
        CASE_K(8)
#ifndef FAST_BUILD // For fast build, skip case 3, 4, 5
    case 3:        // 9 < beam_width <= 16
        CASE_K(16)
    case 4:        // 16 < beam_width <= 32
        CASE_K(32)
    case 5:        // 32 < beam_width <= 64
        CASE_K(64)
#endif             // FAST_BUILD
    default:
        throw std::runtime_error(
            fmtstr("%s:%d Topk kernel of beam search does not support beam_width=%d", __FILE__, __LINE__, beam_width));
    }
}

#undef CASE_K

template void invokeTopkSoftMax<float>(const float* log_probs, const float* bias, const FinishedState* finished,
    const int* sequence_lengths, float* cum_log_probs, float* output_log_probs, int** output_ids_ptr, void* tmp_storage,
    const int temp_storage_size, BeamHypotheses* beam_hyps, const int batch_size, const int beam_width,
    const int vocab_size, const int* end_ids, const float* diversity_rates, const float* length_penalties,
    cudaStream_t stream);

template void invokeTopkSoftMax<half>(const half* log_probs, const half* bias, const FinishedState* finished,
    const int* sequence_lengths, float* cum_log_probs, float* output_log_probs, int** output_ids_ptr, void* tmp_storage,
    const int temp_storage_size, BeamHypotheses* beam_hyps, const int batch_size, const int beam_width,
    const int vocab_size, const int* end_ids, const float* diversity_rates, const float* length_penalties,
    cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
