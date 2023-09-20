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
void topK_softMax_kernelLauncher(const T* log_probs, const T* bias, const bool* finished, const int* sequence_lengths,
    float* cum_log_probs, float* output_log_probs, int** output_ids_ptr, void* temp_storage,
    const int temp_storage_size, BeamHypotheses* beam_hyps, const int batch_size, const int beam_width,
    const int vocab_size, const int* end_ids, T diversity_rate, const float length_penalty, cudaStream_t stream);

#define CASE_K(K, MAX_K)                                                                                               \
    case K ... MAX_K:                                                                                                  \
        topK_softMax_kernelLauncher<T, MAX_K>(log_probs, bias, finished, sequence_lengths, cum_log_probs,              \
            output_log_probs, output_ids_ptr, temp_storage, temp_storage_size, beam_hyps, batch_size, beam_width,      \
            vocab_size, end_ids, diversity_rate, length_penalty, stream);                                              \
        break;

template <typename T>
void invokeTopkSoftMax(const T* log_probs, const T* bias, const bool* finished, const int* sequence_lengths,
    float* cum_log_probs, float* output_log_probs, int** output_ids_ptr, void* temp_storage,
    const int temp_storage_size, BeamHypotheses* beam_hyps, const int batch_size, const int beam_width,
    const int vocab_size, const int* end_ids, const float diversity_rate, const float length_penalty,
    cudaStream_t stream)
{
    switch (beam_width)
    {
        CASE_K(1, 4);
        CASE_K(5, 8);
        CASE_K(9, 16);
        CASE_K(17, 32);
        CASE_K(33, 64);
    default: throw std::runtime_error(fmtstr("Topk kernel of beam search does not support beam_width=%d", beam_width));
    }
}

#undef CASE_K

template void invokeTopkSoftMax<float>(const float* log_probs, const float* bias, const bool* finished,
    const int* sequence_lengths, float* cum_log_probs, float* output_log_probs, int** output_ids_ptr, void* tmp_storage,
    const int temp_storage_size, BeamHypotheses* beam_hyps, const int batch_size, const int beam_width,
    const int vocab_size, const int* end_ids, const float diversity_rate, const float length_penalty,
    cudaStream_t stream);

template void invokeTopkSoftMax<half>(const half* log_probs, const half* bias, const bool* finished,
    const int* sequence_lengths, float* cum_log_probs, float* output_log_probs, int** output_ids_ptr, void* tmp_storage,
    const int temp_storage_size, BeamHypotheses* beam_hyps, const int batch_size, const int beam_width,
    const int vocab_size, const int* end_ids, const float diversity_rate, const float length_penalty,
    cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
