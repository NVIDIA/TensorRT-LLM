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

#include "tensorrt_llm/kernels/decodingCommon.h"
#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{
// We keep tracing `beam_width` beams during iterations, once a beam is finished,
// we record the ids and its normed score in output_ids_tgt and normed_scores
struct BeamHypotheses
{
    // BS:  batch_size
    // BM:  beam_width
    // mSL: max_seq_length
    // %%:  parameter name when we call [generation.py] dynamic_decoder.forward (python workflow)

    // Pointers initialized in these two functions below:
    // [gptDecoder.cpp] GptDecoder<T>::forward or [dynamicDecodeOp.cpp] FtDynamicDecode<T>::forward
    bool* is_done{nullptr};             // [BS]             %% self.beam_hyps_is_done
    float* cum_log_probs{nullptr};      // [BS, BM*2]       %% self.beam_hyps_cum_log_probs
    float* log_probs{nullptr};          // [BS, BM*2, mSL]  %% self.beam_hyps_log_probs
    float* min_normed_scores{nullptr};  // [BS]             %% self.beam_hyps_min_normed_scores
    float* normed_scores{nullptr};      // [BS, BM*2]       %% self.beam_hyps_normed_scores
    int* num_beams{nullptr};            // [BS]             %% self.beam_hyps_num_beams
    int* output_ids_tgt{nullptr};       // [BS, BM*2, mSL]  %% self.beam_hyps_output_ids_tgt
    int* sequence_lengths_tgt{nullptr}; // [BS, BM*2]       %% self.beam_hyps_sequence_lengths_tgt
    int const* input_lengths{nullptr};  // [BS*BM]          %% context_length

    // Pointers initialized in [onlineBeamSearchLayer.cu] invokeSoftMax:
    int const* end_ids{nullptr};        // [BS*BM]          %% self.end_ids
    FinishedState* finished;            // [BS*BM]          %% self.finished
    float* cum_log_probs_src{nullptr};  // [BS, BM]         %% self.cum_log_probs
    float* log_probs_src{nullptr};      // [mSL, BS, BM]    %% self.log_probs_tiled
    int* sequence_lengths_src{nullptr}; // [BS*BM]          %% self.sequence_length_buffer
    // These two pointers are relocated in [dynamicDecodeLayer.cpp] DynamicDecodeLayer<T>::prepareIdsPtrs
    int** output_ids_tgt_ptr{nullptr}; // [BS][BM, mSL]    %% self.output_ids
    int** parent_ids_tgt_ptr{nullptr}; // [BS][BM, mSL]    %% self.parent_ids

    float* diversity_rates{nullptr};   // [BS]             from SamplingConfig
    float* length_penalties{nullptr};  // [BS]             from SamplingConfig
    int* early_stoppings{nullptr};     // [BS]             from SamplingConfig

    // Pointers for function gatherTree
    int const* output_ids_src{nullptr}; //
    int const* parent_ids_src{nullptr}; //

    // Scalar values
    bool is_return_normed_score{true}; // return normed_cum_log_probs or cum_log_probs, always be true now
    int batch_size{0};                 //
    int beam_width{0};                 //
    int ite{0};                        // index of local_batch, always be 0 if pp_size==1
    int local_batch_size{0};           //
    int max_seq_len{0};                //
    int step{0};       // only used in [beamSearchTopkKernels.cu], always be 0 in [onlineSoftmaxBeamsearchKernels*.cu.h]
    int vocab_size{0}; // vocab_size_padded
};

template <typename T>
__device__ __forceinline__ T apply_length_penalty(T log_prob, int length, float length_penalty)
{
    // score = log(prob) / (length ^ length_penalty)
    if (length_penalty == 0.0f || length == 1)
    {
        return log_prob;
    }
    return log_prob / static_cast<T>(powf(static_cast<float>(length), length_penalty));
}

template <typename T>
void invokeTopkBeamSearch(void* workspace, size_t& workspace_size, T* log_probs, int* ids, BeamHypotheses* beam_hyps,
    bool const* finished, int const* sequence_lengths, int const batch_size, int const beam_width,
    int const vocab_size_padded_, const T diversity_rate, float const length_penalty, int const* end_ids,
    cudaStream_t stream);

void invokeInsertUnfinishedPath(BeamHypotheses beam_hyps, FinishedState const* finished, float const* cum_log_probs,
    int const batch_size, int const beam_width, cudaStream_t stream);
} // namespace kernels
} // namespace tensorrt_llm
