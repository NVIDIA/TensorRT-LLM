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

namespace tensorrt_llm
{
namespace kernels
{
static constexpr int nMaxBeamWidth = 64; // max beam width supported now
static constexpr int nSmallTopKBlockSize = 256;
static constexpr int nSmallTopKMaxVocParts = 128;

struct BeamHypotheses
{
    // clang-format off

    // BS: batch_size, BM: beam_width, mSL: max_seq_length
    // %%: parameter name when dynamic_decoder.forward() / gather_tree() are called in [generation.py] (python workflow)

    // Candidate beams: When a beam generates end_id or its sequence length reaches mSL, it becomes a candidate beam to be selected finally.
    // Candidate-Beam-Array (CBA): Arrays (size: BM*2) to place the candidate beams and related information

    // Scalar values
    bool is_return_normed_score{true}; // return normed_score / cum_log_probs, useless yet
    int batch_size{0};                 //
    int beam_width{0};                 //
    int ite{0};                        // index of local_batch, always be 0 when pp_size==1
    int local_batch_size{0};           //
    int max_seq_len{0};                //
    int vocab_size{0};                 // vocab_size_padded

    // Pointers from SamplingConfig
    float const* diversity_rates{nullptr};  // [BS]
    float const* length_penalties{nullptr}; // [BS]
    int const* early_stoppings{nullptr};    // [BS]

    // Pointers from input
    int const* input_lengths{nullptr};  // [BS, BM]          %% context_length
    int const* end_ids{nullptr};        // [BS, BM]          %% self.end_ids

    // Pointers for output
    int* final_output_ids{nullptr};     // [BS, BM, mSL]    %% self.output_ids
    float* log_probs{nullptr};          // [mSL, BS, BM]    %% self.log_probs_tiled
    int* seq_len{nullptr};              // [BS, BM]         %% self.sequence_length_buffer
    float* cum_log_probs{nullptr};      // [BS, BM]         %% self.cum_log_probs

    // Pointers of CBA
    int* output_ids_cba{nullptr};       // [BS, BM*2, mSL]  %% self.beam_hyps_output_ids_cba
    float* log_probs_cba{nullptr};      // [BS, BM*2, mSL]  %% self.beam_hyps_log_probs_cba
    int* seq_len_cba{nullptr};          // [BS, BM*2]       %% self.beam_hyps_seq_len_cba
    float* cum_log_probs_cba{nullptr};  // [BS, BM*2]       %% self.beam_hyps_cum_log_probs_cba
    float* normed_scores_cba{nullptr};  // [BS, BM*2]       %% self.beam_hyps_normed_scores_cba
    int* num_beams{nullptr};            // [BS]             %% self.beam_hyps_num_beams             number of beams in CBA
    float* min_normed_scores{nullptr};  // [BS]             %% self.beam_hyps_min_normed_scores     worst score in CBA

    // Pointers related to beam search process, they are initialized in those two functions:
    // [gptDecoder.cpp] GptDecoder<T>::forward or [dynamicDecodeOp.cpp] FtDynamicDecode<T>::forward
    bool* is_done{nullptr};             // [BS]             %% self.beam_hyps_is_done   whether a whole batch is finished
    FinishedState* finished;            // [BS*BM]          %% self.finished            whether and how a beam is finished

    // Pointers for backtrack of the beams, they are relocated in [dynamicDecodeLayer.cpp] DynamicDecodeLayer<T>::prepareIdsPtrs
    int** output_ids_ptr{nullptr};  // [BS][BM, mSL]    %% self.output_ids
    int** parent_ids_ptr{nullptr};  // [BS][BM, mSL]    %% self.parent_ids

    // Pointers for gather_tree(), read the unfinished beams from them and write to CBA for the final selection
    int const* output_ids_src{nullptr};  // [BS, BM, mSL]   %% self.output_ids
    int const* parent_ids_src{nullptr};  // [BS, BM, mSL]   %% self.parent_ids

    // clang-format on
};

__inline__ int padToNextPowerOfTwo(int const n)
{
    // Pad n up to the nearest power of 2
    int recursor = n - 1;
    int res = 2;
    while (recursor >>= 1)
        res <<= 1;
    return res;
}

template <typename T>
__device__ __forceinline__ T applyLengthPenalty(T const log_prob, int const length, float const length_penalty)
{
    // score = log(prob) / (length ^ length_penalty)
    if (length_penalty == 0.0f || length == 1)
    {
        return log_prob;
    }
    return log_prob / static_cast<T>(powf(static_cast<float>(length), length_penalty));
}

template <typename T>
void invokeTopkSoftMax(T const* logits, T const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
