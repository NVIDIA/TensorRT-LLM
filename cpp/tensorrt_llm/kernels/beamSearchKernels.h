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
#include "tensorrt_llm/runtime/common.h"

namespace tensorrt_llm
{
namespace kernels
{
static constexpr int nMaxBeamWidth = 64; // max beam width supported now
static constexpr int nBlockSizeForSmallBeamWidth = 256;
static constexpr int nMaxVocabPartForStage1FastKernel = 128;

struct BeamHypotheses
{
    // clang-format off

    // MBS: max_batch_size, BS: batch_size, BM: beam_width, MSL: max_seq_length
    // %%: parameter name in file generation.py (python workflow)

    // Candidate beams: a beam which generates end_id or its sequence length reaches MSL
    // Candidate-Beam-Array (CBA): The arrays to place the candidate beams and related information

    // Scalar values
    bool bReturnNormedScore{false};     // return normed_score / cum_log_probs, useless yet
    int nMaxBatchSize{0};               // max batch size by model configuration
    int nBatchSize{0};                  // batch size by runtime input data
    int nBeamWidth{0};                  //
    int nMaxSeqLen{0};                  //
    int nVocabSize{0};                  // vocab_size_padded

    // Pointers from SamplingConfig
    float const* diversityRates{nullptr};   // [BS]
    float const* lengthPenalties{nullptr};  // [BS]
    int const* earlyStoppings{nullptr};     // [BS]

    // Pointers from input
    int const* inputLengths{nullptr};   // [BS, BM]         %% context_length
    int const* endIds{nullptr};         // [BS, BM]         %% self.end_ids
    runtime::SizeType32 const* batchSlots{nullptr}; // [BS]

    // Pointers for output
    int* outputIds{nullptr};            // [BS, BM, MSL]    %% self.output_ids                      only used in gather_tree
    float* logProbs{nullptr};           // [BS, BM, MSL]    %% self.log_probs                       only used in gather_tree
    float* logProbsTiled{nullptr};      // [MSL, MBS, BM]   %% self.log_probs_tiled
    int* sequenceLengths{nullptr};      // [BS, BM]         %% self.sequence_length_buffer
    float* cumLogProbs{nullptr};        // [BS, BM]         %% self.cum_log_probs

    // Pointers of CBA
    int* outputIdsCBA{nullptr};         // [BS, BM*2, MSL]  %% self.beam_hyps_output_ids_cba
    float* logProbsCBA{nullptr};        // [BS, BM*2, MSL]  %% self.beam_hyps_log_probs_cba
    int* sequenceLengthsCBA{nullptr};   // [BS, BM*2]       %% self.beam_hyps_seq_len_cba
    float* cumLogProbsCBA{nullptr};     // [BS, BM*2]       %% self.beam_hyps_cum_log_probs_cba
    float* normedScoresCBA{nullptr};    // [BS, BM*2]       %% self.beam_hyps_normed_scores_cba
    int* numBeamsCBA{nullptr};          // [BS]             %% self.beam_hyps_num_beams             number of beams in CBA
    float* minNormedScoresCBA{nullptr}; // [BS]             %% self.beam_hyps_min_normed_scores     worst score in CBA

    // Pointers related to beam search process, they are initialized in those two functions:
    // [gptDecoder.cpp] GptDecoder<T>::forward or [dynamicDecodeOp.cpp] FtDynamicDecode<T>::forward
    bool* batchDones{nullptr};          // [BS]             %% self.beam_hyps_is_done   whether a whole batch is finished
    FinishedState* finished{nullptr};   // [BS*BM]          %% self.finished            whether and how a beam is finished

    // Pointers for backtrack of the beams, they are relocated in [dynamicDecodeLayer.cpp] DynamicDecodeLayer<T>::prepareIdsPtrs
    int** outputIdsPtr{nullptr};        // [BS][BM, MSL]        %% self.output_ids
    int** parentIdsPtr{nullptr};        // [BS][BM, MSL]        %% self.parent_ids

    // Pointers for gather_tree(), read the unfinished beams from them and write to CBA for the final selection
    int const* outputIdsUnfinish{nullptr};  // [BS, BM, MSL]   %% self.output_ids
    int const* parentIdsUnfinish{nullptr};  // [BS, BM, MSL]   %% self.parent_ids

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
