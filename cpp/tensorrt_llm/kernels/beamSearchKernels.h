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

#include "tensorrt_llm/kernels/topkLastDim.h" // Air TopK

namespace tensorrt_llm
{
namespace kernels
{
static size_t constexpr kMaxBeamWidth = 1024;           // Max beam width supported in TRT-LLM now
static size_t constexpr kMaxBeamWidthForV1 = 8;         // Max beam width for V1 workflow (V2 for larger)
static size_t constexpr kMaxBeamWidthArrayLength = 8;   // Max length of beam width array of a request
static size_t constexpr kThreadForSmallBeamWidth = 256; // Max count of thread for stage 1 in V1 workflow
static size_t constexpr kMaxVPartStage1 = 128;          // Max vocab part count for stage 1 in V1 workflow

struct BeamHypotheses
{
    // clang-format off
    // MBS: max_batch_size, BS: batch_size, BM: beam_width, MSL: max_seq_length
    // %%: parameter name in file generation.py (python workflow)
    // Candidate beams: a beam which generates end_id or its sequence length reaches MSL
    // Candidate-Beam-Array (CBA): The arrays to place the candidate beams and related information

    // Scalar values
    bool bReturnNormedScore{false};         // return `normedScore` or `cumLogProbs`, always be `false` now
    size_t nMaxBatchSize{0};                // buildtime max batch size
    size_t nBatchSize{0};                   // runtime batch size
    size_t nBeamWidth{0};                   //
    size_t nMaxSeqLen{0};                   //
    size_t nVocabSize{0};                   // vocab_size_padded
    size_t nVPart{0};                       // Count of vocab_size_padded divided
    size_t nByteMaxSharedMemoryPerBlock{0};  // Device information
    size_t nByteSharedMemoryStage1{0};       // Dynamic shared memory size of stage 1
    size_t nByteSharedMemoryStage3{0};       // Static shared memory size of stage 3

    // Pointers from SamplingConfig
    float const* diversityRates{nullptr};           // [BS]
    float const* lengthPenalties{nullptr};          // [BS]
    int const* earlyStoppings{nullptr};             // [BS]

    // Pointers from input
    int const* inputLengths{nullptr};               // [BS, BM]         %% context_length
    int const* endIds{nullptr};                     // [BS, BM]         %% self.end_ids
    runtime::SizeType32 const* batchSlots{nullptr}; // [BS]

    // Pointers for output
    int* outputIds{nullptr};                        // [BS, BM, MSL]    %% self.output_ids                      only used in gather_tree
    float* logProbs{nullptr};                       // [BS, BM, MSL]    %% self.log_probs                       only used in gather_tree
    float* logProbsTiled{nullptr};                  // [MSL, MBS, BM]   %% self.log_probs_tiled
    int* sequenceLengths{nullptr};                  // [BS, BM]         %% self.sequence_length_buffer
    float* cumLogProbs{nullptr};                    // [BS, BM]         %% self.cum_log_probs

    // Pointers of CBA
    int* outputIdsCBA{nullptr};                     // [BS, BM*2, MSL]  %% self.beam_hyps_output_ids_cba
    float* logProbsCBA{nullptr};                    // [BS, BM*2, MSL]  %% self.beam_hyps_log_probs_cba
    int* sequenceLengthsCBA{nullptr};               // [BS, BM*2]       %% self.beam_hyps_seq_len_cba
    float* cumLogProbsCBA{nullptr};                 // [BS, BM*2]       %% self.beam_hyps_cum_log_probs_cba
    float* normedScoresCBA{nullptr};                // [BS, BM*2]       %% self.beam_hyps_normed_scores_cba
    int* numBeamsCBA{nullptr};                      // [BS]             %% self.beam_hyps_num_beams             number of beams in CBA
    float* minNormedScoresCBA{nullptr};             // [BS]             %% self.beam_hyps_min_normed_scores     worst score in CBA

    // Pointers related to beam search process, they are initialized in those two functions:
    // [gptDecoder.cpp] GptDecoder<T>::forward or [dynamicDecodeOp.cpp] FtDynamicDecode<T>::forward
    bool* batchDones{nullptr};                      // [BS]             %% self.beam_hyps_is_done               whether a whole batch is finished
    FinishedState* finished{nullptr};               // [BS*BM]          %% self.finished                        whether and how a beam is finished

    // Pointers for backtrack of the beams, they are relocated in [dynamicDecodeLayer.cpp] DynamicDecodeLayer<T>::prepareIdsPtrs
    int** outputIdsPtr{nullptr};                    // [BS][BM, MSL]    %% self.output_ids
    int** parentIdsPtr{nullptr};                    // [BS][BM, MSL]    %% self.parent_ids

    // Pointers for gather_tree(), read the unfinished beams from them and write to CBA for the final selection
    int const* outputIdsUnfinish{nullptr};          // [BS, BM, MSL]   %% self.output_ids
    int const* parentIdsUnfinish{nullptr};          // [BS, BM, MSL]   %% self.parent_ids

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

template <typename T, bool IS_V2>
void invokeTopkBeamSearch(T const* logProbs, T const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream);

template <typename T>
__global__ void addCumLogProbs(T* __restrict pStage1Probs, float const* __restrict cumLogProbs,
    FinishedState const* finished, int const* endIds, float const* diversityRates,
    runtime::SizeType32 const* batchSlots, size_t const nBS, size_t const nBM);

__global__ void gatherId(
    int const* __restrict pStage1Id, int* __restrict pStage2Id, size_t const nBS, size_t const nBM, size_t const nV);

} // namespace kernels
} // namespace tensorrt_llm
