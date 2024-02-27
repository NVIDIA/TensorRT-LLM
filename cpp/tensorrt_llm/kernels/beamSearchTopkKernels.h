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

// In original beam search implementation, if a beam is finished, we set it as
// finished and only continue to do beam search on remain beams (namely,
// beam_width - 1 beams in next step)
//
// In this implementation, when a beam is finished, we trace the path and record
// it in output_ids_tgt, and also record the normalized scores. And the beam
// search continue to use `beam_width` beams in next step.
//
// After we collect `beam_width` beams, we will sort them by their norm_scores.
struct BeamHypotheses
{
    // TODO: simplify the pointers
    // Pointers initialized in function prepareOutputs in gptDecoder.cpp
    bool* is_done{nullptr};             // [batchSize], whether the batch is finished
    const int* input_lengths{nullptr};  // [batchSize]
    float* cum_log_probs{nullptr};      // [batchSize, 2 * beamWidth], outputs.cum_log_probs->template getPtr<float>()
    float* log_probs{nullptr};          // [batchSize, 2 * beamWidth, maxSeqLen], not used?
    float* min_normed_scores{nullptr};  // [batchSize], worst normed scores for each batch
    float* normed_scores{nullptr};      // [batchSize, 2 * beamWidth], cum_log / (length ^ length_penalty)
    int* num_beams{nullptr};            // [batchSize], count of finished beams for each batch
    int* output_ids_tgt{nullptr};       // [batchSize, 2 * beamWidth, maxSeqLen],
    int* sequence_lengths_tgt{nullptr}; // [batchSize, 2 * beamWidth], different from sequence_lengths_src

    // Pointers initialized in function invokeSoftMax in onlineBeamSearchLayer.cu
    const int* end_ids{nullptr};             // get from SoftmaxParams
    const int* output_ids_src{nullptr};      // for gatherTree
    const int* parent_ids_src{nullptr};      // for gatherTree
    const int** output_ids_src_ptr{nullptr}; // get from BeamSearchOutputParams for reading
    const int** parent_ids_src_ptr{nullptr}; // get from BeamSearchOutputParams for reading
    float* log_probs_src{nullptr};           // get from outputs.output_log_probs
    int* sequence_lengths_src{nullptr};      // get from BeamSearchOutputParams
    // For reading in function invokeTopkSoftMax but reading and writing in function invokeUpdate
    int** output_ids_tgt_ptr{nullptr}; // get from BeamSearchOutputParams for writing
    int** parent_ids_tgt_ptr{nullptr}; // get from BeamSearchOutputParams for writing

    // Other scalar values and buffers
    int batch_size{0};
    int beam_width{0};
    int ite{0};
    int local_batch_size{0};
    int max_seq_len{0};
    int step{0}; // useless in online version of beam search
    int vocab_size{0};
    float* diversity_rates{nullptr};
    float* length_penalties{nullptr};
    int* early_stoppings{nullptr};
    bool is_return_normed_score{true}; // return normed_cum_log_probs or cum_log_probs
};

template <typename T>
void invokeTopkBeamSearch(void* workspace, size_t& workspace_size, T* log_probs, int* ids, BeamHypotheses* beam_hyps,
    const bool* finished, const int* sequence_lengths, const int batch_size, const int beam_width,
    const int vocab_size_padded_, const T diversity_rate, const float length_penalty, const int* end_ids,
    cudaStream_t stream);

template <typename T>
void invokeTileEncoderResults(T* tiled_encoder_output, int* tiled_encoder_sequence_length, const T* encoder_output,
    const int* encoder_sequence_length, const size_t batch_size, const size_t beam_width, const size_t mem_max_seq_len,
    const size_t d_model, cudaStream_t stream);

void invokeInsertUnfinishedPath(BeamHypotheses beam_hyps, const FinishedState* finished, const float* cum_log_probs,
    const int batch_size, const int beam_width, cudaStream_t stream);

void invokeCopyBatchMajorToGeneralPtr(
    void* output_ids_ptr, int* output_ids, int batch_size, int beam_width, int max_seq_len, cudaStream_t stream);

void invokeCopyGeneralPtrToBatchMajor(
    int* output_ids, void* output_ids_ptr, int batch_size, int beam_width, int max_seq_len, cudaStream_t stream);

void invokeSeqlenMajorToBatchMajor(
    int* batchMajoredIds, int* seqlenMajorIds, int batch_size, int beam_width, int max_seq_len, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
