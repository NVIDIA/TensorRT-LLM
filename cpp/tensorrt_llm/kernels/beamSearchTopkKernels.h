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
    int* output_ids_tgt = nullptr;
    int* sequence_lengths_tgt = nullptr;
    float* cum_log_probs = nullptr;     // cum_log
    float* normed_scores = nullptr;     // cum_log / (length**length_penalty)
    float* log_probs = nullptr;         // log probs of each generated token
    float* min_normed_scores = nullptr; // record the min normed scores for each batch
    int* num_beams = nullptr;           // the number of finished beams we collect
    bool* is_done = nullptr;

    // Used to set inputs
    const int* output_ids_src;
    const int** output_ids_src_ptr;
    const int* parent_ids_src;
    const int** parent_ids_src_ptr;
    const int* sequence_lengths_src;
    const int* end_ids;
    const float* log_probs_src;
    const int* input_lengths;

    // some variables for kernels
    int step;
    int ite;
    int batch_size;
    int local_batch_size;
    int max_seq_len;
    float* length_penalties;

    bool early_stopping = true;
    bool is_return_normed_score = true; // return normed_cum_log_probs or cum_log_probs
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
