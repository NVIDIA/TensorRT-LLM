/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "gptKernels.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tensorrt_llm
{

namespace kernels
{

struct gatherTreeParam
{
    // TODO rename the parameters
    int* beams = nullptr;            // [batch_size, beam_width, max_seq_len], workspace to put intermediate output_ids
    int* sequence_lengths = nullptr; // [batch_size, beam_width], total lengths of each query
    int max_sequence_length_final_step = 0;
    const int* input_lengths = nullptr; // [batch_size, beam_width]
    // response input lengths (used to slice the ids during postprocessing)
    int* response_input_lengths = nullptr;
    int max_seq_len = 0;
    int batch_size = 0;
    int beam_width = 0;
    const int* step_ids = nullptr;   // [max_seq_len, batch_size, beam_width]
    const int* parent_ids = nullptr; // [max_seq_len, batch_size, beam_width]
    const int* end_tokens = nullptr; // [batch_size], end token ids of each query
    int* output_ids = nullptr;       // the buffer to put finalized ids
    cudaStream_t stream;
    float* cum_log_probs = nullptr;  // [batch_size, beam_width]
    float length_penalty = 1.0f;     // on cpu
};

/*
Do gatherTree on beam search to get final result.
*/
void invokeGatherTree(gatherTreeParam param);

void invokeFinalize(int* output_ids, int* sequence_lengths, float* cum_log_probs, float* output_log_probs,
    const int* topk_output_ids, const int* topk_sequence_lengths, const float* scores, const float* topk_cum_log_probs,
    const float* topk_log_probs, const int* num_beams, const int* input_lengths, const int beam_width,
    const int max_seq_len, const int batch_size, cudaStream_t stream);

void invokeInitializeOutput(int* output_ids, const int* end_ids, int batch_beam, int max_seq_len, cudaStream_t stream);

void invokeCopyNextStepIds(int* next_step_ids, int** output_ids_ptr, const int* sequence_lengths, int batch_size,
    int beam_width, int max_seq_len, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
