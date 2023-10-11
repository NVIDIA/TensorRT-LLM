/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/stopCriteriaKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{
__global__ void stop_words_criterion(const int** output_ids, const int** parent_ids, const int* stop_words,
    bool* finished, const int* sequence_lengths, size_t id_offset, size_t stop_words_len, int batch_size,
    int beam_width, int max_seq_len)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y / beam_width;
    const int beam_idx = blockIdx.y % beam_width;

    const int* base_stop_words = stop_words + batch_idx * 2 * stop_words_len;
    const int* base_offsets = base_stop_words + stop_words_len;

    if (id >= stop_words_len || base_offsets[id] < 0)
    {
        return;
    }

    const int item_end = base_offsets[id];
    const int item_start = (id > 0) ? base_offsets[id - 1] : 0;
    const int item_size = item_end - item_start;

    /* The single-token case unconditionally bans the token */
    bool should_stop = false;

    const int current_step
        = sequence_lengths[blockIdx.y] - 1; // need to minus 1 because the sequence_lengths is updated in this step
    /* Enough previously generated tokens to look for a match */
    if (current_step + 1 >= item_size)
    {
        should_stop = true;
        int parent_id = beam_idx;
        const bool gather_beam = beam_width > 1;

        for (int token_idx = item_size - 1; token_idx >= 0; token_idx--)
        {
            const int previous_token
                = output_ids[batch_idx][parent_id * max_seq_len + current_step - (item_size - 1) + token_idx];
            if (previous_token != base_stop_words[item_start + token_idx])
            {
                should_stop = false;
                break;
            }
            if (gather_beam)
            {
                parent_id = parent_ids == nullptr
                    ? 0
                    : parent_ids[batch_idx][parent_id * max_seq_len + current_step - (item_size - 1) + token_idx];

                if (parent_id < 0 || parent_id >= beam_width)
                {
                    should_stop = false;
                    break;
                }
            }
        }
    }

    if (should_stop)
    {
        finished[batch_idx * beam_width + beam_idx] = true;
    }
}

void invokeStopWordsCriterion(const int** output_ids, const int** parent_ids, const int* stop_words, bool* finished,
    const int* sequence_lengths, size_t id_offset, size_t stop_words_len, int batch_size, int beam_width,
    int max_seq_len, cudaStream_t stream)
{
    // Check if we have sampled a word from the stop_words list. If so, stop the
    // sequence.
    dim3 block, grid;
    constexpr size_t max_block_size{256};
    block.x = min(((stop_words_len + 32 - 1) / 32) * 32, max_block_size);
    grid.x = (stop_words_len + block.x - 1) / block.x;
    grid.y = batch_size * beam_width;

    stop_words_criterion<<<grid, block, 0, stream>>>(output_ids, parent_ids, stop_words, finished, sequence_lengths,
        id_offset, stop_words_len, batch_size, beam_width, max_seq_len);
    sync_check_cuda_error();
}

__global__ void length_criterion(bool* finished, int* finished_sum, const uint32_t* sequence_limit_length,
    const int* sequence_lengths, int batch_size, int beam_width)
{
    int thread_finished_count = 0;
    for (int index = threadIdx.x; index < batch_size * beam_width; index += blockDim.x)
    {
        const int batch_idx = index / beam_width;

        finished[index] |= sequence_lengths[index] >= sequence_limit_length[batch_idx];
        thread_finished_count += finished[index] ? 1 : 0;
    }

    if (finished_sum)
    {
        int block_finished_count = 0;
        if (blockDim.x <= 32)
        {
            block_finished_count = warpReduceSum(thread_finished_count);
        }
        else
        {
            block_finished_count = blockReduceSum(thread_finished_count);
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            finished_sum[0] = block_finished_count;
        }
    }
}

void invokeLengthCriterion(bool* finished, int* finished_sum, const uint32_t* sequence_limit_length,
    const int* sequence_lengths, int batch_size, int beam_width, cudaStream_t stream)
{
    // Check if we have attained the sequence length limit. If so, stop the
    // sequence. In addition, check if all sequences are stopped and return the
    // result in should_stop
    dim3 block{min(512, uint32_t(batch_size * beam_width))};
    dim3 grid{1};

    length_criterion<<<grid, block, 0, stream>>>(
        finished, finished_sum, sequence_limit_length, sequence_lengths, batch_size, beam_width);
    sync_check_cuda_error();
}

} // namespace kernels
} // namespace tensorrt_llm
