/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/banRepeatNgram.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
__global__ void ban_repeat_ngram(T* logits, const int** output_ids_buf, const FinishedState* finished_buf,
    const int* parent_ids_buf, int batch_size, int beam_width, const int* no_repeat_ngram_size_buf, int id_offset,
    int vocab_size_padded, size_t step)
{
    /**
     * Find subsequences that match the last (ngram_size - 1) generated tokens. The next-tokens of those matching
     * subsequences should be banned from the next-token logits, such that existing N-grams in current generated
     * sequence will not be generated again.
     *
     * Note 1: no-repeat restriction is per-beam instead of across-beam.
     * Note 2: since beam search results are stored and retrieved by backtracking (parent_ids), the entire sequence for
     * the current can only be obtained by backtracking all the way back.
     * Note 3: for greedy search, actually a more efficient implementation can be adopted (since we're not constrained
     * by the beam backtrack retrieval, we can have all threads loading into shared mem in parallel AND use normal order
     * traversal). But for simplicity and consistency, greedy search and beam search implementation are kept the same.
     *
     * The overlap between adjacent threads indicates wasted global memory access. Used shared memory instead.
     * Shared memory benefit is more significant as ngram_size increases. Shared memory reuse is for
     * in-bound positions only. For leftside out-of-boundary tokens, access by global memory.
     */

    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_batch_idx = blockIdx.y / beam_width;
    const int beam_idx = blockIdx.y % beam_width;
    const bool beam_search = beam_width > 1;
    const int no_repeat_ngram_size = no_repeat_ngram_size_buf[local_batch_idx];

    // case 1: ngram_size == 0 --> this means no ngram limit
    // case 2: generated length must be greater than ngram_size to do ngram check
    if (no_repeat_ngram_size == 0 || step < no_repeat_ngram_size)
    {
        return;
    }

    // if the beam has already finished, skip ngram check
    if ((finished_buf != nullptr) && (finished_buf[id_offset + local_batch_idx * beam_width + beam_idx].isFinished()))
    {
        return;
    }

    // shared mem layout: per-thread token that each thread is mapped to, plus (ngram_size - 1) extra tokens beyond
    // block boundary, plus (ngram_size - 1) most recent generated tokens in the beam
    extern __shared__ int shared_tokens[];
    int shared_tokens_length = blockDim.x + no_repeat_ngram_size - 1;
    int* last_tokens = &shared_tokens[shared_tokens_length];
    int last_tokens_length = no_repeat_ngram_size - 1;

    // retrieve the entire beam by backtracking from last token to current token  (in reverse order)
    // single thread vs parallel thread is equivalent as it's bound by the longest iteration
    if (threadIdx.x == 0)
    {
        int parent_id = beam_idx;
        int start_record_idx = min(output_idx + shared_tokens_length, (int) step);
        int shared_token_idx = start_record_idx == step ? step - output_idx - 1 : shared_tokens_length - 1;
        int last_token_idx = last_tokens_length - 1;
        // write to shared mem in reverse order; boundary condition when thread block covers more than step

        for (int curr_idx = step - 1; curr_idx >= output_idx; curr_idx--)
        {
            if (last_token_idx >= 0)
            {
                last_tokens[last_token_idx--] = output_ids_buf[blockIdx.y][curr_idx * batch_size * beam_width
                    + id_offset + local_batch_idx * beam_width + parent_id];
            }

            // before reaching the part of current block, traverse only; after that, record the tokens
            if (curr_idx < start_record_idx)
            {
                shared_tokens[shared_token_idx--] = output_ids_buf[blockIdx.y][curr_idx * batch_size * beam_width
                    + id_offset + local_batch_idx * beam_width + parent_id];
            }

            if (beam_search)
            {
                parent_id = parent_ids_buf[curr_idx * batch_size * beam_width + id_offset + local_batch_idx * beam_width
                    + parent_id];
            }
        }
    }

    __syncthreads();

    if (output_idx > step - no_repeat_ngram_size)
    {
        return;
    }

    bool ban_ngram = true;

    // ngram check (in regular order)
    for (int ngram_idx = 0; ngram_idx < no_repeat_ngram_size - 1; ngram_idx++)
    {
        if (shared_tokens[threadIdx.x + ngram_idx] != last_tokens[ngram_idx])
        {
            ban_ngram = false;
            break;
        }
    }

    // erase banned next token's prob as -INF
    if (ban_ngram)
    {
        int banned_token = shared_tokens[threadIdx.x + no_repeat_ngram_size - 1]; // ban the last token in the ngram
        logits[local_batch_idx * beam_width * vocab_size_padded + beam_idx * vocab_size_padded + banned_token]
            = static_cast<T>(-INFINITY); // note: "logits" passed in is already with batchxbeam offset
    }
}

template <typename T>
void invokeBanRepeatNgram(T* logits, const int** output_ids_buf, const FinishedState* finished_buf,
    const int* parent_ids_buf, int batch_size, int local_batch_size, int beam_width,
    const int* no_repeat_ngram_size_buf, int id_offset, int vocab_size_padded, size_t step, cudaStream_t stream)
{
    // each input in the local batch can have different no_repeat_ngram_size. Use max for shmem allocation
    // getting the max of current batch and allocate shmem as needed is ideal. But here the ngram_buf is on GPU, while
    // this max value is on CPU for kernel launch. Instead of really finding the max and extra CPU-GPU memcpy, we simply
    // use a constant. In practice, ngram size is usually very small, like 3 or 4.
    int max_no_repeat_ngram_size = 32;

    // step (current generated length, except start token) is from 1 ~ max_seq_len
    dim3 block, grid;
    constexpr size_t max_blocks{256};
    block.x = min(((step + 32 - 1) / 32) * 32, max_blocks);
    grid.x = (step + block.x - 1) / block.x;
    grid.y = local_batch_size * beam_width;

    // dynamically allocate shared memory of int[blockDim + 2*(ngram_size - 1)], where ngram_size - 1 is for boundary
    // token's ngram and for most recent tokens
    ban_repeat_ngram<<<grid, block, (block.x + 2 * (max_no_repeat_ngram_size - 1)) * sizeof(int), stream>>>(logits,
        output_ids_buf, finished_buf, parent_ids_buf, batch_size, beam_width, no_repeat_ngram_size_buf, id_offset,
        vocab_size_padded, step);
    sync_check_cuda_error();
}

#define INVOKE_BAN_REPEAT_NGRAM(T)                                                                                     \
    template void invokeBanRepeatNgram(T* logits, const int** output_ids_buf, const FinishedState* finished_buf,       \
        const int* parent_ids_buf, int batch_size, int local_batch_size, int beam_width,                               \
        const int* no_repeat_ngram_size_buf, int id_offset, int vocab_size_padded, size_t step, cudaStream_t stream);

INVOKE_BAN_REPEAT_NGRAM(float)
INVOKE_BAN_REPEAT_NGRAM(half)
#ifdef ENABLE_BF16
INVOKE_BAN_REPEAT_NGRAM(__nv_bfloat16)
#endif
#undef INVOKE_BAN_REPEAT_NGRAM

} // namespace kernels

} // namespace tensorrt_llm
