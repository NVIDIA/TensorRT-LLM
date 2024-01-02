/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/banBadWords.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
__global__ void ban_bad_words(T* logits, const int** output_ids_ptr, const int** parent_ids_ptr, int batch_size,
    int beam_width, const int* bad_words, size_t bad_words_len, bool share_words, int vocab_size_padded,
    const int* sequence_lengths, const int max_seq_len)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y / beam_width;
    const int beam_idx = blockIdx.y % beam_width;

    const int* base_bad_words = share_words ? bad_words : bad_words + batch_idx * 2 * bad_words_len;
    const int* base_bad_words_offsets = base_bad_words + bad_words_len;

    if (id >= bad_words_len || base_bad_words_offsets[id] < 0)
    {
        return;
    }

    const int item_end = base_bad_words_offsets[id];
    const int item_start = (id > 0) ? base_bad_words_offsets[id - 1] : 0;
    const int item_size = item_end - item_start;

    /* The single-token case unconditionally bans the token */
    bool should_ban = item_size == 1;
    const int current_step{sequence_lengths[blockIdx.y]};
    /* Multi-token case and enough previously generated tokens to look for a match
     */
    if (item_size > 1 && current_step >= item_size - 1)
    {
        should_ban = true;
        int parent_id = beam_idx;
        const bool gather_beam = beam_width > 1;

        for (int token_idx = item_size - 2; token_idx >= 0; token_idx--)
        {
            const int previous_token
                = output_ids_ptr[batch_idx][parent_id * max_seq_len + current_step - (item_size - 1) + token_idx];

            if (previous_token != base_bad_words[item_start + token_idx])
            {
                should_ban = false;
                break;
            }
            if (gather_beam)
            {
                parent_id = parent_ids_ptr == nullptr
                    ? 0
                    : parent_ids_ptr[batch_idx][parent_id * max_seq_len + current_step - (item_size - 1) + token_idx];

                if (parent_id < 0 || parent_id >= beam_width)
                {
                    should_ban = false;
                    break;
                }
            }
        }
    }

    if (should_ban)
    {
        int banned_token = base_bad_words[item_end - 1];
        if (0 < banned_token && banned_token < vocab_size_padded)
        {
            logits[batch_idx * beam_width * vocab_size_padded + beam_idx * vocab_size_padded + banned_token]
                = static_cast<T>(-INFINITY);
        }
    }
}

template <typename T>
void invokeBanBadWords(T* logits, const int** output_ids_ptr, const int** parent_ids_ptr, int batch_size,
    int local_batch_size, int beam_width, const int* bad_words, bool share_words, size_t bad_words_len,
    int vocab_size_padded, const int* sequence_lengths, int max_seq_len, cudaStream_t stream)
{
    dim3 block, grid;
    constexpr size_t max_blocks{256};
    block.x = min(((bad_words_len + 32 - 1) / 32) * 32, max_blocks);
    grid.x = (bad_words_len + block.x - 1) / block.x;
    grid.y = local_batch_size * beam_width;

    ban_bad_words<<<grid, block, 0, stream>>>(logits, output_ids_ptr, parent_ids_ptr, batch_size, beam_width, bad_words,
        bad_words_len, share_words, vocab_size_padded, sequence_lengths, max_seq_len);
    sync_check_cuda_error();
}

template void invokeBanBadWords(half* logits, const int** output_ids_ptr, const int** parent_ids_ptr, int batch_size,
    int local_batch_size, int beam_width, const int* bad_words, bool share_words, size_t bad_words_len,
    int vocab_size_padded, const int* sequence_lengths, int max_seq_len, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeBanBadWords(__nv_bfloat16* logits, const int** output_ids_ptr, const int** parent_ids_ptr,
    int batch_size, int local_batch_size, int beam_width, const int* bad_words, bool share_words, size_t bad_words_len,
    int vocab_size_padded, const int* sequence_lengths, int max_seq_len, cudaStream_t stream);
#endif
template void invokeBanBadWords(float* logits, const int** output_ids_ptr, const int** parent_ids_ptr, int batch_size,
    int local_batch_size, int beam_width, const int* bad_words, bool share_words, size_t bad_words_len,
    int vocab_size_padded, const int* sequence_lengths, int max_seq_len, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
