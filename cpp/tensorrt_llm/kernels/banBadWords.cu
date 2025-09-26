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
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T>
__global__ void ban_bad_words(T* logits, TokenIdType const** output_ids_ptr, SizeType32 const** parent_ids_ptr,
    SizeType32 const* batch_slots, SizeType32 beam_width, TokenIdType const* const* bad_words_ptrs,
    SizeType32 const* bad_words_lens, SizeType32 vocab_size_padded, SizeType32 const* sequence_lengths,
    SizeType32 max_seq_len)
{
    auto const id = blockIdx.x * blockDim.x + threadIdx.x;
    auto const batch_idx = blockIdx.y / beam_width;
    auto const beam_idx = blockIdx.y % beam_width;
    auto const batch_slot = batch_slots != nullptr ? batch_slots[batch_idx] : batch_idx;
    auto const batch_beam_idx = batch_slot * beam_width + beam_idx;

    auto const* base_bad_words = bad_words_ptrs[batch_slot];
    auto const bad_words_len = bad_words_lens[batch_slot];
    auto const* base_bad_words_offsets = base_bad_words + bad_words_len;

    if (id >= bad_words_len || base_bad_words_offsets[id] < 0)
    {
        return;
    }

    auto const item_end = base_bad_words_offsets[id];
    auto const item_start = (id > 0) ? base_bad_words_offsets[id - 1] : 0;
    auto const item_size = item_end - item_start;

    /* The single-token case unconditionally bans the token */
    bool should_ban = item_size == 1;
    auto const current_step{sequence_lengths[batch_beam_idx]};
    /* Multi-token case and enough previously generated tokens to look for a match
     */
    if (item_size > 1 && current_step >= item_size - 1)
    {
        should_ban = true;
        auto parent_id = static_cast<SizeType32>(beam_idx);
        bool const gather_beam = beam_width > 1;

        for (auto token_idx = item_size - 2; token_idx >= 0; token_idx--)
        {
            auto const previous_token
                = output_ids_ptr[batch_slot][parent_id * max_seq_len + current_step - (item_size - 1) + token_idx];

            if (previous_token != base_bad_words[item_start + token_idx])
            {
                should_ban = false;
                break;
            }
            if (gather_beam)
            {
                parent_id = parent_ids_ptr == nullptr
                    ? SizeType32{0}
                    : parent_ids_ptr[batch_slot][parent_id * max_seq_len + current_step - (item_size - 1) + token_idx];

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
        auto banned_token = base_bad_words[item_end - 1];
        if (0 <= banned_token && banned_token < vocab_size_padded)
        {
            logits[batch_idx * beam_width * vocab_size_padded + beam_idx * vocab_size_padded + banned_token]
                = static_cast<T>(-INFINITY);
        }
    }
}

template <typename T>
void invokeBanBadWords(T* logits, TokenIdType const** output_ids_ptr, SizeType32 const** parent_ids_ptr,
    SizeType32 const* batch_slot, SizeType32 batch_size, SizeType32 beam_width, TokenIdType const* const* bad_words,
    SizeType32 const* bad_words_lens, SizeType32 max_bad_words_len, SizeType32 vocab_size_padded,
    SizeType32 const* sequence_lengths, SizeType32 max_seq_len, cudaStream_t stream)
{
    dim3 block, grid;
    constexpr SizeType32 max_blocks{256};
    block.x = min(((max_bad_words_len + 32 - 1) / 32) * 32, max_blocks);
    grid.x = (max_bad_words_len + block.x - 1) / block.x;
    grid.y = batch_size * beam_width;

    ban_bad_words<<<grid, block, 0, stream>>>(logits, output_ids_ptr, parent_ids_ptr, batch_slot, beam_width, bad_words,
        bad_words_lens, vocab_size_padded, sequence_lengths, max_seq_len);
    sync_check_cuda_error(stream);
}

template void invokeBanBadWords(half* logits, TokenIdType const** output_ids_ptr, SizeType32 const** parent_ids_ptr,
    SizeType32 const* batch_slot, SizeType32 batch_size, SizeType32 beam_width, TokenIdType const* const* bad_words,
    SizeType32 const* bad_words_lens, SizeType32 max_bad_words_len, SizeType32 vocab_size_padded,
    SizeType32 const* sequence_lengths, SizeType32 max_seq_len, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeBanBadWords(__nv_bfloat16* logits, TokenIdType const** output_ids_ptr,
    SizeType32 const** parent_ids_ptr, SizeType32 const* batch_slot, SizeType32 batch_size, SizeType32 beam_width,
    TokenIdType const* const* bad_words, SizeType32 const* bad_words_lens, SizeType32 max_bad_words_len,
    SizeType32 vocab_size_padded, SizeType32 const* sequence_lengths, SizeType32 max_seq_len, cudaStream_t stream);
#endif
template void invokeBanBadWords(float* logits, TokenIdType const** output_ids_ptr, SizeType32 const** parent_ids_ptr,
    SizeType32 const* batch_slot, SizeType32 batch_size, SizeType32 beam_width, TokenIdType const* const* bad_words,
    SizeType32 const* bad_words_lens, SizeType32 max_bad_words_len, SizeType32 vocab_size_padded,
    SizeType32 const* sequence_lengths, SizeType32 max_seq_len, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
