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

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/kernels/beamSearchTopkKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

__global__ void insertUnfinishedPath(BeamHypotheses beam_hyps, FinishedState const* finished,
    float const* cum_log_probs, int const batch_size, int const beam_width)
{
    int const bid = blockIdx.x;
    int const tgt_start_idx = beam_hyps.num_beams[bid];
    int const max_seq_len{beam_hyps.max_seq_len};
    float const length_penalty{beam_hyps.length_penalties == nullptr ? 1.0f : beam_hyps.length_penalties[bid]};
    if (beam_hyps.is_done[bid])
    {
        return;
    }
    for (int beam_idx = 0; beam_idx < beam_width; beam_idx++)
    {
        if (threadIdx.x == 0)
        {
            int const src_beam_idx = bid * beam_width + beam_idx;
            int const tgt_beam_idx = bid * beam_width * 2 + beam_idx + tgt_start_idx;

            int const last_token_idx = beam_hyps.sequence_lengths_src[src_beam_idx] - 1;

            beam_hyps.output_ids_tgt[tgt_beam_idx * max_seq_len + last_token_idx]
                = beam_hyps.output_ids_src[src_beam_idx * max_seq_len + last_token_idx];
            if (beam_hyps.log_probs != nullptr && beam_hyps.log_probs_src != nullptr)
            {
                beam_hyps.log_probs[tgt_beam_idx * max_seq_len + last_token_idx]
                    = beam_hyps.log_probs_src[last_token_idx * batch_size * beam_width + src_beam_idx];
            }
            int prev_id = beam_hyps.parent_ids_src[src_beam_idx * max_seq_len + last_token_idx];
            for (int token_idx = last_token_idx - 1; token_idx >= 0; token_idx--)
            {
                // output_ids_tgt need to use max_seq_len + 1 because its shape is
                // [bs, beam_width, max_seq_len + 1]
                beam_hyps.output_ids_tgt[tgt_beam_idx * max_seq_len + token_idx]
                    = beam_hyps.output_ids_src[bid * beam_width * max_seq_len + prev_id * max_seq_len + token_idx];
                if (beam_hyps.log_probs != nullptr && beam_hyps.log_probs_src != nullptr)
                {
                    beam_hyps.log_probs[tgt_beam_idx * max_seq_len + token_idx]
                        = beam_hyps.log_probs_src[token_idx * batch_size * beam_width + bid * beam_width + prev_id];
                }
                prev_id = beam_hyps.parent_ids_src[bid * beam_width * max_seq_len + prev_id * max_seq_len + token_idx];
            }
            beam_hyps.sequence_lengths_tgt[tgt_beam_idx] = last_token_idx + 1;

            // TODO huggingface uses total length to normalize the scores, instead of number of generated tokens.
            // Check that is it reasonable or not.
            beam_hyps.normed_scores[tgt_beam_idx] = apply_length_penalty(cum_log_probs[src_beam_idx],
                finished[src_beam_idx].isFinished() ? last_token_idx + 1 : last_token_idx, length_penalty);
            beam_hyps.cum_log_probs[tgt_beam_idx] = cum_log_probs[src_beam_idx];

            beam_hyps.num_beams[bid]++;
        }
    }
}

void invokeInsertUnfinishedPath(BeamHypotheses beam_hyps, FinishedState const* finished, float const* cum_log_probs,
    int const batch_size, int const beam_width, cudaStream_t stream)
{
    insertUnfinishedPath<<<batch_size, 256, 0, stream>>>(beam_hyps, finished, cum_log_probs, batch_size, beam_width);
}
} // namespace kernels
} // namespace tensorrt_llm
