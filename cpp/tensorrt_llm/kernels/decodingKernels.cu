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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/decodingKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{

namespace kernels
{

__global__ void gatherTree(gatherTreeParam param)
{
    for (int batchbeam_idx = blockIdx.x * blockDim.x + threadIdx.x; batchbeam_idx < param.batch_size * param.beam_width;
         batchbeam_idx += gridDim.x * blockDim.x)
    {
        const int batch = batchbeam_idx / param.beam_width;
        const int beam = batchbeam_idx % param.beam_width;
        const int input_len = param.input_lengths == nullptr ? 0 : param.input_lengths[batchbeam_idx];

        const int* parent_ids = param.parent_ids;
        const int* step_ids = param.step_ids;

        // TODO optimize the reduce_max operation for large beam_width
        int max_len = -1;
        bool update_response_input_length = param.response_input_lengths != nullptr;
        // int selected_beam_index = 0;
        for (int beam_idx = 0; beam_idx < param.beam_width; beam_idx++)
        {
            int tmp_len = param.sequence_lengths[batch * param.beam_width + beam_idx]
                + param.max_sequence_length_final_step - 1;
            param.sequence_lengths[batch * param.beam_width + beam_idx] = tmp_len;
            if (update_response_input_length)
            {
                param.response_input_lengths[batch * param.beam_width + beam_idx] = input_len;
            }
            if (tmp_len > max_len)
            {
                max_len = tmp_len;
            }
        }
        const int max_seq_len_b = min(param.max_seq_len, max_len);
        if (max_seq_len_b <= 0)
        {
            continue;
        }

        const int initial_tgt_ix
            = batch * param.beam_width * param.max_seq_len + beam * param.max_seq_len + max_seq_len_b - 1;
        const int initial_parent_ix
            = batch * param.beam_width * param.max_seq_len + beam * param.max_seq_len + max_seq_len_b - 1;
        param.output_ids[initial_tgt_ix] = __ldg(step_ids + initial_parent_ix);
        int parent = parent_ids == nullptr ? 0 : __ldg(parent_ids + initial_parent_ix) % param.beam_width;
        bool found_bad = false;

        for (int level = max_seq_len_b - 2; level >= 0; --level)
        {
            const int level_beam_ix = batch * param.beam_width * param.max_seq_len + beam * param.max_seq_len + level;
            const int level_parent_ix
                = batch * param.beam_width * param.max_seq_len + parent * param.max_seq_len + level;
            if (parent < 0 || parent > param.beam_width)
            {
                param.output_ids[level_beam_ix] = param.end_tokens[batch];
                parent = -1;
                found_bad = true;
            }
            else
            {
                param.output_ids[level_beam_ix] = __ldg(step_ids + level_parent_ix);
                parent = parent_ids == nullptr ? 0 : __ldg(parent_ids + level_parent_ix) % param.beam_width;
            }
        }
        // set the padded part as end_token
        // input_len
        for (int index = max_len; index < param.max_seq_len; ++index)
        {
            param.output_ids[batch * param.beam_width * param.max_seq_len + beam * param.max_seq_len + index]
                = param.end_tokens[batch];
        }

        // Not necessary when using a BeamSearchDecoder, but necessary
        // when a user feeds in possibly broken trajectory (i.e., non-eos
        // entries in a beam following eos entries).
        if (!found_bad)
        {
            bool finished = false;
            // skip the step 0 because it is often the start token
            int start_step = 1;
            for (int time = start_step; time < max_seq_len_b; ++time)
            {
                const int level_beam_ix
                    = batch * param.beam_width * param.max_seq_len + beam * param.max_seq_len + time;
                if (finished)
                {
                    param.output_ids[level_beam_ix] = param.end_tokens[batch];
                }
                else if (param.output_ids[level_beam_ix] == param.end_tokens[batch])
                {
                    finished = true;
                }
            }
        }
    }
}

template <typename T>
__device__ __forceinline__ T apply_length_penalty(T log_prob, int length, float length_penalty)
{
    // score = log(prob) / (length)^length_penalty.
    if (length_penalty == 0.0f || length == 1)
    {
        return log_prob;
    }
    return log_prob / static_cast<T>(powf(length, length_penalty));
}

struct RankNorm
{
    int rank;
    float norm;
};

inline __device__ RankNorm swap(const RankNorm& rankNorm, int mask, int dir)
{
    // Exchange the rank and norm inside the warp.
    RankNorm other;
    other.rank = __shfl_xor_sync(unsigned(-1), rankNorm.rank, mask);
    other.norm = __shfl_xor_sync(unsigned(-1), rankNorm.norm, mask);

    // Update the sorted values.
    bool doSwap = (rankNorm.norm != other.norm) && ((rankNorm.norm > other.norm) == dir);
    RankNorm res;
    res.rank = doSwap ? other.rank : rankNorm.rank;
    res.norm = doSwap ? other.norm : rankNorm.norm;

    return res;
}

inline __device__ uint32_t bfe(uint32_t a, uint32_t start, uint32_t len = 1)
{
    uint32_t d;
    asm volatile("bfe.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(start), "r"(len));
    return d;
}

__global__ void finalized(gatherTreeParam param)
{
    const int beam_idx = static_cast<int>(threadIdx.x);
    const int beam_width{param.beam_width};

    extern __shared__ char array[];
    int* s_rank = (int*) (array);
    int* s_length = (int*) (s_rank + beam_width);
    float* s_scores = (float*) (s_length + beam_width);
    float* s_normed_scores = (float*) (s_scores + beam_width);
    int* s_ids = (int*) (s_normed_scores + beam_width);

    if (beam_idx < beam_width)
    {
        const int idx = blockIdx.x * param.beam_width + beam_idx;
        const int num_generated_token{param.sequence_lengths[idx] - param.input_lengths[idx]};
        s_normed_scores[beam_idx]
            = apply_length_penalty(param.cum_log_probs[idx], num_generated_token, param.length_penalty);
        s_length[beam_idx] = param.sequence_lengths[idx];
        s_scores[beam_idx] = param.cum_log_probs[idx];
    }
    for (int idx = beam_idx; idx < beam_width * param.max_seq_len; idx += blockDim.x)
    {
        s_ids[idx] = param.output_ids[blockIdx.x * param.beam_width * param.max_seq_len + idx];
    }
    __syncthreads();

    RankNorm rankNorm;
    rankNorm.rank = beam_idx;
    rankNorm.norm = beam_idx < beam_width ? s_normed_scores[beam_idx] : -FLT_MAX;

    if (beam_width < 32)
    {
        int warpid = threadIdx.x / 32;
        int laneid = threadIdx.x % 32;

        if (warpid == 0 && beam_width > 1)
        {
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 1) ^ bfe(laneid, 0)); //  2
        }

        if (warpid == 0 && beam_width > 2)
        {
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 2) ^ bfe(laneid, 1)); //  3~4
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 2) ^ bfe(laneid, 0));
        }

        if (warpid == 0 && beam_width > 4)
        {
            rankNorm = swap(rankNorm, 0x04, bfe(laneid, 3) ^ bfe(laneid, 2)); //  5~8
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 3) ^ bfe(laneid, 1));
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 3) ^ bfe(laneid, 0));
        }

        if (warpid == 0 && beam_width > 8)
        {
            rankNorm = swap(rankNorm, 0x08, bfe(laneid, 4) ^ bfe(laneid, 3)); // 9~16
            rankNorm = swap(rankNorm, 0x04, bfe(laneid, 4) ^ bfe(laneid, 2));
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 4) ^ bfe(laneid, 1));
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 4) ^ bfe(laneid, 0));
        }

        if (warpid == 0 && beam_width > 16)
        {
            rankNorm = swap(rankNorm, 0x10, bfe(laneid, 4) ^ bfe(laneid, 4)); // 17~32
            rankNorm = swap(rankNorm, 0x08, bfe(laneid, 4) ^ bfe(laneid, 3));
            rankNorm = swap(rankNorm, 0x04, bfe(laneid, 4) ^ bfe(laneid, 2));
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 4) ^ bfe(laneid, 1));
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 4) ^ bfe(laneid, 0));
        }
    }
    else
    {
        // Not supported! We must have a check before calling that kernel.
    }

    if (beam_idx < beam_width)
    {
        s_rank[beam_idx] = rankNorm.rank;
    }

    __syncthreads();

    if (beam_idx < beam_width)
    {
        auto src_idx{rankNorm.rank};
        auto tgt_idx{blockIdx.x * param.beam_width + beam_idx};
        param.sequence_lengths[tgt_idx] = s_length[src_idx];
        param.cum_log_probs[tgt_idx] = s_scores[src_idx];
    }

    for (int beam_idx = 0; beam_idx < beam_width; beam_idx++)
    {
        for (int i = threadIdx.x; i < s_length[s_rank[beam_idx]]; i += blockDim.x)
        {
            param.output_ids[blockIdx.x * beam_width * param.max_seq_len + beam_idx * param.max_seq_len + i]
                = s_ids[s_rank[beam_idx] * param.max_seq_len + i];
        }
    }
}

void invokeGatherTree(gatherTreeParam param)
{
    int batchbeam = param.batch_size * param.beam_width;
    dim3 grid(1), block(batchbeam);
    // though decoder do not support > 1024 for now
    if (batchbeam > 1024)
    {
        grid.x = ceil(param.batch_size * param.beam_width / 1024.);
        block.x = 1024;
    }
    gatherTree<<<grid, block, 0, param.stream>>>(param);
    sync_check_cuda_error();

    if (param.beam_width > 1)
    {
        TLLM_CHECK_WITH_INFO(param.beam_width <= 32, "TRT-LLM does not support beam width > 32 now");
        // sort results by normalized cum_log_probs
        dim3 grid(param.batch_size);
        dim3 block(divUp(param.beam_width, 32) * 32);

        auto shm_size = param.beam_width * (sizeof(float) * 2 + sizeof(int) * 2 + sizeof(int) * param.max_seq_len);
        finalized<<<grid, block, shm_size, param.stream>>>(param);
    }
}

__global__ void finalize(int* output_ids, int* sequence_lengths, float* cum_log_probs, float* output_log_probs,
    const int* topk_output_ids, const int* topk_sequence_lengths, const float* scores, const float* topk_cum_log_probs,
    const float* topk_log_probs, const int* num_beams, const int* input_lengths, const int beam_width,
    const int max_seq_len)
{
    // output_ids: [bs, beam_width, max_seq_len]
    // sequence_lengths: [bs, beam_width]
    // cum_log_probs: [bs, beam_width]
    // output_log_probs: [bs, beam_width, max_seq_len]
    // topk_output_ids: [bs, 2 * beam_width, max_seq_len + 1]
    // topk_sequence_lengths: [bs, 2 * beam_width]
    // scores: [bs, 2 * beam_width]
    // topk_cum_log_probs: [bs, 2 * beam_width]
    // topk_log_probs: [bs, 2 * beam_width, max_seq_len + 1]
    // num_beams: [bs]

    // This kernel do a sorting for scores first, and then put the topk_output_ids
    // into output_ids by the rank of scores.
    // Note that we remove the start_token (the id at first position) from topk_output_ids

    extern __shared__ char array[];
    int* s_rank = (int*) (array);                                 // [beam_width]
    float* s_scores = (float*) (s_rank + beam_width);             // [2 * beam_width]
    int* s_sequence_lengths = (int*) (s_scores + beam_width * 2); // [beam_width]
    const int num_beam = num_beams[blockIdx.x];
    if (threadIdx.x < num_beam)
    {
        s_scores[threadIdx.x] = scores[blockIdx.x * beam_width * 2 + threadIdx.x];
    }
    __syncthreads();

    if (num_beam < 32)
    {
        const int beam_idx = threadIdx.x;
        RankNorm rankNorm;
        rankNorm.rank = beam_idx;
        rankNorm.norm = beam_idx < num_beam ? s_scores[beam_idx] : -FLT_MAX;

        int warpid = threadIdx.x / 32;
        int laneid = threadIdx.x % 32;

        if (warpid == 0 && num_beam > 1)
        {
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 1) ^ bfe(laneid, 0)); //  2
        }

        if (warpid == 0 && num_beam > 2)
        {
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 2) ^ bfe(laneid, 1)); //  3~4
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 2) ^ bfe(laneid, 0));
        }

        if (warpid == 0 && num_beam > 4)
        {
            rankNorm = swap(rankNorm, 0x04, bfe(laneid, 3) ^ bfe(laneid, 2)); //  5~8
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 3) ^ bfe(laneid, 1));
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 3) ^ bfe(laneid, 0));
        }

        if (warpid == 0 && num_beam > 8)
        {
            rankNorm = swap(rankNorm, 0x08, bfe(laneid, 4) ^ bfe(laneid, 3)); // 9~16
            rankNorm = swap(rankNorm, 0x04, bfe(laneid, 4) ^ bfe(laneid, 2));
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 4) ^ bfe(laneid, 1));
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 4) ^ bfe(laneid, 0));
        }

        if (warpid == 0 && num_beam > 16)
        {
            rankNorm = swap(rankNorm, 0x10, bfe(laneid, 4) ^ bfe(laneid, 4)); // 17~32
            rankNorm = swap(rankNorm, 0x08, bfe(laneid, 4) ^ bfe(laneid, 3));
            rankNorm = swap(rankNorm, 0x04, bfe(laneid, 4) ^ bfe(laneid, 2));
            rankNorm = swap(rankNorm, 0x02, bfe(laneid, 4) ^ bfe(laneid, 1));
            rankNorm = swap(rankNorm, 0x01, bfe(laneid, 4) ^ bfe(laneid, 0));
        }

        if (beam_idx < beam_width)
        {
            s_rank[beam_idx] = rankNorm.rank;
        }

        __syncthreads();
    }
    else
    {
        for (int i = 0; i < beam_width; i++)
        {
            float score = threadIdx.x < num_beams[blockIdx.x] ? s_scores[threadIdx.x] : -FLT_MAX;
            float max_score = blockReduceMax<float>(score);

            if (threadIdx.x == 0)
            {
                for (int j = 0; j < beam_width * 2; j++)
                {
                    if (s_scores[j] == max_score)
                    {
                        s_rank[i] = j;
                        s_scores[j] = -FLT_MAX;
                        break;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (threadIdx.x < beam_width)
    {
        s_sequence_lengths[threadIdx.x] = topk_sequence_lengths[blockIdx.x * beam_width * 2 + s_rank[threadIdx.x]];
        sequence_lengths[blockIdx.x * beam_width + threadIdx.x] = s_sequence_lengths[threadIdx.x];

        if (cum_log_probs != nullptr)
        {
            cum_log_probs[blockIdx.x * beam_width + threadIdx.x]
                = topk_cum_log_probs[blockIdx.x * beam_width * 2 + s_rank[threadIdx.x]];
        }
    }
    __syncthreads();

    for (int beam_idx = 0; beam_idx < beam_width; beam_idx++)
    {
        // start from step 1 to skip the start token
        for (int i = threadIdx.x; i < s_sequence_lengths[beam_idx]; i += blockDim.x)
        {
            output_ids[blockIdx.x * beam_width * max_seq_len + beam_idx * max_seq_len + i]
                = topk_output_ids[blockIdx.x * (beam_width * 2) * max_seq_len + s_rank[beam_idx] * max_seq_len + i];
            if (output_log_probs != nullptr)
            {
                output_log_probs[blockIdx.x * beam_width * max_seq_len + beam_idx * max_seq_len + i]
                    = topk_log_probs[blockIdx.x * (beam_width * 2) * max_seq_len + s_rank[beam_idx] * max_seq_len + i];
            }
        }
    }
}

void invokeFinalize(int* output_ids, int* sequence_lengths, float* cum_log_probs, float* output_log_probs,
    const int* topk_output_ids, const int* topk_sequence_lengths, const float* scores, const float* topk_cum_log_probs,
    const float* topk_log_probs, const int* num_beams, const int* input_lengths, const int beam_width,
    const int max_seq_len, const int batch_size, cudaStream_t stream)
{
    TLLM_LOG_DEBUG("%s %s start", __FILE__, __PRETTY_FUNCTION__);
    dim3 block(beam_width * 2);
    block.x = (block.x + 31) / 32 * 32;
    TLLM_CHECK(block.x < 1024);
    finalize<<<batch_size, block, beam_width * sizeof(int) * 2 + (beam_width * 2) * sizeof(float), stream>>>(output_ids,
        sequence_lengths, cum_log_probs, output_log_probs, topk_output_ids, topk_sequence_lengths, scores,
        topk_cum_log_probs, topk_log_probs, num_beams, input_lengths, beam_width, max_seq_len);
}

__global__ void initializeOutput(int* output_ids, const int* end_ids, const int max_seq_len)
{
    for (int i = threadIdx.x; i < max_seq_len; i += blockDim.x)
    {
        output_ids[blockIdx.x * max_seq_len + i] = end_ids[blockIdx.x];
    }
}

void invokeInitializeOutput(int* output_ids, const int* end_ids, int batch_beam, int max_seq_len, cudaStream_t stream)
{
    initializeOutput<<<batch_beam, 256, 0, stream>>>(output_ids, end_ids, max_seq_len);
}

__global__ void copyNextStepIds(int* next_step_ids, int** output_ids_ptr, const int* sequence_lengths, int batch_size,
    int beam_width, int max_seq_len)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * beam_width;
         index += blockDim.x * gridDim.x)
    {
        const int batch_idx{index / beam_width};
        const int beam_idx{index % beam_width};
        next_step_ids[index] = output_ids_ptr[batch_idx][beam_idx * max_seq_len + sequence_lengths[index] - 1];
    }
}

void invokeCopyNextStepIds(int* next_step_ids, int** output_ids_ptr, const int* sequence_lengths, int batch_size,
    int beam_width, int max_seq_len, cudaStream_t stream)
{
    dim3 block(min(256, batch_size * beam_width));
    dim3 grid(divUp(batch_size * beam_width, block.x));
    copyNextStepIds<<<grid, block, 0, stream>>>(
        next_step_ids, output_ids_ptr, sequence_lengths, batch_size, beam_width, max_seq_len);
}

} // namespace kernels
} // namespace tensorrt_llm
