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

#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/kernels/recoverFromRingAtten.h"

#include "math.h"
#include <cooperative_groups.h>
#include <cuda/barrier>

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename Tout>
__global__ void reduce4ring_attention(
    // this is the accumulated results for all finished ring attention blocks
    Tout* __restrict__ accu_output,         // b x s_block x h x d
    float* __restrict__ accu_softmax_stats, // b x s_block x h x 2 (max/sum)
    // this is the new ring attention block results
    Tout* __restrict__ output,         // b x s_block x h x d
    float* __restrict__ softmax_stats, // b x s_block x h x 2 (max/sum)
    // necessary constant parameters
    int const b, int const s_block, int const h, int const d, int const block_seq_len, int* cu_seqlens)
{
    auto block = cooperative_groups::this_thread_block();
    int batchid = blockIdx.x;
    int block_seq_idx = blockIdx.y;
    int block_s_start = block_seq_idx * block_seq_len;
    int block_s_end = (block_seq_idx + 1) * block_seq_len;
    block_s_end = s_block < block_s_end ? s_block : block_s_end;
    int64_t output_start_offset = batchid * s_block * d + block_s_start * d;
    int64_t lm_start_offset = (batchid * s_block + block_s_start) * 2;

    float* accu_softmax_sum = accu_softmax_stats + 1;
    float* accu_max = accu_softmax_stats;
    float* softmax_sum = softmax_stats + 1;
    float* max = softmax_stats;

#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_suppress static_var_with_dynamic_init
// https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/barrier.html
#endif
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
    if (block.thread_rank() == 0)
    {
        init(&barrier, block.size());
    }
    block.sync();

    int s_len = block_s_end - block_s_start;
    int laneid = threadIdx.x % 32;
    int local_warpid = threadIdx.x / 32;
    int warp_num = blockDim.x / 32;
    int loop_on_s = (s_len + warp_num * 32 - 1) / (warp_num * 32);
    for (int l = 0; l < loop_on_s; l++)
    {
        int s_ = local_warpid + warp_num * laneid + l * warp_num * 32;
        float scaled_my_ss1_ = 1.0, scaled_my_ss2_ = 1.0;
        if (s_ < s_len)
        {
            uint64_t lm_start_offset_ = lm_start_offset + s_ * 2;
            float my_accu_ss = accu_softmax_sum[lm_start_offset_] == 0.0 ? 1.0 : accu_softmax_sum[lm_start_offset_];
            float my_ss = softmax_sum[lm_start_offset_] == 0.0 ? 1.0 : softmax_sum[lm_start_offset_];

            float cur_max = (accu_max[lm_start_offset_] > max[lm_start_offset_]) ? accu_max[lm_start_offset_]
                                                                                 : max[lm_start_offset_];
            float scale1 = exp(accu_max[lm_start_offset_] - cur_max);
            float scale2 = exp(max[lm_start_offset_] - cur_max);
            float cur_softmax_sum = my_accu_ss * scale1 + my_ss * scale2;
            if (cur_softmax_sum == 0)
                cur_softmax_sum = 1.0;
            scaled_my_ss1_ = scale1 * my_accu_ss / cur_softmax_sum;
            scaled_my_ss2_ = scale2 * my_ss / cur_softmax_sum;
            accu_softmax_sum[lm_start_offset_] = cur_softmax_sum;
            accu_max[lm_start_offset_] = cur_max;
        }
        int sid = l * warp_num * 32 + local_warpid;
        int s_end = (l + 1) * warp_num * 32 < s_len ? (l + 1) * warp_num * 32 : s_len;
        for (int ss = 0;; ss++)
        {
            uint64_t output_start_offset_ = output_start_offset + sid * d;
            float scaled_my_ss1 = __shfl_sync(0xffffffff, scaled_my_ss1_, ss, 32);
            float scaled_my_ss2 = __shfl_sync(0xffffffff, scaled_my_ss2_, ss, 32);
            for (int eid = laneid; eid < d; eid += 32)
            {
                accu_output[output_start_offset_ + eid]
                    = (float) accu_output[output_start_offset_ + eid] * scaled_my_ss1
                    + (float) output[output_start_offset_ + eid] * scaled_my_ss2;
            }
            sid += warp_num;
            if (sid >= s_end)
                break;
        }
    }
    barrier.arrive_and_wait();
    return;
}

template <typename Tout>
void invokeRecoverFromRA(Tout* accu_output, float* accu_softmax_stats, Tout* output, float* softmax_stats, int b, int s,
    int h, int d, int* cu_seqlens, cudaStream_t stream)
{
    int threads_per_block = 128;
    int saturated_s_block_dim = 3000 / b + 1;
    s = s * h;
    int block_seq_len = (s / saturated_s_block_dim + 255) / 256 * 256;
    block_seq_len = block_seq_len < 256 ? 256 : block_seq_len;
    int dim_s = (s + block_seq_len - 1) / block_seq_len;

    dim3 block_num(b, dim_s, 1);
    reduce4ring_attention<Tout><<<block_num, threads_per_block, 0, stream>>>(
        accu_output, accu_softmax_stats, output, softmax_stats, b, s, h, d, block_seq_len, cu_seqlens);
}

#define INSTANTIATE_RECOVER_RA(Tout)                                                                                   \
    template void invokeRecoverFromRA(Tout* accu_output, float* accu_softmax_stats, Tout* output,                      \
        float* softmax_stats, int b, int s, int h, int d, int* cu_seqlens, cudaStream_t stream)
INSTANTIATE_RECOVER_RA(float);
INSTANTIATE_RECOVER_RA(half);
#ifdef ENABLE_BF16
INSTANTIATE_RECOVER_RA(__nv_bfloat16);
#endif
} // namespace kernels
} // namespace tensorrt_llm
