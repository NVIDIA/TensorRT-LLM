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

#include <cuda_runtime_api.h>

#include "buildRelativeAttentionBiasKernel.h"

namespace tensorrt_llm
{
namespace kernels
{

// refer to
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/bert_preprocess_kernels.cu#L291
template <typename T>
__global__ void buildRelativeAttentionBias(T* relative_attention_bias, T const* relative_attention_bias_table,
    int const head_num, int const seq_len, int const num_bucket, bool const is_bidirectional, int const max_distance)
{

    int const head_id = blockIdx.x;
    for (int seq_id = threadIdx.x; seq_id < seq_len * seq_len; seq_id += blockDim.x)
    {
        int row_id = seq_id / seq_len;
        int col_id = seq_id % seq_len;

        int relative_position = col_id - row_id;

        int relative_buckets = 0;
        int tmp_num_bucket = num_bucket;
        if (is_bidirectional)
        {
            tmp_num_bucket /= 2;
            if (relative_position > 0)
            {
                relative_buckets += tmp_num_bucket;
            }
            else
            {
                relative_position *= -1;
            }
        }
        else
        {
            relative_position = abs(relative_position);
        }

        int max_exact = tmp_num_bucket / 2;
        bool is_small = relative_position < max_exact;

        int relative_position_if_large = max_exact
            + (int) (logf(relative_position * 1.0f / max_exact) / logf((float) max_distance / max_exact)
                * (tmp_num_bucket - max_exact));

        relative_position_if_large = min(relative_position_if_large, tmp_num_bucket - 1);

        relative_buckets += is_small ? relative_position : relative_position_if_large;

        relative_attention_bias[head_id * seq_len * seq_len + seq_id]
            = relative_attention_bias_table[head_id * num_bucket + relative_buckets];
    }
}

template <typename T>
void invokeBuildRelativeAttentionBias(T* relative_attention_bias, T const* relative_attention_bias_table,
    int const head_num, int const seq_len, int const num_bucket, bool const is_bidirectional, int const max_distance,
    cudaStream_t stream)
{
    dim3 grid(head_num);
    dim3 block(256);
    buildRelativeAttentionBias<<<grid, block, 0, stream>>>(relative_attention_bias, relative_attention_bias_table,
        head_num, seq_len, num_bucket, is_bidirectional, max_distance);
}

template void invokeBuildRelativeAttentionBias<float>(float* relative_attention_bias,
    float const* relative_attention_bias_table, int const head_num, int const seq_len, int const num_bucket,
    bool const is_bidirectional, int const max_distance, cudaStream_t stream);

template void invokeBuildRelativeAttentionBias<half>(half* relative_attention_bias,
    half const* relative_attention_bias_table, int const head_num, int const seq_len, int const num_bucket,
    bool const is_bidirectional, int const max_distance, cudaStream_t stream);

#ifdef ENABLE_BF16
template void invokeBuildRelativeAttentionBias<__nv_bfloat16>(__nv_bfloat16* relative_attention_bias,
    __nv_bfloat16 const* relative_attention_bias_table, int const head_num, int const seq_len, int const num_bucket,
    bool const is_bidirectional, int const max_distance, cudaStream_t stream);
#endif

} // namespace kernels
} // namespace tensorrt_llm
