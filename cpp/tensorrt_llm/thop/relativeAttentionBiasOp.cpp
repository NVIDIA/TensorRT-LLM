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

#include "tensorrt_llm/kernels/buildRelativeAttentionBiasKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

namespace th = torch;
namespace tl = tensorrt_llm;
namespace tk = tensorrt_llm::kernels;

namespace torch_ext
{

template <typename T>
void handleInvokeRelativeAttentionBias(th::Tensor& relative_attention_bias, th::Tensor& relative_attention_bias_table,
    int64_t const num_head, int64_t const max_seq_len, int64_t const num_bucket, bool const is_bidirectional,
    int64_t const max_distance, cudaStream_t stream)
{

    T* relative_attention_bias_ptr = get_ptr<T>(relative_attention_bias);
    T const* relative_attention_bias_table_ptr = get_ptr<T>(relative_attention_bias_table);

    tk::invokeBuildRelativeAttentionBias<T>(relative_attention_bias_ptr, relative_attention_bias_table_ptr, num_head,
        (max_seq_len + 1), num_bucket, is_bidirectional, max_distance, stream);
}

void buildRelativeAttentionBias(
    th::Tensor& relative_attention_bias,       // sizeof(T) * num_head * (max_seq_len + 1) * (max_seq_len + 1)
    th::Tensor& relative_attention_bias_table, // sizeof(T) * num_head * num_bucket
    int64_t const num_head, int64_t const max_seq_len, int64_t const num_bucket, bool const is_bidirectional,
    int64_t const max_distance)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    switch (relative_attention_bias_table.scalar_type())
    {
    case at::ScalarType::Float:
        handleInvokeRelativeAttentionBias<float>(relative_attention_bias, relative_attention_bias_table, num_head,
            max_seq_len, num_bucket, is_bidirectional, max_distance, stream);
        break;
    case at::ScalarType::Half:
        handleInvokeRelativeAttentionBias<half>(relative_attention_bias, relative_attention_bias_table, num_head,
            max_seq_len, num_bucket, is_bidirectional, max_distance, stream);
        break;
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
        handleInvokeRelativeAttentionBias<__nv_bfloat16>(relative_attention_bias, relative_attention_bias_table,
            num_head, max_seq_len, num_bucket, is_bidirectional, max_distance, stream);
        break;
#endif
    default: throw std::runtime_error("Unimplemented scalar type");
    }

    sync_check_cuda_error(stream);
}

} // namespace torch_ext

static auto relative_attention_bias
    = torch::RegisterOperators("tensorrt_llm::relative_attention_bias", &torch_ext::buildRelativeAttentionBias);
