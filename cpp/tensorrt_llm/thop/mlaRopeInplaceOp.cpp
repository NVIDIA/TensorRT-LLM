/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/mlaKernels.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Apply RoPE in-place to only the last rope_dim elements of each head.
// data: [num_tokens, num_heads, nope_dim + rope_dim]
// position_ids: [num_tokens]
// cos_sin_cache: [max_positions, 2, rope_dim/2] float
// is_neox: true = neox style (first/second half), false = interleaved style (even/odd)
void mla_rope_inplace(torch::Tensor data, torch::Tensor position_ids, torch::Tensor cos_sin_cache, int64_t num_heads,
    int64_t nope_dim, int64_t rope_dim, bool inverse, bool is_neox)
{
    auto stream = at::cuda::getCurrentCUDAStream(data.get_device());
    int const num_tokens = data.size(0);
    auto const dtype = data.scalar_type();

    TORCH_CHECK(data.dim() == 3, "data must be 3D [num_tokens, num_heads, head_dim]");
    TORCH_CHECK(data.size(1) == num_heads, "data.size(1) must equal num_heads");
    TORCH_CHECK(data.size(2) == nope_dim + rope_dim, "data.size(2) must equal nope_dim + rope_dim");
    TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
    TORCH_CHECK(position_ids.dim() == 1 && position_ids.size(0) == num_tokens, "position_ids shape mismatch");
    TORCH_CHECK(position_ids.scalar_type() == torch::kInt32, "position_ids must be int32");
    TORCH_CHECK(cos_sin_cache.scalar_type() == torch::kFloat32, "cos_sin_cache must be float32");

    if (dtype == torch::kBFloat16)
    {
        tk::invokeMLARoPEInplace(static_cast<__nv_bfloat16*>(data.data_ptr()), position_ids.data_ptr<int32_t>(),
            cos_sin_cache.data_ptr<float>(), num_tokens, static_cast<int>(num_heads), static_cast<int>(nope_dim),
            static_cast<int>(rope_dim), inverse, is_neox, stream);
    }
    else if (dtype == torch::kFloat16)
    {
        tk::invokeMLARoPEInplace(static_cast<half*>(data.data_ptr()), position_ids.data_ptr<int32_t>(),
            cos_sin_cache.data_ptr<float>(), num_tokens, static_cast<int>(num_heads), static_cast<int>(nope_dim),
            static_cast<int>(rope_dim), inverse, is_neox, stream);
    }
    else
    {
        TORCH_CHECK(false, "mla_rope_inplace: unsupported dtype, expected bf16 or fp16");
    }
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mla_rope_inplace("
        "Tensor(a!) data"
        ", Tensor position_ids"
        ", Tensor cos_sin_cache"
        ", int num_heads"
        ", int nope_dim"
        ", int rope_dim"
        ", bool inverse"
        ", bool is_neox"
        ") -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mla_rope_inplace", &tensorrt_llm::torch_ext::mla_rope_inplace);
}
