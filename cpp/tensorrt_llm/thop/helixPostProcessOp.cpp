/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/helixKernels.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

torch::Tensor helix_post_process(torch::Tensor const& gathered_o, torch::Tensor const& gathered_stats, double scale)
{
    CHECK_TH_CUDA(gathered_o);
    CHECK_CONTIGUOUS(gathered_o);
    CHECK_TH_CUDA(gathered_stats);
    CHECK_CONTIGUOUS(gathered_stats);

    TORCH_CHECK(gathered_o.dim() == 3, "gathered_o must be 3D tensor [cp_size, num_tokens, num_heads * kv_lora_rank]");
    TORCH_CHECK(gathered_stats.dim() == 4, "gathered_stats must be 4D tensor [cp_size, num_tokens, num_heads, 2]");

    auto const cp_size = gathered_stats.sizes()[0];
    auto const num_tokens = gathered_stats.sizes()[1];
    auto const num_heads = gathered_stats.sizes()[2];

    TORCH_CHECK(gathered_o.sizes()[2] % num_heads == 0, "last dimension of gathered_o must be divisible by num_heads");
    auto const kv_lora_rank = gathered_o.sizes()[2] / num_heads;

    // check remaining input tensor dimensions
    TORCH_CHECK(gathered_o.sizes()[0] == cp_size, "gathered_o first dimension must match cp_size");
    TORCH_CHECK(gathered_o.sizes()[1] == num_tokens, "gathered_o second dimension must match num_tokens");
    TORCH_CHECK(gathered_o.sizes()[2] == num_heads * kv_lora_rank,
        "gathered_o third dimension must match num_heads * kv_lora_rank");

    TORCH_CHECK(gathered_stats.sizes()[3] == 2, "gathered_stats fourth dimension must be 2");

    // Check data types
    TORCH_CHECK(
        gathered_o.scalar_type() == at::ScalarType::Half || gathered_o.scalar_type() == at::ScalarType::BFloat16,
        "gathered_o must be half or bfloat16");
    TORCH_CHECK(gathered_stats.scalar_type() == at::ScalarType::Float, "gathered_stats must be float32");

    // Check alignment requirements for gathered_o (16-byte aligned for async memcpy)
    TORCH_CHECK(reinterpret_cast<uintptr_t>(gathered_o.data_ptr()) % 16 == 0, "gathered_o must be 16-byte aligned");

    // Check that kv_lora_rank * sizeof(data_type) is a multiple of 16
    size_t data_type_size = torch::elementSize(gathered_o.scalar_type());
    TORCH_CHECK((kv_lora_rank * data_type_size) % 16 == 0, "kv_lora_rank * sizeof(data_type) must be a multiple of 16");

    // Create output tensor
    std::vector<int64_t> output_shape = {num_tokens, num_heads * kv_lora_rank};
    torch::Tensor output = torch::empty(output_shape, gathered_o.options());

    // Get CUDA stream
    auto stream = at::cuda::getCurrentCUDAStream(gathered_o.get_device());

#define CALL_CPP_OP(T)                                                                                                 \
    tensorrt_llm::kernels::HelixPostProcParams<T> params{reinterpret_cast<T*>(output.mutable_data_ptr()),              \
        reinterpret_cast<T const*>(gathered_o.data_ptr()), reinterpret_cast<float2 const*>(gathered_stats.data_ptr()), \
        static_cast<int>(cp_size), static_cast<int>(num_tokens), static_cast<int>(num_heads),                          \
        static_cast<int>(kv_lora_rank)};                                                                               \
    tensorrt_llm::kernels::helixPostProcess(params, stream);

    if (gathered_o.scalar_type() == at::ScalarType::Half)
    {
        CALL_CPP_OP(__half);
    }
    else if (gathered_o.scalar_type() == at::ScalarType::BFloat16)
    {
#ifdef ENABLE_BF16
        CALL_CPP_OP(__nv_bfloat16);
#else
        TLLM_THROW("BFloat16 must be enabled to use helix_post_process with bf16 tensors.");
#endif
    }

    if (scale != 1.0)
    {
        output *= scale;
    }

    return output;
}

template <typename T, typename Fn>
inline torch::Tensor helix_post_process_native_impl(
    torch::Tensor const& gathered_o, torch::Tensor const& gathered_stats, double scale, int cp_dim, Fn fn)
{
    CHECK_TH_CUDA(gathered_o);
    CHECK_CONTIGUOUS(gathered_o);
    CHECK_TH_CUDA(gathered_stats);
    CHECK_CONTIGUOUS(gathered_stats);

    // Only cp_dim=2 is supported
    TORCH_CHECK(cp_dim == 2,
        "cp_dim must be 2. Expects tensor shapes to be: \n"
        "gathered_o: [num_tokens, num_heads, cp_size, kv_lora_rank], \n"
        "gathered_stats: [num_tokens, num_heads, cp_size, 2]");

    // For cp_dim=2: tokens_dim=0, heads_dim=1
    auto tokens_dim = 0;
    auto heads_dim = 1;

    TORCH_CHECK(gathered_o.dim() == 4, "gathered_o must be 4D tensor [num_tokens, num_heads, cp_size, kv_lora_rank]");
    TORCH_CHECK(gathered_stats.dim() == 4, "gathered_stats must be 4D tensor [num_tokens, num_heads, cp_size, 2]");

    auto const num_tokens = gathered_stats.sizes()[tokens_dim];
    auto const num_heads = gathered_stats.sizes()[heads_dim];
    auto const cp_size = gathered_stats.sizes()[2];
    auto const kv_lora_rank = gathered_o.sizes()[3];

    // check remaining input tensor dimensions
    TORCH_CHECK(gathered_o.sizes()[2] == cp_size, "gathered_o cp_size dim must match");
    TORCH_CHECK(gathered_o.sizes()[tokens_dim] == num_tokens, "gathered_o tokens_dim must match num_tokens");
    TORCH_CHECK(gathered_o.sizes()[heads_dim] == num_heads, "gathered_o heads_dim must match num_heads");

    TORCH_CHECK(gathered_stats.sizes()[3] == 2, "gathered_stats last dimension must be 2");

    // Check data types
    TORCH_CHECK(
        gathered_o.scalar_type() == at::ScalarType::Half || gathered_o.scalar_type() == at::ScalarType::BFloat16,
        "gathered_o must be half or bfloat16");
    TORCH_CHECK(gathered_stats.scalar_type() == at::ScalarType::Float, "gathered_stats must be float32");

    // Check alignment requirements for gathered_o (16-byte aligned for async
    // memcpy)
    TORCH_CHECK(reinterpret_cast<uintptr_t>(gathered_o.data_ptr()) % 16 == 0, "gathered_o must be 16-byte aligned");

    // Check that kv_lora_rank * sizeof(data_type) is a multiple of 16
    size_t data_type_size = torch::elementSize(gathered_o.scalar_type());
    TORCH_CHECK((kv_lora_rank * data_type_size) % 16 == 0, "kv_lora_rank * sizeof(data_type) must be a multiple of 16");

    // Create output tensor
    std::vector<int64_t> output_shape = {num_tokens, num_heads * kv_lora_rank};
    torch::Tensor output = torch::empty(output_shape, gathered_o.options());

    // Get CUDA stream
    auto stream = at::cuda::getCurrentCUDAStream(gathered_o.get_device());

    tensorrt_llm::kernels::HelixPostProcParams<T> params{reinterpret_cast<T*>(output.mutable_data_ptr()),
        reinterpret_cast<T const*>(gathered_o.data_ptr()), reinterpret_cast<float2 const*>(gathered_stats.data_ptr()),
        static_cast<int>(cp_size), static_cast<int>(num_tokens), static_cast<int>(num_heads),
        static_cast<int>(kv_lora_rank)};
    fn(params, stream);

    if (scale != 1.0)
    {
        output *= scale;
    }

    return output;
}

inline torch::Tensor helix_post_process_native(
    torch::Tensor const& gathered_o, torch::Tensor const& gathered_stats, double scale, int64_t cp_dim)
{
    TORCH_CHECK(cp_dim == 2, "cp_dim must be 2. Only cp_dim=2 layout is supported.");
    if (gathered_o.scalar_type() == at::ScalarType::Half)
    {
        return helix_post_process_native_impl<__half>(
            gathered_o, gathered_stats, scale, int(cp_dim), tensorrt_llm::kernels::helixPostProcessNative<__half>);
    }
    else if (gathered_o.scalar_type() == at::ScalarType::BFloat16)
    {
#ifdef ENABLE_BF16
        return helix_post_process_native_impl<__nv_bfloat16>(gathered_o, gathered_stats, scale, int(cp_dim),
            tensorrt_llm::kernels::helixPostProcessNative<__nv_bfloat16>);
#else
        TLLM_THROW("BFloat16 must be enabled to use helix_post_process_native with bf16 tensors.");
#endif
    }
    else
    {
        TLLM_THROW("helix_post_process_native only supports half and bfloat16 tensors.");
    }
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("helix_post_process(Tensor gathered_o, Tensor gathered_stats, float scale) -> Tensor");
    m.def(
        "helix_post_process_native(Tensor gathered_o, Tensor gathered_stats, float "
        "scale, int cp_dim) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("helix_post_process", helix_post_process);
    m.impl("helix_post_process_native", &helix_post_process_native);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
