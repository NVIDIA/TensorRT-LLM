/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/fusedCatFp8Scatter.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

std::tuple<at::Tensor, at::Tensor> fused_cat_fp8_scatter(at::Tensor const& pe, at::Tensor const& nope, bool use_ue8m0,
    at::Tensor& k_cache, at::Tensor const& slot_mapping_fp8, at::Tensor const& slot_mapping_scale)
{
    CHECK_TH_CUDA(pe);
    CHECK_TH_CUDA(nope);
    CHECK_TH_CUDA(k_cache);
    CHECK_TH_CUDA(slot_mapping_fp8);
    CHECK_TH_CUDA(slot_mapping_scale);

    TORCH_CHECK(pe.scalar_type() == at::ScalarType::BFloat16, "pe must be BF16, got ", pe.scalar_type());
    TORCH_CHECK(nope.scalar_type() == at::ScalarType::BFloat16, "nope must be BF16, got ", nope.scalar_type());
    TORCH_CHECK(pe.dim() >= 2, "pe must be >= 2D, got ", pe.dim(), "D");
    TORCH_CHECK(nope.dim() >= 2, "nope must be >= 2D, got ", nope.dim(), "D");

    TORCH_CHECK(pe.stride(-1) == 1, "pe must have contiguous innermost dim");
    TORCH_CHECK(nope.stride(-1) == 1, "nope must have contiguous innermost dim");

    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be 4D [num_blocks, block_size, 1, per_token_size]");
    TORCH_CHECK(slot_mapping_fp8.dim() == 1, "slot_mapping_fp8 must be 1D");
    TORCH_CHECK(slot_mapping_scale.dim() == 1, "slot_mapping_scale must be 1D");
    TORCH_CHECK(slot_mapping_fp8.scalar_type() == torch::kInt64, "slot_mapping_fp8 must be int64");
    TORCH_CHECK(slot_mapping_scale.scalar_type() == torch::kInt64, "slot_mapping_scale must be int64");

    auto const pe_dim = static_cast<int32_t>(pe.size(-1));
    auto const nope_dim = static_cast<int32_t>(nope.size(-1));
    auto const head_dim = pe_dim + nope_dim;

    TORCH_CHECK(head_dim == 128, "head_dim (pe_dim + nope_dim) must be 128, got ", head_dim);

    auto const pe_M = pe.numel() / pe_dim;
    auto const nope_M = nope.numel() / nope_dim;
    TORCH_CHECK(pe_M == nope_M, "pe and nope must have same number of rows");
    auto const M = static_cast<int32_t>(pe_M);

    TORCH_CHECK(slot_mapping_fp8.size(0) == M, "slot_mapping_fp8 length must equal num rows");
    TORCH_CHECK(slot_mapping_scale.size(0) == M, "slot_mapping_scale length must equal num rows");

    auto const pe_row_stride = static_cast<int32_t>(pe.stride(-2));
    auto const nope_row_stride = static_cast<int32_t>(nope.stride(-2));

    // Allocate contiguous output tensors
    at::Tensor fp8_out
        = at::detail::empty_cuda({M, head_dim}, at::ScalarType::Float8_e4m3fn, pe.device(), std::nullopt);
    at::Tensor scale_out = at::detail::empty_cuda({M, 1}, at::ScalarType::Float, pe.device(), std::nullopt);

    int32_t cache_dim_0 = static_cast<int32_t>(k_cache.size(0));
    int32_t cache_dim_1 = static_cast<int32_t>(k_cache.size(1));
    int32_t cache_dim_2 = static_cast<int32_t>(k_cache.size(2));
    int32_t cache_dim_3 = static_cast<int32_t>(k_cache.size(3));

    int64_t cache_stride_0 = static_cast<int64_t>(k_cache.stride(0));
    int64_t cache_stride_1 = static_cast<int64_t>(k_cache.stride(1));
    int64_t cache_stride_2 = static_cast<int64_t>(k_cache.stride(2));
    int64_t cache_stride_3 = static_cast<int64_t>(k_cache.stride(3));

    auto stream = at::cuda::getCurrentCUDAStream(pe.get_device());

    tensorrt_llm::kernels::invokeFusedCatFp8Scatter(reinterpret_cast<__nv_fp8_e4m3*>(fp8_out.data_ptr()),
        reinterpret_cast<float*>(scale_out.data_ptr()), k_cache.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16 const*>(pe.data_ptr()), reinterpret_cast<__nv_bfloat16 const*>(nope.data_ptr()),
        slot_mapping_fp8.data_ptr<int64_t>(), slot_mapping_scale.data_ptr<int64_t>(), M, pe_dim, nope_dim, head_dim,
        pe_row_stride, nope_row_stride, use_ue8m0, cache_dim_0, cache_dim_1, cache_dim_2, cache_dim_3, cache_stride_0,
        cache_stride_1, cache_stride_2, cache_stride_3, stream);

    return {fp8_out, scale_out};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_cat_fp8_scatter(Tensor pe, Tensor nope, bool use_ue8m0, "
        "Tensor(a!) k_cache, Tensor slot_mapping_fp8, Tensor slot_mapping_scale) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_cat_fp8_scatter", &tensorrt_llm::torch_ext::fused_cat_fp8_scatter);
}
