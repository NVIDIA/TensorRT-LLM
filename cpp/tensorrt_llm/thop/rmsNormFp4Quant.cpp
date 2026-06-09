/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/kernels/rmsNormFp4QuantKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>

#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Fused residual-add + RMSNorm + NVFP4 input-quantize in one kernel.
// Replaces the (flashinfer fused_add_rmsnorm + standalone fp4_quantize) pair on
// the no-allreduce / attention-DP path. Each rank operates on its local tokens.
//
// Inputs:
//   hidden_states : [..., hidden_size] BF16/FP16 — read-only. The residual sum
//                   (hidden_states + residual) is returned as a fresh
//                   residual_out tensor; hidden_states itself is not mutated, so
//                   the op is functionalizable under torch.compile.
//   residual      : [..., hidden_size] same dtype, read-only.
//   norm_weight   : [hidden_size] same dtype, RMSNorm gamma.
//   scale_factor  : [] float32, = (448 * 6) / amax for static-quant Linear.
//   eps           : RMSNorm epsilon.
//   return_norm_out : when true, also return the BF16 normed value (needed by
//                     DSA indexer's pre_indexer_proj).
//
// Returns: [quant_out, scale_out, residual_out] or
//          [norm_out, quant_out, scale_out, residual_out] when return_norm_out.
std::vector<at::Tensor> fused_add_rmsnorm_fp4_quantize(at::Tensor const& hidden_states, at::Tensor residual,
    at::Tensor const& norm_weight, at::Tensor const& scale_factor, double eps, bool return_norm_out)
{
    CHECK_TH_CUDA(hidden_states);
    CHECK_CONTIGUOUS(hidden_states);
    CHECK_TH_CUDA(residual);
    CHECK_CONTIGUOUS(residual);
    CHECK_TH_CUDA(norm_weight);
    CHECK_CONTIGUOUS(norm_weight);
    CHECK_INPUT(scale_factor, torch::kFloat32);
    TORCH_CHECK(
        hidden_states.scalar_type() == residual.scalar_type(), "hidden_states and residual must have matching dtype");
    TORCH_CHECK(hidden_states.scalar_type() == norm_weight.scalar_type(),
        "hidden_states and norm_weight must have matching dtype");

    auto const& input_shape = hidden_states.sizes();
    auto const rank = input_shape.size();
    TORCH_CHECK(rank >= 2, "hidden_states should be >=2D");
    int64_t m = 1;
    for (size_t i = 0; i < rank - 1; i++)
    {
        m *= input_shape[i];
    }
    auto const k = input_shape[rank - 1];
    int64_t const sf_vec_size = 16;
    TORCH_CHECK(k % sf_vec_size == 0, "hidden_size must be divisible by 16");

    std::vector<int64_t> quant_shape(input_shape.begin(), input_shape.end());
    quant_shape[rank - 1] = k / 2;
    at::Tensor quant_out = at::detail::empty_cuda(quant_shape, FLOAT4_E2M1X2, hidden_states.device(), std::nullopt);
    at::Tensor scale_out = at::detail::empty_cuda({tensorrt_llm::computeSwizzledLayoutSFSize(m, k / sf_vec_size)},
        SF_DTYPE, hidden_states.device(), std::nullopt);

    at::Tensor norm_out;
    void* norm_out_ptr = nullptr;
    if (return_norm_out)
    {
        norm_out = torch::empty_like(hidden_states);
        norm_out_ptr = norm_out.mutable_data_ptr();
    }

    // Give the kernel a fresh buffer (seeded with hidden_states) to add the
    // residual into and write residual_out back to, so the input hidden_states is
    // never mutated. This keeps the op functionalizable under torch.compile (no
    // output aliases an input). The kernel reads intermediate_buffer, adds
    // residual_buffer, and writes residual_out back into intermediate_buffer.
    at::Tensor residual_out = torch::empty_like(hidden_states);
    residual_out.copy_(hidden_states);

    tensorrt_llm::kernels::RmsNormFp4QuantParams params{};
    params.bias_buffer = nullptr;
    params.residual_buffer = residual.data_ptr();
    params.weight_buffer = norm_weight.data_ptr();
    params.intermediate_buffer = residual_out.mutable_data_ptr();
    params.hidden_size = static_cast<int>(k);
    params.eps = static_cast<float>(eps);
    params.elts_total = hidden_states.numel();

    auto const stream = at::cuda::getCurrentCUDAStream(hidden_states.get_device());
    auto const dtype = tensorrt_llm::runtime::TorchUtils::dataType(hidden_states.scalar_type());

    tensorrt_llm::kernels::residualRmsNormFp4Quant(params, quant_out.mutable_data_ptr(), scale_out.mutable_data_ptr(),
        norm_out_ptr, static_cast<float const*>(scale_factor.data_ptr()), tensorrt_llm::QuantizationSFLayout::SWIZZLED,
        dtype, stream);

    // residual_out holds the residual sum (= original hidden + original residual).
    if (return_norm_out)
    {
        return {norm_out, quant_out, scale_out, residual_out};
    }
    return {quant_out, scale_out, residual_out};
}

// Residual-less variant of fused_add_rmsnorm_fp4_quantize. Replaces the
// (flashinfer rmsnorm + standalone fp4_quantize) pair on intra-layer paths that
// have NO residual add — e.g. DSv3.2/Kimi-K2.5 MLA's q_a_layernorm feeding the
// static-NVFP4 q_b_proj. The kernel reads intermediate_buffer (=hidden_states),
// skips the residual add (residual_buffer == nullptr selects Residual=false in
// the launcher), RMSNorms it, and FP4-quantizes the result. hidden_states is
// NOT modified (no residual write-back happens when Residual=false).
//
// Inputs:
//   hidden_states : [..., hidden_size] BF16/FP16 — read-only.
//   norm_weight   : [hidden_size] same dtype, RMSNorm gamma.
//   scale_factor  : [] float32, = (448 * 6) / amax for static-quant Linear.
//   eps           : RMSNorm epsilon.
//   return_norm_out : when true, also return the BF16 normed value.
//
// Returns: [quant_out, scale_out] or [norm_out, quant_out, scale_out] when
//          return_norm_out.
std::vector<at::Tensor> fused_rmsnorm_fp4_quantize(at::Tensor const& hidden_states, at::Tensor const& norm_weight,
    at::Tensor const& scale_factor, double eps, bool return_norm_out)
{
    CHECK_TH_CUDA(hidden_states);
    CHECK_TH_CUDA(norm_weight);
    CHECK_CONTIGUOUS(norm_weight);
    CHECK_INPUT(scale_factor, torch::kFloat32);
    TORCH_CHECK(hidden_states.scalar_type() == norm_weight.scalar_type(),
        "hidden_states and norm_weight must have matching dtype");

    auto const& input_shape = hidden_states.sizes();
    auto const rank = input_shape.size();
    TORCH_CHECK(rank >= 2, "hidden_states should be >=2D");
    // hidden_states may be a column-slice of a wider projection (e.g. the leading
    // q_lora_rank columns of kv_a_proj_with_mqa): its last dim is unit-stride but
    // its row stride may exceed hidden_size. We read it in place via an input row
    // stride and skip the otherwise-required contiguous copy. The kernel only
    // reads with this stride; all outputs are written packed.
    TORCH_CHECK(hidden_states.stride(rank - 1) == 1, "hidden_states last dim must be unit-stride");
    // All leading dims must be densely packed on top of the row pitch so that a
    // single per-row element stride describes the flattened [m, k] layout. The
    // only permitted non-packing is a row pitch larger than k (a column slice).
    for (size_t i = 0; i + 2 < rank; i++)
    {
        TORCH_CHECK(hidden_states.stride(i) == hidden_states.stride(i + 1) * input_shape[i + 1],
            "hidden_states leading dims must be densely packed");
    }
    int64_t m = 1;
    for (size_t i = 0; i < rank - 1; i++)
    {
        m *= input_shape[i];
    }
    auto const k = input_shape[rank - 1];
    int64_t const sf_vec_size = 16;
    TORCH_CHECK(k % sf_vec_size == 0, "hidden_size must be divisible by 16");
    // Element stride between consecutive logical rows. For a contiguous tensor
    // this equals k, so input_row_stride==0 (packed) and behavior is identical.
    int64_t const row_stride = hidden_states.stride(rank - 2);
    int const input_row_stride = (row_stride == k) ? 0 : static_cast<int>(row_stride);

    std::vector<int64_t> quant_shape(input_shape.begin(), input_shape.end());
    quant_shape[rank - 1] = k / 2;
    at::Tensor quant_out = at::detail::empty_cuda(quant_shape, FLOAT4_E2M1X2, hidden_states.device(), std::nullopt);
    at::Tensor scale_out = at::detail::empty_cuda({tensorrt_llm::computeSwizzledLayoutSFSize(m, k / sf_vec_size)},
        SF_DTYPE, hidden_states.device(), std::nullopt);

    at::Tensor norm_out;
    void* norm_out_ptr = nullptr;
    if (return_norm_out)
    {
        // The kernel writes norm_out packed (stride hidden_size), so allocate a
        // packed (contiguous) tensor rather than mirroring a possibly-strided
        // input layout.
        norm_out = at::detail::empty_cuda(
            input_shape.vec(), hidden_states.scalar_type(), hidden_states.device(), std::nullopt);
        norm_out_ptr = norm_out.mutable_data_ptr();
    }

    // residual_buffer == nullptr selects the Residual=false kernel path: the
    // kernel RMSNorms intermediate_buffer (=hidden_states) directly without any
    // add or write-back, so hidden_states is left unmodified.
    tensorrt_llm::kernels::RmsNormFp4QuantParams params{};
    params.bias_buffer = nullptr;
    params.residual_buffer = nullptr;
    params.weight_buffer = norm_weight.data_ptr();
    params.intermediate_buffer = const_cast<void*>(hidden_states.data_ptr());
    params.hidden_size = static_cast<int>(k);
    params.eps = static_cast<float>(eps);
    params.elts_total = hidden_states.numel();

    auto const stream = at::cuda::getCurrentCUDAStream(hidden_states.get_device());
    auto const dtype = tensorrt_llm::runtime::TorchUtils::dataType(hidden_states.scalar_type());

    tensorrt_llm::kernels::residualRmsNormFp4Quant(params, quant_out.mutable_data_ptr(), scale_out.mutable_data_ptr(),
        norm_out_ptr, static_cast<float const*>(scale_factor.data_ptr()), tensorrt_llm::QuantizationSFLayout::SWIZZLED,
        dtype, stream, input_row_stride);

    if (return_norm_out)
    {
        return {norm_out, quant_out, scale_out};
    }
    return {quant_out, scale_out};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_add_rmsnorm_fp4_quantize("
        "Tensor hidden_states,"
        "Tensor residual,"
        "Tensor norm_weight,"
        "Tensor scale_factor,"
        "float eps,"
        "bool return_norm_out) -> Tensor[]");
    m.def(
        "fused_rmsnorm_fp4_quantize("
        "Tensor hidden_states,"
        "Tensor norm_weight,"
        "Tensor scale_factor,"
        "float eps,"
        "bool return_norm_out) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_add_rmsnorm_fp4_quantize", &tensorrt_llm::torch_ext::fused_add_rmsnorm_fp4_quantize);
    m.impl("fused_rmsnorm_fp4_quantize", &tensorrt_llm::torch_ext::fused_rmsnorm_fp4_quantize);
}
