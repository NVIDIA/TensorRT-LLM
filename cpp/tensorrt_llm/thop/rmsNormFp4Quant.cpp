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

#include <limits>
#include <tuple>
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
        norm_out = at::detail::empty_cuda(
            input_shape.vec(), hidden_states.scalar_type(), hidden_states.device(), std::nullopt);
        norm_out_ptr = norm_out.mutable_data_ptr();
    }

    // Freshly allocate residual_out (input + residual is written here by the
    // kernel) the same way as quant_out/scale_out and the ws op
    // (fusedAddRMSNormQuant.cpp): empty_cuda, not empty_like. empty_cuda is a
    // plain allocation, whereas empty_like carries the input's layout and can be
    // lowered to a separate node in the torch.compile trace. The kernel reads
    // hidden_states (intermediate_buffer) read-only and writes the sum into this
    // distinct buffer, so hidden_states is never mutated and no output aliases an
    // input (the op stays functionalizable) -- with no pre-kernel copy.
    at::Tensor residual_out
        = at::detail::empty_cuda(input_shape.vec(), hidden_states.scalar_type(), hidden_states.device(), std::nullopt);

    tensorrt_llm::kernels::RmsNormFp4QuantParams params{};
    params.bias_buffer = nullptr;
    params.residual_buffer = residual.data_ptr();
    params.weight_buffer = norm_weight.data_ptr();
    params.intermediate_buffer = hidden_states.data_ptr();
    params.scale_factor_ptr = static_cast<float const*>(scale_factor.data_ptr());
    params.quant_out = quant_out.mutable_data_ptr();
    params.scale_out = scale_out.mutable_data_ptr();
    params.norm_out = norm_out_ptr;
    params.residual_out_buffer = residual_out.mutable_data_ptr();
    params.hidden_size = static_cast<int>(k);
    params.eps = static_cast<float>(eps);
    params.elts_total = hidden_states.numel();
    params.sf_layout = tensorrt_llm::QuantizationSFLayout::SWIZZLED;

    auto const stream = at::cuda::getCurrentCUDAStream(hidden_states.get_device());
    auto const dtype = tensorrt_llm::runtime::TorchUtils::dataType(hidden_states.scalar_type());

    tensorrt_llm::kernels::residualRmsNormFp4Quant(params, dtype, stream);

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
    params.intermediate_buffer = hidden_states.data_ptr();
    params.scale_factor_ptr = static_cast<float const*>(scale_factor.data_ptr());
    params.quant_out = quant_out.mutable_data_ptr();
    params.scale_out = scale_out.mutable_data_ptr();
    params.norm_out = norm_out_ptr;
    params.hidden_size = static_cast<int>(k);
    params.eps = static_cast<float>(eps);
    params.elts_total = hidden_states.numel();
    params.sf_layout = tensorrt_llm::QuantizationSFLayout::SWIZZLED;
    params.input_row_stride = input_row_stride;

    auto const stream = at::cuda::getCurrentCUDAStream(hidden_states.get_device());
    auto const dtype = tensorrt_llm::runtime::TorchUtils::dataType(hidden_states.scalar_type());

    tensorrt_llm::kernels::residualRmsNormFp4Quant(params, dtype, stream);

    if (return_norm_out)
    {
        return {norm_out, quant_out, scale_out};
    }
    return {quant_out, scale_out};
}

std::tuple<at::Tensor, at::Tensor> pad_nvfp4_activation(
    at::Tensor const& fp4_input, at::Tensor const& sf_input, int64_t padded_hidden_size)
{
    CHECK_TH_CUDA(fp4_input);
    CHECK_CONTIGUOUS(fp4_input);
    CHECK_TH_CUDA(sf_input);
    CHECK_CONTIGUOUS(sf_input);
    TORCH_CHECK(fp4_input.get_device() == sf_input.get_device(), "fp4_input and sf_input must be on the same device");
    TORCH_CHECK(fp4_input.dim() == 2, "fp4_input must have shape [num_tokens, hidden_size / 2]");
    TORCH_CHECK(sf_input.dim() == 2, "sf_input must have shape [num_tokens, hidden_size / 16]");
    TORCH_CHECK(fp4_input.size(0) == sf_input.size(0), "fp4_input and sf_input must have the same row count");
    TORCH_CHECK(fp4_input.element_size() == 1, "fp4_input elements must occupy one byte");
    TORCH_CHECK(sf_input.element_size() == 1, "sf_input elements must occupy one byte");

    constexpr int64_t kValuesPerFp4Byte = 2;
    constexpr int64_t kSfVecSize = 16;
    int64_t const hidden_size = fp4_input.size(1) * kValuesPerFp4Byte;
    TORCH_CHECK(sf_input.size(1) * kSfVecSize == hidden_size,
        "fp4_input and sf_input must describe the same logical hidden size");
    TORCH_CHECK(padded_hidden_size >= hidden_size, "padded_hidden_size must not be smaller than hidden_size");
    TORCH_CHECK(padded_hidden_size % kSfVecSize == 0, "padded_hidden_size must be divisible by 16");
    TORCH_CHECK(padded_hidden_size <= std::numeric_limits<int>::max(), "padded_hidden_size exceeds INT_MAX");
    TORCH_CHECK(fp4_input.size(0) <= std::numeric_limits<int>::max(), "num_tokens exceeds CUDA grid limit");

    at::Tensor fp4_output = at::detail::empty_cuda({fp4_input.size(0), padded_hidden_size / kValuesPerFp4Byte},
        fp4_input.scalar_type(), fp4_input.device(), std::nullopt);
    at::Tensor sf_output = at::detail::empty_cuda(
        {sf_input.size(0), padded_hidden_size / kSfVecSize}, sf_input.scalar_type(), sf_input.device(), std::nullopt);

    auto const stream = at::cuda::getCurrentCUDAStream(fp4_input.get_device());
    tensorrt_llm::kernels::padNvfp4Activation(fp4_input.const_data_ptr(), sf_input.const_data_ptr(),
        fp4_output.mutable_data_ptr(), sf_output.mutable_data_ptr(), fp4_input.size(0), static_cast<int>(hidden_size),
        static_cast<int>(padded_hidden_size), stream);
    return {fp4_output, sf_output};
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
    m.def(
        "pad_nvfp4_activation("
        "Tensor fp4_input,"
        "Tensor sf_input,"
        "int padded_hidden_size) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_add_rmsnorm_fp4_quantize", &tensorrt_llm::torch_ext::fused_add_rmsnorm_fp4_quantize);
    m.impl("fused_rmsnorm_fp4_quantize", &tensorrt_llm::torch_ext::fused_rmsnorm_fp4_quantize);
    m.impl("pad_nvfp4_activation", &tensorrt_llm::torch_ext::pad_nvfp4_activation);
}
