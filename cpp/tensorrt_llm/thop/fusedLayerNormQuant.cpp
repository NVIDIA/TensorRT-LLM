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

#include "tensorrt_llm/kernels/fusedLayerNormQuant/fusedLayerNormQuant.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <optional>
#include <tuple>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

/*
 * Fused LayerNorm + NVFP4 quantization.
 *
 * Supports two mutually exclusive configurations:
 *   - AdaLN modulation (norm1, norm3 in Wan 2.2):
 *       provide scale_msa and shift_msa; ln_weight and ln_bias must be None.
 *       Output: y = ((x - mean) * rstd) * (1 + scale_msa) + shift_msa, quantized to FP4.
 *   - Plain LN affine (norm2 in Wan 2.2):
 *       provide ln_weight and ln_bias; scale_msa and shift_msa must be None.
 *       Output: y = ((x - mean) * rstd) * ln_weight + ln_bias, quantized to FP4.
 *   - Both None: plain LN (no affine, no modulation), then quantize.
 *
 * Shapes:
 *   input:      [M, N]   bf16 or fp16, contiguous. (M = B * seq_len_per_batch)
 *   ln_weight:  [N]      bf16 or fp16 (must match input dtype). Optional.
 *   ln_bias:    [N]      bf16 or fp16. Optional.
 *   scale_msa:  [B, N]   bf16 or fp16. Optional. Broadcasts: row r reads batch r / seq_len_per_batch.
 *   shift_msa:  [B, N]   bf16 or fp16. Optional.
 *   sf_scale:   scalar   fp32 (calibrated module.input_scale of the downstream Linear).
 * Returns:
 *   output_fp4: [M, N/2] uint8 (each byte = 2 packed FP4 nibbles).
 *   output_sf:  [sf_size] uint8, swizzled layout (each byte is one FP8 e4m3 scale).
 *
 * v1 supports only N = 5120 (Wan 2.2 hidden_size). Other N values will trigger a TORCH_CHECK.
 */
std::tuple<at::Tensor, at::Tensor> fused_layernorm_quantize(at::Tensor const& input,
    std::optional<at::Tensor> const& ln_weight, std::optional<at::Tensor> const& ln_bias,
    std::optional<at::Tensor> const& scale_msa, std::optional<at::Tensor> const& shift_msa, at::Tensor const& sf_scale,
    int64_t seq_len_per_batch, double eps, int64_t sf_vec_size)
{
    CHECK_TH_CUDA(input);
    CHECK_CONTIGUOUS(input);
    CHECK_INPUT(sf_scale, torch::kFloat32);
    TORCH_CHECK(sf_scale.numel() == 1, "sf_scale must contain exactly one element (got %ld).", sf_scale.numel());

    TORCH_CHECK(sf_vec_size == 16, "sf_vec_size must be 16 for NVFP4.");

    auto const& inputShape = input.sizes();
    TORCH_CHECK(inputShape.size() == 2, "input should be 2D tensor [M, N].");
    int64_t const m = inputShape[0];
    int64_t const n = inputShape[1];
    // v1 of the fused kernel hardcodes N=5120 (Wan 2.2 hidden_size); the
    // launcher's kN_HARDCODED is baked into the kernel template. Enforce
    // here so unsupported hidden sizes hit a clear TORCH_CHECK rather than
    // a generic kernel assert.
    int64_t constexpr kSupportedHiddenSize = 5120;
    TORCH_CHECK(n == kSupportedHiddenSize,
        "fused_layernorm_quantize v1 supports only N=%ld (Wan 2.2 hidden_size); got N=%ld.", kSupportedHiddenSize, n);
    TORCH_CHECK(n % sf_vec_size == 0, "N must be divisible by sf_vec_size (16).");

    bool const has_ln_affine = ln_weight.has_value() && ln_bias.has_value();
    bool const has_modulation = scale_msa.has_value() && shift_msa.has_value();
    TORCH_CHECK(
        !(has_ln_affine && has_modulation), "ln_weight/ln_bias and scale_msa/shift_msa are mutually exclusive.");
    TORCH_CHECK(ln_weight.has_value() == ln_bias.has_value(),
        "ln_weight and ln_bias must be provided together (or both omitted).");
    TORCH_CHECK(scale_msa.has_value() == shift_msa.has_value(),
        "scale_msa and shift_msa must be provided together (or both omitted).");

    if (has_ln_affine)
    {
        auto const& w = ln_weight.value();
        auto const& b = ln_bias.value();
        CHECK_TH_CUDA(w);
        CHECK_CONTIGUOUS(w);
        CHECK_TH_CUDA(b);
        CHECK_CONTIGUOUS(b);
        TORCH_CHECK(w.sizes().size() == 1 && w.sizes()[0] == n, "ln_weight must be 1D of length N.");
        TORCH_CHECK(b.sizes().size() == 1 && b.sizes()[0] == n, "ln_bias must be 1D of length N.");
        TORCH_CHECK(w.scalar_type() == input.scalar_type(), "ln_weight dtype must match input dtype.");
        TORCH_CHECK(b.scalar_type() == input.scalar_type(), "ln_bias dtype must match input dtype.");
    }

    if (has_modulation)
    {
        auto const& s = scale_msa.value();
        auto const& sh = shift_msa.value();
        CHECK_TH_CUDA(s);
        CHECK_CONTIGUOUS(s);
        CHECK_TH_CUDA(sh);
        CHECK_CONTIGUOUS(sh);
        TORCH_CHECK(s.sizes().size() == 2 && s.sizes()[1] == n, "scale_msa must be 2D [B, N].");
        TORCH_CHECK(sh.sizes().size() == 2 && sh.sizes()[1] == n, "shift_msa must be 2D [B, N].");
        TORCH_CHECK(s.sizes()[0] == sh.sizes()[0], "scale_msa and shift_msa must share batch dim.");
        TORCH_CHECK(s.scalar_type() == input.scalar_type(), "scale_msa dtype must match input dtype.");
        TORCH_CHECK(sh.scalar_type() == input.scalar_type(), "shift_msa dtype must match input dtype.");
        TORCH_CHECK(seq_len_per_batch > 0, "seq_len_per_batch must be positive when modulation is used.");
        TORCH_CHECK(m % seq_len_per_batch == 0, "M must be divisible by seq_len_per_batch.");
        TORCH_CHECK(s.sizes()[0] == m / seq_len_per_batch, "scale_msa batch dim must equal M / seq_len_per_batch.");
    }

    // Allocate output tensors.
    at::Tensor output_fp4 = at::detail::empty_cuda({m, n / 2}, torch::kUInt8, input.device(), std::nullopt);
    int64_t const sfSize = tensorrt_llm::computeSwizzledLayoutSFSize(m, n / sf_vec_size);
    at::Tensor output_sf = at::detail::empty_cuda({sfSize}, SF_DTYPE, input.device(), std::nullopt);

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

#define LAUNCH_FUSED_LAYERNORM_QUANT(T)                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        tensorrt_llm::kernels::FusedLayerNormQuantParams<T> params{};                                                  \
        params.x = reinterpret_cast<T const*>(input.data_ptr());                                                       \
        params.ln_weight = has_ln_affine ? reinterpret_cast<T const*>(ln_weight.value().data_ptr()) : nullptr;         \
        params.ln_bias = has_ln_affine ? reinterpret_cast<T const*>(ln_bias.value().data_ptr()) : nullptr;             \
        params.scale_msa = has_modulation ? reinterpret_cast<T const*>(scale_msa.value().data_ptr()) : nullptr;        \
        params.shift_msa = has_modulation ? reinterpret_cast<T const*>(shift_msa.value().data_ptr()) : nullptr;        \
        params.y_fp4 = reinterpret_cast<uint32_t*>(output_fp4.data_ptr());                                             \
        params.sf_out = reinterpret_cast<uint32_t*>(output_sf.data_ptr());                                             \
        params.sf_scale = sf_scale.data_ptr<float>();                                                                  \
        params.M = static_cast<int>(m);                                                                                \
        params.N = static_cast<int>(n);                                                                                \
        params.seq_len_per_batch = static_cast<int>(seq_len_per_batch);                                                \
        params.has_ln_affine = has_ln_affine;                                                                          \
        params.has_modulation = has_modulation;                                                                        \
        params.eps = static_cast<float>(eps);                                                                          \
        params.stream = stream;                                                                                        \
        tensorrt_llm::kernels::invokeFusedLayerNormQuant<T>(params);                                                   \
    } while (0)

    if (input.scalar_type() == at::ScalarType::Half)
    {
        LAUNCH_FUSED_LAYERNORM_QUANT(half);
    }
    else if (input.scalar_type() == at::ScalarType::BFloat16)
    {
#ifdef ENABLE_BF16
        LAUNCH_FUSED_LAYERNORM_QUANT(__nv_bfloat16);
#else
        C10_THROW_ERROR(NotImplementedError, "BFloat16 must be enabled for fused_layernorm_quantize with bf16 input.");
#endif
    }
    else
    {
        C10_THROW_ERROR(NotImplementedError, "fused_layernorm_quantize only supports fp16/bf16 input.");
    }

#undef LAUNCH_FUSED_LAYERNORM_QUANT

    return std::make_tuple(output_fp4, output_sf);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_layernorm_quantize(Tensor input, Tensor? ln_weight, Tensor? ln_bias, "
        "Tensor? scale_msa, Tensor? shift_msa, Tensor sf_scale, int seq_len_per_batch=1, "
        "float eps=1e-6, int sf_vec_size=16) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_layernorm_quantize", &tensorrt_llm::torch_ext::fused_layernorm_quantize);
}
