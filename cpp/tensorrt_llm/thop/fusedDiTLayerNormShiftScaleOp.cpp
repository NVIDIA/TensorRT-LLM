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

#include "tensorrt_llm/kernels/fusedDiTLayerNormShiftScaleKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

// Validate and fill common fields of DiTLayerNormShiftScaleParams.
// Returns {M, D, has_ln_affine, has_modulation}.
struct ValidatedParams
{
    int64_t M;
    int64_t D;
    bool has_ln_affine;
    bool has_modulation;
};

ValidatedParams validateInputs(at::Tensor const& x, std::optional<at::Tensor> const& ln_weight,
    std::optional<at::Tensor> const& ln_bias, std::optional<at::Tensor> const& scale_msa,
    std::optional<at::Tensor> const& shift_msa, int64_t seq_len_per_batch)
{
    CHECK_TH_CUDA(x);
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [M, D], got ", x.dim(), " dims");
    CHECK_TYPE(x, torch::kBFloat16);

    int64_t const M = x.size(0);
    int64_t const D = x.size(1);
    TORCH_CHECK(D == 5120, "fused_dit_layernorm_shift_scale only supports D=5120 (got ", D, ")");
    TORCH_CHECK(D % 16 == 0, "D must be divisible by 16 (NVFP4 group size)");

    bool const has_ln_affine = ln_weight.has_value();
    bool const has_modulation = scale_msa.has_value();

    TORCH_CHECK(!(has_ln_affine && has_modulation),
        "fused_dit_layernorm_shift_scale: ln_weight/ln_bias and scale_msa/shift_msa are mutually exclusive");

    if (has_ln_affine)
    {
        TORCH_CHECK(
            ln_weight.has_value() && ln_bias.has_value(), "ln_weight and ln_bias must both be provided together");
        TORCH_CHECK(ln_weight->dim() == 1 && ln_weight->size(0) == D, "ln_weight must be 1D [D]");
        TORCH_CHECK(ln_bias->dim() == 1 && ln_bias->size(0) == D, "ln_bias must be 1D [D]");
        CHECK_TYPE(ln_weight.value(), torch::kBFloat16);
        CHECK_TYPE(ln_bias.value(), torch::kBFloat16);
        TORCH_CHECK(ln_weight->is_contiguous(), "ln_weight must be contiguous");
        TORCH_CHECK(ln_bias->is_contiguous(), "ln_bias must be contiguous");
    }

    if (has_modulation)
    {
        TORCH_CHECK(
            scale_msa.has_value() && shift_msa.has_value(), "scale_msa and shift_msa must both be provided together");
        TORCH_CHECK(seq_len_per_batch > 0, "seq_len_per_batch must be positive when using AdaLN modulation");
        TORCH_CHECK(
            M % seq_len_per_batch == 0, "M (", M, ") must be divisible by seq_len_per_batch (", seq_len_per_batch, ")");
        int64_t const B = M / seq_len_per_batch;
        TORCH_CHECK(scale_msa->dim() == 2 && scale_msa->size(0) == B && scale_msa->size(1) == D,
            "scale_msa must be [B, D] = [", B, ", ", D, "]");
        TORCH_CHECK(shift_msa->dim() == 2 && shift_msa->size(0) == B && shift_msa->size(1) == D,
            "shift_msa must be [B, D] = [", B, ", ", D, "]");
        CHECK_TYPE(scale_msa.value(), torch::kBFloat16);
        CHECK_TYPE(shift_msa.value(), torch::kBFloat16);
        TORCH_CHECK(scale_msa->is_contiguous(), "scale_msa must be contiguous");
        TORCH_CHECK(shift_msa->is_contiguous(), "shift_msa must be contiguous");
    }

    return {M, D, has_ln_affine, has_modulation};
}

} // namespace

at::Tensor fused_dit_layernorm_shift_scale(at::Tensor const& x, std::optional<at::Tensor> const& ln_weight,
    std::optional<at::Tensor> const& ln_bias, std::optional<at::Tensor> const& scale_msa,
    std::optional<at::Tensor> const& shift_msa, int64_t seq_len_per_batch, double eps)
{
    auto const v = validateInputs(x, ln_weight, ln_bias, scale_msa, shift_msa, seq_len_per_batch);

    at::Tensor out = torch::empty_like(x);

    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

    tensorrt_llm::kernels::DiTLayerNormShiftScaleParams params;
    params.x = reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr());
    params.ln_weight = v.has_ln_affine ? reinterpret_cast<__nv_bfloat16 const*>(ln_weight->data_ptr()) : nullptr;
    params.ln_bias = v.has_ln_affine ? reinterpret_cast<__nv_bfloat16 const*>(ln_bias->data_ptr()) : nullptr;
    params.scale_msa = v.has_modulation ? reinterpret_cast<__nv_bfloat16 const*>(scale_msa->data_ptr()) : nullptr;
    params.shift_msa = v.has_modulation ? reinterpret_cast<__nv_bfloat16 const*>(shift_msa->data_ptr()) : nullptr;
    params.out_bf16 = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    params.out_fp4 = nullptr;
    params.out_sf = nullptr;
    params.sf_scale = nullptr;
    params.M = static_cast<int>(v.M);
    params.D = static_cast<int>(v.D);
    params.seq_len_per_batch = static_cast<int>(seq_len_per_batch);
    params.eps = static_cast<float>(eps);

    tensorrt_llm::kernels::launchFusedDiTLayerNormShiftScaleKernel(
        params, v.has_ln_affine, v.has_modulation, /*has_quant=*/false, stream);

    return out;
}

std::tuple<at::Tensor, at::Tensor> fused_dit_layernorm_shift_scale_quant(at::Tensor const& x,
    std::optional<at::Tensor> const& ln_weight, std::optional<at::Tensor> const& ln_bias,
    std::optional<at::Tensor> const& scale_msa, std::optional<at::Tensor> const& shift_msa, at::Tensor const& sf_scale,
    int64_t seq_len_per_batch, double eps)
{
    auto const v = validateInputs(x, ln_weight, ln_bias, scale_msa, shift_msa, seq_len_per_batch);

    CHECK_INPUT(sf_scale, torch::kFloat32);
    TORCH_CHECK(sf_scale.numel() == 1, "sf_scale must be a scalar tensor (1 element), got numel=", sf_scale.numel());

    int64_t const M = v.M;
    int64_t const D = v.D;

    // y_fp4: [M, D/2] uint8 (2 FP4 nibbles per byte).
    auto const opt_u8 = torch::TensorOptions().dtype(FLOAT4_E2M1X2).device(x.device());
    at::Tensor y_fp4 = torch::empty({M, D / 2}, opt_u8);

    // sf_out: swizzled NVFP4 scale factors.  The layout tiles are 128×4 in (tokens, sf_cols),
    // so we round up both dimensions before allocating.
    int64_t const sf_cols = D / 16;
    int64_t const sfSize = (M + 127) / 128 * 128 * (sf_cols + 3) / 4 * 4;
    at::Tensor sf_out = torch::empty({sfSize}, torch::TensorOptions().dtype(SF_DTYPE).device(x.device()));

    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

    tensorrt_llm::kernels::DiTLayerNormShiftScaleParams params;
    params.x = reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr());
    params.ln_weight = v.has_ln_affine ? reinterpret_cast<__nv_bfloat16 const*>(ln_weight->data_ptr()) : nullptr;
    params.ln_bias = v.has_ln_affine ? reinterpret_cast<__nv_bfloat16 const*>(ln_bias->data_ptr()) : nullptr;
    params.scale_msa = v.has_modulation ? reinterpret_cast<__nv_bfloat16 const*>(scale_msa->data_ptr()) : nullptr;
    params.shift_msa = v.has_modulation ? reinterpret_cast<__nv_bfloat16 const*>(shift_msa->data_ptr()) : nullptr;
    params.out_bf16 = nullptr;
    params.out_fp4 = reinterpret_cast<uint32_t*>(y_fp4.data_ptr());
    params.out_sf = reinterpret_cast<uint32_t*>(sf_out.data_ptr());
    params.sf_scale = reinterpret_cast<float const*>(sf_scale.data_ptr());
    params.M = static_cast<int>(M);
    params.D = static_cast<int>(D);
    params.seq_len_per_batch = static_cast<int>(seq_len_per_batch);
    params.eps = static_cast<float>(eps);

    tensorrt_llm::kernels::launchFusedDiTLayerNormShiftScaleKernel(
        params, v.has_ln_affine, v.has_modulation, /*has_quant=*/true, stream);

    return std::make_tuple(y_fp4, sf_out);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

// Register the ops with PyTorch.
TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_dit_layernorm_shift_scale(Tensor x, Tensor? ln_weight, Tensor? ln_bias, "
        "Tensor? scale_msa, Tensor? shift_msa, int seq_len_per_batch, float eps) -> Tensor");
    m.def(
        "fused_dit_layernorm_shift_scale_quant(Tensor x, Tensor? ln_weight, Tensor? ln_bias, "
        "Tensor? scale_msa, Tensor? shift_msa, Tensor sf_scale, int seq_len_per_batch, float eps) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_dit_layernorm_shift_scale", &tensorrt_llm::torch_ext::fused_dit_layernorm_shift_scale);
    m.impl("fused_dit_layernorm_shift_scale_quant", &tensorrt_llm::torch_ext::fused_dit_layernorm_shift_scale_quant);
}
