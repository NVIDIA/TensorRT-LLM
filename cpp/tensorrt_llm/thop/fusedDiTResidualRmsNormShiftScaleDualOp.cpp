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

#include "tensorrt_llm/kernels/fusedDiTNormKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Fused residual + weightless RMSNorm + DUAL AdaLN modulation .
//
// Each of the 4 modulators is built inline by the C++ op:
//   m[b,d] = m_table[d].to(bf16) + m_ts[b,d]    (bf16 narrow first, bf16 hw add)
// Matches PyTorch eager `_get_av_ca_ada_values` semantics. This folds the
// previously separate broadcast-add prep kernel into Phase 0b.
//
// In-place:
//   x   <- x + attn2_out
// Computes:
//   normed     = rsqrt(mean(x_new^2) + eps) * x_new          (over last dim)
//   out_dir1[b,s,d] = normed[b,s,d] * (1 + scale_dir1[b,d]) + shift_dir1[b,d]
//   out_dir2[b,s,d] = normed[b,s,d] * (1 + scale_dir2[b,d]) + shift_dir2[b,d]
//
// Inputs:
//   x:         bf16, [..., D] contig, MUTATED in-place
//   attn2_out: bf16, same shape as x
//   scale_dir1_table, shift_dir1_table, scale_dir2_table, shift_dir2_table: fp32 [D]
//   scale_dir1_ts,    shift_dir1_ts,    scale_dir2_ts,    shift_dir2_ts:    bf16 [..., D], inner stride 1
//   eps:       RMSNorm epsilon
//
// Returns (out_dir1, out_dir2), both bf16 [..., D].
std::tuple<torch::Tensor, torch::Tensor> fused_dit_resid_rms_shift_scale_dual(torch::Tensor x,
    torch::Tensor const& attn2_out, torch::Tensor const& scale_dir1_table, torch::Tensor const& scale_dir1_ts,
    torch::Tensor const& shift_dir1_table, torch::Tensor const& shift_dir1_ts, torch::Tensor const& scale_dir2_table,
    torch::Tensor const& scale_dir2_ts, torch::Tensor const& shift_dir2_table, torch::Tensor const& shift_dir2_ts,
    double eps)
{
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims; got ", x.dim());
    TORCH_CHECK(attn2_out.dim() >= 2, "attn2_out must have at least 2 dims; got ", attn2_out.dim());
    TORCH_CHECK(x.sizes() == attn2_out.sizes(), "x and attn2_out shape mismatch");
    TORCH_CHECK(scale_dir1_table.dim() == 1 && shift_dir1_table.dim() == 1 && scale_dir2_table.dim() == 1
            && shift_dir2_table.dim() == 1,
        "All 4 modulator tables must be 1-D [D]");
    TORCH_CHECK(scale_dir1_table.size(0) == x.size(-1) && shift_dir1_table.size(0) == x.size(-1)
            && scale_dir2_table.size(0) == x.size(-1) && shift_dir2_table.size(0) == x.size(-1),
        "All modulator tables must have size == x last dim");
    TORCH_CHECK(scale_dir1_ts.size(-1) == x.size(-1) && shift_dir1_ts.size(-1) == x.size(-1)
            && scale_dir2_ts.size(-1) == x.size(-1) && shift_dir2_ts.size(-1) == x.size(-1),
        "All modulator ts last dim must equal x last dim");

    CHECK_INPUT(x, torch::kBFloat16);
    CHECK_INPUT(attn2_out, torch::kBFloat16);
    CHECK_TYPE(scale_dir1_table, torch::kFloat32);
    CHECK_TYPE(shift_dir1_table, torch::kFloat32);
    CHECK_TYPE(scale_dir2_table, torch::kFloat32);
    CHECK_TYPE(shift_dir2_table, torch::kFloat32);
    CHECK_TYPE(scale_dir1_ts, torch::kBFloat16);
    CHECK_TYPE(shift_dir1_ts, torch::kBFloat16);
    CHECK_TYPE(scale_dir2_ts, torch::kBFloat16);
    CHECK_TYPE(shift_dir2_ts, torch::kBFloat16);
    TORCH_CHECK(scale_dir1_table.is_contiguous() && shift_dir1_table.is_contiguous() && scale_dir2_table.is_contiguous()
            && shift_dir2_table.is_contiguous(),
        "All modulator tables must be contiguous");
    TORCH_CHECK(scale_dir1_ts.stride(-1) == 1, "scale_dir1_ts must have inner-dim stride 1");
    TORCH_CHECK(shift_dir1_ts.stride(-1) == 1, "shift_dir1_ts must have inner-dim stride 1");
    TORCH_CHECK(scale_dir2_ts.stride(-1) == 1, "scale_dir2_ts must have inner-dim stride 1");
    TORCH_CHECK(shift_dir2_ts.stride(-1) == 1, "shift_dir2_ts must have inner-dim stride 1");

    int64_t const hidden_dim = x.size(-1);
    int64_t const num_tokens = x.numel() / hidden_dim;
    int64_t const n_rows = scale_dir1_ts.numel() / hidden_dim;

    TORCH_CHECK(shift_dir1_ts.numel() / hidden_dim == n_rows, "shift_dir1_ts row count mismatch");
    TORCH_CHECK(scale_dir2_ts.numel() / hidden_dim == n_rows, "scale_dir2_ts row count mismatch");
    TORCH_CHECK(shift_dir2_ts.numel() / hidden_dim == n_rows, "shift_dir2_ts row count mismatch");
    TORCH_CHECK(num_tokens % n_rows == 0, "num_tokens (", num_tokens, ") must be divisible by n_rows (", n_rows, ")");

    int64_t const tokens_per_batch = num_tokens / n_rows;
    auto ts_stride_of = [&](torch::Tensor const& t) -> int64_t { return (n_rows > 1) ? t.stride(-2) : hidden_dim; };

    auto out_dir1 = torch::empty_like(x);
    auto out_dir2 = torch::empty_like(x);

    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

    tensorrt_llm::kernels::AdaLNNormParams params;
    params.x = reinterpret_cast<__nv_bfloat16*>(x.data_ptr());
    params.attn = reinterpret_cast<__nv_bfloat16 const*>(attn2_out.data_ptr());
    params.scale_table[0] = reinterpret_cast<float const*>(scale_dir1_table.data_ptr());
    params.scale_ts[0] = reinterpret_cast<__nv_bfloat16 const*>(scale_dir1_ts.data_ptr());
    params.scale_ts_stride[0] = static_cast<int>(ts_stride_of(scale_dir1_ts));
    params.shift_table[0] = reinterpret_cast<float const*>(shift_dir1_table.data_ptr());
    params.shift_ts[0] = reinterpret_cast<__nv_bfloat16 const*>(shift_dir1_ts.data_ptr());
    params.shift_ts_stride[0] = static_cast<int>(ts_stride_of(shift_dir1_ts));
    params.scale_table[1] = reinterpret_cast<float const*>(scale_dir2_table.data_ptr());
    params.scale_ts[1] = reinterpret_cast<__nv_bfloat16 const*>(scale_dir2_ts.data_ptr());
    params.scale_ts_stride[1] = static_cast<int>(ts_stride_of(scale_dir2_ts));
    params.shift_table[1] = reinterpret_cast<float const*>(shift_dir2_table.data_ptr());
    params.shift_ts[1] = reinterpret_cast<__nv_bfloat16 const*>(shift_dir2_ts.data_ptr());
    params.shift_ts_stride[1] = static_cast<int>(ts_stride_of(shift_dir2_ts));
    params.out_bf16[0] = reinterpret_cast<__nv_bfloat16*>(out_dir1.data_ptr());
    params.out_bf16[1] = reinterpret_cast<__nv_bfloat16*>(out_dir2.data_ptr());
    params.num_tokens = static_cast<int>(num_tokens);
    params.tokens_per_batch = static_cast<int>(tokens_per_batch);
    params.eps = static_cast<float>(eps);
    tensorrt_llm::kernels::launchFusedDiTNorm</*HAS_RESIDUAL=*/true, /*HAS_GATE=*/false, /*HAS_MODULATE=*/true,
        /*NUM_OUT=*/2, /*HAS_QUANT=*/false>(params, static_cast<int>(hidden_dim), stream);

    return std::make_tuple(out_dir1, out_dir2);
}

// Same op as above, but emits two FP4 + SF pairs directly. Replaces the bf16 (out_dir1, out_dir2)
// pair plus the two standalone tunable_fp4_quantize calls that production currently runs.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> fused_dit_resid_rms_shift_scale_dual_quant(
    torch::Tensor x, torch::Tensor const& attn2_out, torch::Tensor const& scale_dir1_table,
    torch::Tensor const& scale_dir1_ts, torch::Tensor const& shift_dir1_table, torch::Tensor const& shift_dir1_ts,
    torch::Tensor const& scale_dir2_table, torch::Tensor const& scale_dir2_ts, torch::Tensor const& shift_dir2_table,
    torch::Tensor const& shift_dir2_ts, torch::Tensor const& sf_scale1, torch::Tensor const& sf_scale2, double eps)
{
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims; got ", x.dim());
    TORCH_CHECK(attn2_out.dim() >= 2, "attn2_out must have at least 2 dims; got ", attn2_out.dim());
    TORCH_CHECK(x.sizes() == attn2_out.sizes(), "x and attn2_out shape mismatch");
    TORCH_CHECK(scale_dir1_table.dim() == 1 && shift_dir1_table.dim() == 1 && scale_dir2_table.dim() == 1
            && shift_dir2_table.dim() == 1,
        "All 4 modulator tables must be 1-D [D]");
    TORCH_CHECK(scale_dir1_table.size(0) == x.size(-1) && shift_dir1_table.size(0) == x.size(-1)
            && scale_dir2_table.size(0) == x.size(-1) && shift_dir2_table.size(0) == x.size(-1),
        "All modulator tables must have size == x last dim");
    TORCH_CHECK(scale_dir1_ts.size(-1) == x.size(-1) && shift_dir1_ts.size(-1) == x.size(-1)
            && scale_dir2_ts.size(-1) == x.size(-1) && shift_dir2_ts.size(-1) == x.size(-1),
        "All modulator ts last dim must equal x last dim");

    CHECK_INPUT(x, torch::kBFloat16);
    CHECK_INPUT(attn2_out, torch::kBFloat16);
    CHECK_TYPE(scale_dir1_table, torch::kFloat32);
    CHECK_TYPE(shift_dir1_table, torch::kFloat32);
    CHECK_TYPE(scale_dir2_table, torch::kFloat32);
    CHECK_TYPE(shift_dir2_table, torch::kFloat32);
    CHECK_TYPE(scale_dir1_ts, torch::kBFloat16);
    CHECK_TYPE(shift_dir1_ts, torch::kBFloat16);
    CHECK_TYPE(scale_dir2_ts, torch::kBFloat16);
    CHECK_TYPE(shift_dir2_ts, torch::kBFloat16);
    CHECK_TYPE(sf_scale1, torch::kFloat32);
    CHECK_TYPE(sf_scale2, torch::kFloat32);
    TORCH_CHECK(scale_dir1_table.is_contiguous() && shift_dir1_table.is_contiguous() && scale_dir2_table.is_contiguous()
            && shift_dir2_table.is_contiguous(),
        "All modulator tables must be contiguous");
    TORCH_CHECK(scale_dir1_ts.stride(-1) == 1, "scale_dir1_ts must have inner-dim stride 1");
    TORCH_CHECK(shift_dir1_ts.stride(-1) == 1, "shift_dir1_ts must have inner-dim stride 1");
    TORCH_CHECK(scale_dir2_ts.stride(-1) == 1, "scale_dir2_ts must have inner-dim stride 1");
    TORCH_CHECK(shift_dir2_ts.stride(-1) == 1, "shift_dir2_ts must have inner-dim stride 1");
    TORCH_CHECK(sf_scale1.numel() >= 1 && sf_scale2.numel() >= 1, "sf_scale tensors must contain >= 1 element");

    int64_t const hidden_dim = x.size(-1);
    int64_t const num_tokens = x.numel() / hidden_dim;
    int64_t const n_rows = scale_dir1_ts.numel() / hidden_dim;

    TORCH_CHECK(shift_dir1_ts.numel() / hidden_dim == n_rows, "shift_dir1_ts row count mismatch");
    TORCH_CHECK(scale_dir2_ts.numel() / hidden_dim == n_rows, "scale_dir2_ts row count mismatch");
    TORCH_CHECK(shift_dir2_ts.numel() / hidden_dim == n_rows, "shift_dir2_ts row count mismatch");
    TORCH_CHECK(num_tokens % n_rows == 0, "num_tokens (", num_tokens, ") must be divisible by n_rows (", n_rows, ")");
    TORCH_CHECK(hidden_dim % 16 == 0, "hidden_dim must be divisible by 16 (NVFP4 group size)");

    int64_t const tokens_per_batch = num_tokens / n_rows;
    auto ts_stride_of = [&](torch::Tensor const& t) -> int64_t { return (n_rows > 1) ? t.stride(-2) : hidden_dim; };

    auto opt_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(x.device());
    // SF: 128x4 SWIZZLED layout, padded rows to multiple of 128 and cols to multiple of 4.
    int64_t const sf_cols = hidden_dim / 16;
    int64_t const padded_rows = (num_tokens + 127) / 128 * 128;
    int64_t const padded_cols = (sf_cols + 3) / 4 * 4;
    int64_t const sf_numel = padded_rows * padded_cols;
    auto out1_fp4 = torch::empty({num_tokens, hidden_dim / 2}, opt_u8);
    auto out1_sf = torch::empty({sf_numel}, opt_u8);
    auto out2_fp4 = torch::empty({num_tokens, hidden_dim / 2}, opt_u8);
    auto out2_sf = torch::empty({sf_numel}, opt_u8);

    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

    tensorrt_llm::kernels::AdaLNNormParams params;
    params.x = reinterpret_cast<__nv_bfloat16*>(x.data_ptr());
    params.attn = reinterpret_cast<__nv_bfloat16 const*>(attn2_out.data_ptr());
    params.scale_table[0] = reinterpret_cast<float const*>(scale_dir1_table.data_ptr());
    params.scale_ts[0] = reinterpret_cast<__nv_bfloat16 const*>(scale_dir1_ts.data_ptr());
    params.scale_ts_stride[0] = static_cast<int>(ts_stride_of(scale_dir1_ts));
    params.shift_table[0] = reinterpret_cast<float const*>(shift_dir1_table.data_ptr());
    params.shift_ts[0] = reinterpret_cast<__nv_bfloat16 const*>(shift_dir1_ts.data_ptr());
    params.shift_ts_stride[0] = static_cast<int>(ts_stride_of(shift_dir1_ts));
    params.scale_table[1] = reinterpret_cast<float const*>(scale_dir2_table.data_ptr());
    params.scale_ts[1] = reinterpret_cast<__nv_bfloat16 const*>(scale_dir2_ts.data_ptr());
    params.scale_ts_stride[1] = static_cast<int>(ts_stride_of(scale_dir2_ts));
    params.shift_table[1] = reinterpret_cast<float const*>(shift_dir2_table.data_ptr());
    params.shift_ts[1] = reinterpret_cast<__nv_bfloat16 const*>(shift_dir2_ts.data_ptr());
    params.shift_ts_stride[1] = static_cast<int>(ts_stride_of(shift_dir2_ts));
    params.out_fp4[0] = reinterpret_cast<uint32_t*>(out1_fp4.data_ptr());
    params.out_sf[0] = reinterpret_cast<uint32_t*>(out1_sf.data_ptr());
    params.sf_scale[0] = reinterpret_cast<float const*>(sf_scale1.data_ptr());
    params.out_fp4[1] = reinterpret_cast<uint32_t*>(out2_fp4.data_ptr());
    params.out_sf[1] = reinterpret_cast<uint32_t*>(out2_sf.data_ptr());
    params.sf_scale[1] = reinterpret_cast<float const*>(sf_scale2.data_ptr());
    params.num_tokens = static_cast<int>(num_tokens);
    params.tokens_per_batch = static_cast<int>(tokens_per_batch);
    params.eps = static_cast<float>(eps);
    tensorrt_llm::kernels::launchFusedDiTNorm</*HAS_RESIDUAL=*/true, /*HAS_GATE=*/false, /*HAS_MODULATE=*/true,
        /*NUM_OUT=*/2, /*HAS_QUANT=*/true>(params, static_cast<int>(hidden_dim), stream);

    return std::make_tuple(out1_fp4, out1_sf, out2_fp4, out2_sf);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_dit_resid_rms_shift_scale_dual("
        "Tensor(a!) x, Tensor attn2_out, "
        "Tensor scale_dir1_table, Tensor scale_dir1_ts, "
        "Tensor shift_dir1_table, Tensor shift_dir1_ts, "
        "Tensor scale_dir2_table, Tensor scale_dir2_ts, "
        "Tensor shift_dir2_table, Tensor shift_dir2_ts, "
        "float eps) -> (Tensor, Tensor)");
    m.def(
        "fused_dit_resid_rms_shift_scale_dual_quant("
        "Tensor(a!) x, Tensor attn2_out, "
        "Tensor scale_dir1_table, Tensor scale_dir1_ts, "
        "Tensor shift_dir1_table, Tensor shift_dir1_ts, "
        "Tensor scale_dir2_table, Tensor scale_dir2_ts, "
        "Tensor shift_dir2_table, Tensor shift_dir2_ts, "
        "Tensor sf_scale1, Tensor sf_scale2, "
        "float eps) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_dit_resid_rms_shift_scale_dual", &fused_dit_resid_rms_shift_scale_dual);
    m.impl("fused_dit_resid_rms_shift_scale_dual_quant", &fused_dit_resid_rms_shift_scale_dual_quant);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
