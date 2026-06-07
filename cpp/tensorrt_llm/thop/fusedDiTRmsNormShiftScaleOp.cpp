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

// Fused weight-less RMSNorm + AdaLN affine modulation for LTX-2 DiT.
//
// The modulator is computed inline: scale[d] = scale_table[d] + scale_ts[b, d]
// (and same for shift). This folds the previously-separate broadcast-add prep
// kernel (Inductor leaf POI before this op) into Phase 0b of the GPU kernel.
//
// Math:
//   normed     = rsqrt(mean(x^2) + eps) * x
//   out[b,s,d] = normed[b,s,d] * (1 + scale_table[d] + scale_ts[b,d])
//                              + shift_table[d] + shift_ts[b,d]
//
// Inputs:
//   x:           bf16, [..., D] (any rank with last dim == D)
//   scale_table: fp32, [D]              -- per-block scale_shift_table[slot], broadcast over batch
//   scale_ts:    bf16, last dim == D, inner stride 1; the row stride (stride(-2))
//                lets the caller pass an unbind view of a [..., 6, D] timestep
//                tensor without a .contiguous() copy
//   shift_table: fp32, [D]
//   shift_ts:    bf16, same constraints as scale_ts
//   eps:         RMSNorm epsilon
//
// Returns a new bf16 tensor with the same shape as ``x``.
torch::Tensor fused_dit_rmsnorm_shift_scale(torch::Tensor const& x, torch::Tensor const& scale_table,
    torch::Tensor const& scale_ts, torch::Tensor const& shift_table, torch::Tensor const& shift_ts, double eps)
{
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims; got ", x.dim());
    TORCH_CHECK(scale_table.dim() == 1 && shift_table.dim() == 1, "scale_table and shift_table must be 1-D [D]; got ",
        scale_table.dim(), " and ", shift_table.dim());
    TORCH_CHECK(scale_table.size(0) == x.size(-1) && shift_table.size(0) == x.size(-1),
        "scale_table/shift_table size must equal x last dim");
    TORCH_CHECK(scale_ts.size(-1) == x.size(-1) && shift_ts.size(-1) == x.size(-1),
        "scale_ts/shift_ts last dim must equal x last dim");

    CHECK_INPUT(x, torch::kBFloat16);
    CHECK_TYPE(scale_table, torch::kFloat32);
    CHECK_TYPE(shift_table, torch::kFloat32);
    CHECK_TYPE(scale_ts, torch::kBFloat16);
    CHECK_TYPE(shift_ts, torch::kBFloat16);
    TORCH_CHECK(scale_table.is_contiguous(), "scale_table must be contiguous");
    TORCH_CHECK(shift_table.is_contiguous(), "shift_table must be contiguous");
    TORCH_CHECK(
        scale_ts.stride(-1) == 1, "scale_ts must have inner-dim stride 1; got stride(-1)=", scale_ts.stride(-1));
    TORCH_CHECK(
        shift_ts.stride(-1) == 1, "shift_ts must have inner-dim stride 1; got stride(-1)=", shift_ts.stride(-1));

    int64_t const hidden_dim = x.size(-1);
    int64_t const num_tokens = x.numel() / hidden_dim;
    int64_t const n_rows = scale_ts.numel() / hidden_dim;

    TORCH_CHECK(shift_ts.numel() / hidden_dim == n_rows, "shift_ts row count mismatch");
    TORCH_CHECK(num_tokens % n_rows == 0, "num_tokens (", num_tokens, ") must be divisible by n_rows (", n_rows, ")");

    int64_t const tokens_per_batch = num_tokens / n_rows;
    auto ts_stride_of = [&](torch::Tensor const& t) -> int64_t { return (n_rows > 1) ? t.stride(-2) : hidden_dim; };

    auto out = torch::empty_like(x);

    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

    tensorrt_llm::kernels::AdaLNNormParams params;
    params.x = reinterpret_cast<__nv_bfloat16*>(x.data_ptr());
    params.scale_table[0] = reinterpret_cast<float const*>(scale_table.data_ptr());
    params.scale_ts[0] = reinterpret_cast<__nv_bfloat16 const*>(scale_ts.data_ptr());
    params.scale_ts_stride[0] = static_cast<int>(ts_stride_of(scale_ts));
    params.shift_table[0] = reinterpret_cast<float const*>(shift_table.data_ptr());
    params.shift_ts[0] = reinterpret_cast<__nv_bfloat16 const*>(shift_ts.data_ptr());
    params.shift_ts_stride[0] = static_cast<int>(ts_stride_of(shift_ts));
    params.out_bf16[0] = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    params.num_tokens = static_cast<int>(num_tokens);
    params.tokens_per_batch = static_cast<int>(tokens_per_batch);
    params.eps = static_cast<float>(eps);
    tensorrt_llm::kernels::launchFusedDiTNorm</*HAS_RESIDUAL=*/false, /*HAS_GATE=*/false, /*HAS_MODULATE=*/true,
        /*NUM_OUT=*/1, /*HAS_QUANT=*/false>(params, static_cast<int>(hidden_dim), stream);

    return out;
}

// Same op as above, but emits packed FP4 + 128x4 SWIZZLED SF directly.
// Replaces (out bf16) + a subsequent standalone tunable_fp4_quantize call.
//
// Returns (out_fp4, out_sf):
//   out_fp4: uint8 [num_tokens, D/2]                       (packed FP4)
//   out_sf:  uint8 [pad_up(num_tokens, 128) * pad_up(D/16, 4)]  (128x4 SWIZZLED SF, matches
//                                                                default fp4_quantize / nvfp4_gemm)
std::tuple<torch::Tensor, torch::Tensor> fused_dit_rmsnorm_shift_scale_quant(torch::Tensor const& x,
    torch::Tensor const& scale_table, torch::Tensor const& scale_ts, torch::Tensor const& shift_table,
    torch::Tensor const& shift_ts, torch::Tensor const& sf_scale, double eps)
{
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims; got ", x.dim());
    TORCH_CHECK(scale_table.dim() == 1 && shift_table.dim() == 1, "scale_table and shift_table must be 1-D [D]");
    TORCH_CHECK(scale_table.size(0) == x.size(-1) && shift_table.size(0) == x.size(-1),
        "scale_table/shift_table size must equal x last dim");
    TORCH_CHECK(scale_ts.size(-1) == x.size(-1) && shift_ts.size(-1) == x.size(-1),
        "scale_ts/shift_ts last dim must equal x last dim");

    CHECK_INPUT(x, torch::kBFloat16);
    CHECK_TYPE(scale_table, torch::kFloat32);
    CHECK_TYPE(shift_table, torch::kFloat32);
    CHECK_TYPE(scale_ts, torch::kBFloat16);
    CHECK_TYPE(shift_ts, torch::kBFloat16);
    CHECK_TYPE(sf_scale, torch::kFloat32);
    TORCH_CHECK(scale_table.is_contiguous(), "scale_table must be contiguous");
    TORCH_CHECK(shift_table.is_contiguous(), "shift_table must be contiguous");
    TORCH_CHECK(scale_ts.stride(-1) == 1, "scale_ts must have inner-dim stride 1");
    TORCH_CHECK(shift_ts.stride(-1) == 1, "shift_ts must have inner-dim stride 1");
    TORCH_CHECK(sf_scale.numel() >= 1, "sf_scale must contain at least 1 element");

    int64_t const hidden_dim = x.size(-1);
    int64_t const num_tokens = x.numel() / hidden_dim;
    int64_t const n_rows = scale_ts.numel() / hidden_dim;

    TORCH_CHECK(shift_ts.numel() / hidden_dim == n_rows, "shift_ts row count mismatch");
    TORCH_CHECK(num_tokens % n_rows == 0, "num_tokens (", num_tokens, ") must be divisible by n_rows (", n_rows, ")");
    TORCH_CHECK(hidden_dim % 16 == 0, "hidden_dim must be divisible by 16 (NVFP4 group size)");

    int64_t const tokens_per_batch = num_tokens / n_rows;
    auto ts_stride_of = [&](torch::Tensor const& t) -> int64_t { return (n_rows > 1) ? t.stride(-2) : hidden_dim; };

    auto opt_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(x.device());
    int64_t const sf_cols = hidden_dim / 16;
    int64_t const padded_rows = (num_tokens + 127) / 128 * 128;
    int64_t const padded_cols = (sf_cols + 3) / 4 * 4;
    auto out_fp4 = torch::empty({num_tokens, hidden_dim / 2}, opt_u8);
    auto out_sf = torch::empty({padded_rows * padded_cols}, opt_u8);

    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

    tensorrt_llm::kernels::AdaLNNormParams params;
    params.x = reinterpret_cast<__nv_bfloat16*>(x.data_ptr());
    params.scale_table[0] = reinterpret_cast<float const*>(scale_table.data_ptr());
    params.scale_ts[0] = reinterpret_cast<__nv_bfloat16 const*>(scale_ts.data_ptr());
    params.scale_ts_stride[0] = static_cast<int>(ts_stride_of(scale_ts));
    params.shift_table[0] = reinterpret_cast<float const*>(shift_table.data_ptr());
    params.shift_ts[0] = reinterpret_cast<__nv_bfloat16 const*>(shift_ts.data_ptr());
    params.shift_ts_stride[0] = static_cast<int>(ts_stride_of(shift_ts));
    params.out_fp4[0] = reinterpret_cast<uint32_t*>(out_fp4.data_ptr());
    params.out_sf[0] = reinterpret_cast<uint32_t*>(out_sf.data_ptr());
    params.sf_scale[0] = reinterpret_cast<float const*>(sf_scale.data_ptr());
    params.num_tokens = static_cast<int>(num_tokens);
    params.tokens_per_batch = static_cast<int>(tokens_per_batch);
    params.eps = static_cast<float>(eps);
    tensorrt_llm::kernels::launchFusedDiTNorm</*HAS_RESIDUAL=*/false, /*HAS_GATE=*/false, /*HAS_MODULATE=*/true,
        /*NUM_OUT=*/1, /*HAS_QUANT=*/true>(params, static_cast<int>(hidden_dim), stream);

    return std::make_tuple(out_fp4, out_sf);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_dit_rmsnorm_shift_scale(Tensor x, Tensor scale_table, Tensor scale_ts, "
        "Tensor shift_table, Tensor shift_ts, float eps) -> Tensor");
    m.def(
        "fused_dit_rmsnorm_shift_scale_quant("
        "Tensor x, Tensor scale_table, Tensor scale_ts, Tensor shift_table, Tensor shift_ts, "
        "Tensor sf_scale, float eps) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_dit_rmsnorm_shift_scale", &fused_dit_rmsnorm_shift_scale);
    m.impl("fused_dit_rmsnorm_shift_scale_quant", &fused_dit_rmsnorm_shift_scale_quant);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
