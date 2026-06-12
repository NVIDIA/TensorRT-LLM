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

// Fused residual add + gate multiply + weightless RMSNorm + optional NVFP4 quant
//
// The gate modulator is built inline by the C++ op:
//   gate[b,d] = gate_table[d].to(bf16) + gate_ts[b,d]    (bf16 narrow first, bf16 hw add)
// This folds the upstream broadcast-add Triton prep kernel (which Inductor would
// fuse with the `attn_out * gate` mul) into Phase 0b. Matches PyTorch eager
// `_get_ada_values` semantics byte-for-byte.
//
// In-place:
//   x <- x + attn_out * gate
// Computes:
//   normed[n, d] = rsqrt(mean(x_new[n, :]^2) + eps) * x_new[n, d]
//   bf16 variant:  out_bf16 = bf16(normed)
//   quant variant: (fp4, sf) = nvfp4_quantize(normed, sf_scale)
//
// Inputs:
//   x:           bf16, [..., D] contig, MUTATED in-place to x + attn_out * gate
//   attn_out:    bf16, same shape as x
//   gate_table:  fp32, [D]                    (broadcast over batch)
//   gate_ts:     bf16, [..., D], inner stride 1; row stride (stride(-2)) may
//                match a [B, T_t, K, D] unbind view without .contiguous()
//   sf_scale:    fp32 scalar (quant variant; calibrated downstream input_scale)
//   eps:         RMSNorm epsilon
//
// Returns bf16 variant:  out_bf16: bf16 [..., D]
// Returns quant variant: (out_fp4 [num_tokens, D/2] uint8, out_sf [SWIZZLED] uint8)
torch::Tensor fused_dit_gate_resid_rmsnorm(torch::Tensor x, torch::Tensor const& attn_out,
    torch::Tensor const& gate_table, torch::Tensor const& gate_ts, double eps)
{
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims; got ", x.dim());
    TORCH_CHECK(attn_out.dim() >= 2, "attn_out must have at least 2 dims; got ", attn_out.dim());
    TORCH_CHECK(x.sizes() == attn_out.sizes(), "x and attn_out shape mismatch");
    TORCH_CHECK(gate_table.dim() == 1, "gate_table must be 1-D [D]; got dim=", gate_table.dim());
    TORCH_CHECK(gate_table.size(0) == x.size(-1), "gate_table size must equal x last dim");
    TORCH_CHECK(gate_ts.size(-1) == x.size(-1), "gate_ts last dim must equal x last dim");

    CHECK_INPUT(x, torch::kBFloat16);
    CHECK_INPUT(attn_out, torch::kBFloat16);
    CHECK_TYPE(gate_table, torch::kFloat32);
    CHECK_TYPE(gate_ts, torch::kBFloat16);
    TORCH_CHECK(gate_table.is_contiguous(), "gate_table must be contiguous");
    TORCH_CHECK(gate_ts.stride(-1) == 1, "gate_ts must have inner-dim stride 1");

    int64_t const hidden_dim = x.size(-1);
    int64_t const num_tokens = x.numel() / hidden_dim;
    int64_t const n_rows = gate_ts.numel() / hidden_dim;

    TORCH_CHECK(num_tokens % n_rows == 0, "num_tokens (", num_tokens, ") must be divisible by n_rows (", n_rows, ")");

    int64_t const tokens_per_batch = num_tokens / n_rows;
    auto ts_stride_of = [&](torch::Tensor const& t) -> int64_t { return (n_rows > 1) ? t.stride(-2) : hidden_dim; };

    auto out_bf16 = torch::empty_like(x);
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

    tensorrt_llm::kernels::AdaLNNormParams params;
    params.x = reinterpret_cast<__nv_bfloat16*>(x.data_ptr());
    params.attn = reinterpret_cast<__nv_bfloat16 const*>(attn_out.data_ptr());
    params.gate_table = reinterpret_cast<float const*>(gate_table.data_ptr());
    params.gate_ts = reinterpret_cast<__nv_bfloat16 const*>(gate_ts.data_ptr());
    params.gate_ts_stride = static_cast<int>(ts_stride_of(gate_ts));
    params.out_bf16[0] = reinterpret_cast<__nv_bfloat16*>(out_bf16.data_ptr());
    params.num_tokens = static_cast<int>(num_tokens);
    params.tokens_per_batch = static_cast<int>(tokens_per_batch);
    params.eps = static_cast<float>(eps);
    tensorrt_llm::kernels::launchFusedDiTNorm</*HAS_RESIDUAL=*/true, /*HAS_GATE=*/true, /*HAS_NORM=*/true,
        /*HAS_SHIFT_SCALE=*/false,
        /*NUM_OUT=*/1, /*HAS_QUANT=*/false>(params, static_cast<int>(hidden_dim), stream);

    return out_bf16;
}

std::tuple<torch::Tensor, torch::Tensor> fused_dit_gate_resid_rmsnorm_quant(torch::Tensor x,
    torch::Tensor const& attn_out, torch::Tensor const& gate_table, torch::Tensor const& gate_ts,
    torch::Tensor const& sf_scale, double eps)
{
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims; got ", x.dim());
    TORCH_CHECK(attn_out.dim() >= 2, "attn_out must have at least 2 dims; got ", attn_out.dim());
    TORCH_CHECK(x.sizes() == attn_out.sizes(), "x and attn_out shape mismatch");
    TORCH_CHECK(gate_table.dim() == 1, "gate_table must be 1-D [D]");
    TORCH_CHECK(gate_table.size(0) == x.size(-1), "gate_table size must equal x last dim");
    TORCH_CHECK(gate_ts.size(-1) == x.size(-1), "gate_ts last dim must equal x last dim");

    CHECK_INPUT(x, torch::kBFloat16);
    CHECK_INPUT(attn_out, torch::kBFloat16);
    CHECK_TYPE(gate_table, torch::kFloat32);
    CHECK_TYPE(gate_ts, torch::kBFloat16);
    CHECK_TYPE(sf_scale, torch::kFloat32);
    TORCH_CHECK(gate_table.is_contiguous(), "gate_table must be contiguous");
    TORCH_CHECK(gate_ts.stride(-1) == 1, "gate_ts must have inner-dim stride 1");
    TORCH_CHECK(sf_scale.numel() >= 1, "sf_scale must contain at least 1 element");

    int64_t const hidden_dim = x.size(-1);
    int64_t const num_tokens = x.numel() / hidden_dim;
    int64_t const n_rows = gate_ts.numel() / hidden_dim;

    TORCH_CHECK(num_tokens % n_rows == 0, "num_tokens (", num_tokens, ") must be divisible by n_rows (", n_rows, ")");
    TORCH_CHECK(hidden_dim % 16 == 0, "hidden_dim must be divisible by 16 (NVFP4 group size)");

    int64_t const tokens_per_batch = num_tokens / n_rows;
    auto ts_stride_of = [&](torch::Tensor const& t) -> int64_t { return (n_rows > 1) ? t.stride(-2) : hidden_dim; };

    // FP4 + SF (128x4 SWIZZLED layout; matches default fp4_quantize / nvfp4_gemm contract).
    auto opt_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(x.device());
    auto out_fp4 = torch::empty({num_tokens, hidden_dim / 2}, opt_u8);
    int64_t const sf_cols = hidden_dim / 16;
    int64_t const padded_rows = (num_tokens + 127) / 128 * 128;
    int64_t const padded_cols = (sf_cols + 3) / 4 * 4;
    auto out_sf = torch::empty({padded_rows * padded_cols}, opt_u8);

    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

    tensorrt_llm::kernels::AdaLNNormParams params;
    params.x = reinterpret_cast<__nv_bfloat16*>(x.data_ptr());
    params.attn = reinterpret_cast<__nv_bfloat16 const*>(attn_out.data_ptr());
    params.gate_table = reinterpret_cast<float const*>(gate_table.data_ptr());
    params.gate_ts = reinterpret_cast<__nv_bfloat16 const*>(gate_ts.data_ptr());
    params.gate_ts_stride = static_cast<int>(ts_stride_of(gate_ts));
    params.out_fp4[0] = reinterpret_cast<uint32_t*>(out_fp4.data_ptr());
    params.out_sf[0] = reinterpret_cast<uint32_t*>(out_sf.data_ptr());
    params.sf_scale[0] = reinterpret_cast<float const*>(sf_scale.data_ptr());
    params.num_tokens = static_cast<int>(num_tokens);
    params.tokens_per_batch = static_cast<int>(tokens_per_batch);
    params.eps = static_cast<float>(eps);
    tensorrt_llm::kernels::launchFusedDiTNorm</*HAS_RESIDUAL=*/true, /*HAS_GATE=*/true, /*HAS_NORM=*/true,
        /*HAS_SHIFT_SCALE=*/false,
        /*NUM_OUT=*/1, /*HAS_QUANT=*/true>(params, static_cast<int>(hidden_dim), stream);

    return std::make_tuple(out_fp4, out_sf);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_dit_gate_resid_rmsnorm("
        "Tensor(a!) x, Tensor attn_out, Tensor gate_table, Tensor gate_ts, float eps) -> Tensor");
    m.def(
        "fused_dit_gate_resid_rmsnorm_quant("
        "Tensor(a!) x, Tensor attn_out, Tensor gate_table, Tensor gate_ts, Tensor sf_scale, float eps) "
        "-> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_dit_gate_resid_rmsnorm", &fused_dit_gate_resid_rmsnorm);
    m.impl("fused_dit_gate_resid_rmsnorm_quant", &fused_dit_gate_resid_rmsnorm_quant);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
