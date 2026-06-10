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

// Fused residual add + gate multiply (no RMSNorm, no shift_scale).
//
// Computes in-place:
//   gate[b, d] = gate_table[d].to(bf16) + gate_ts[b, d]    (bf16 narrow first, bf16 hw add)
//   x[n, d]   <- x[n, d] + attn_out[n, d] * gate[b, d]
//
// Used by LTX-2 at the FFN-output gate site:
//   vx = vx + ff(vx_scaled) * vgate_mlp
// where vgate_mlp is `_get_ada_values(table, ts, slice(5, 6))` -- pass the (table, ts)
// pair directly so the broadcast-add prep doesn't need a separate elementwise kernel.
//
// Inputs:
//   x:           bf16, [..., D] contig, MUTATED in-place to x + attn_out * gate
//   attn_out:    bf16, same shape as x
//   gate_table:  fp32, [D]                    (broadcast over batch)
//   gate_ts:     bf16, [..., D], inner stride 1
//
// Returns x (the mutated tensor) for syntactic convenience.
torch::Tensor fused_dit_gate_resid(
    torch::Tensor x, torch::Tensor const& attn_out, torch::Tensor const& gate_table, torch::Tensor const& gate_ts)
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

    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

    tensorrt_llm::kernels::AdaLNNormParams params;
    params.x = reinterpret_cast<__nv_bfloat16*>(x.data_ptr());
    params.attn = reinterpret_cast<__nv_bfloat16 const*>(attn_out.data_ptr());
    params.gate_table = reinterpret_cast<float const*>(gate_table.data_ptr());
    params.gate_ts = reinterpret_cast<__nv_bfloat16 const*>(gate_ts.data_ptr());
    params.gate_ts_stride = static_cast<int>(ts_stride_of(gate_ts));
    params.num_tokens = static_cast<int>(num_tokens);
    params.tokens_per_batch = static_cast<int>(tokens_per_batch);
    // eps unused (no RMSNorm)
    tensorrt_llm::kernels::launchFusedDiTNorm</*HAS_RESIDUAL=*/true, /*HAS_GATE=*/true,
        /*HAS_NORM=*/false, /*HAS_SHIFT_SCALE=*/false,
        /*NUM_OUT=*/1, /*HAS_QUANT=*/false>(params, static_cast<int>(hidden_dim), stream);

    return x;
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_dit_gate_resid("
        "Tensor(a!) x, Tensor attn_out, Tensor gate_table, Tensor gate_ts) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_dit_gate_resid", &fused_dit_gate_resid);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
