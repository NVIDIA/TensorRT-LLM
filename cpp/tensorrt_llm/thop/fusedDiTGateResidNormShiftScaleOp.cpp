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

#include "tensorrt_llm/kernels/fusedDiTGateResidNormShiftScaleKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <optional>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{
// Validate a (table, ts) modulator pair against x's last dim and row count;
// returns the per-row ts stride (handles both flat [B, D] and [B, T_t, K, D] unbind views).
int check_modulator(
    torch::Tensor const& table, torch::Tensor const& ts, int64_t hidden_dim, int64_t n_rows, char const* name)
{
    TORCH_CHECK(table.dim() == 1 && table.size(0) == hidden_dim, name, "_table must be 1-D [D]");
    CHECK_TYPE(table, torch::kFloat32);
    TORCH_CHECK(table.is_contiguous(), name, "_table must be contiguous");
    TORCH_CHECK(ts.size(-1) == hidden_dim, name, "_ts last dim must equal x last dim");
    CHECK_TYPE(ts, torch::kBFloat16);
    TORCH_CHECK(ts.stride(-1) == 1, name, "_ts must have inner-dim stride 1");
    TORCH_CHECK(ts.numel() / hidden_dim == n_rows, name, "_ts row count mismatch");
    return static_cast<int>((n_rows > 1) ? ts.stride(-2) : hidden_dim);
}

// The AdaLN variants the fused kernel actually supports. The op infers the flags from which
// optional args the caller provides; any combination NOT in this table is rejected up front
// (clear error) before dispatch. num_out is the user-facing output count: 0 = gate-residual only
// (no norm output; x_new is the result), 1 = single output, 2 = dual. quant_ok=false => only a
// bf16 instantiation exists (no NVFP4-quant kernel for that variant).
struct FusedDiTVariant
{
    bool residual;
    bool gate;
    bool norm;
    bool shift_scale;
    int64_t num_out;
    bool quant_ok;
    char const* name;
};

constexpr FusedDiTVariant kSupportedVariants[] = {
    {false, false, true, true, 1, true, "rmsnorm_shift_scale"},
    {true, false, true, true, 2, true, "resid_rmsnorm_shift_scale_dual"},
    {true, true, true, true, 1, true, "gate_resid_rmsnorm_shift_scale"},
    {true, true, true, false, 1, true, "gate_resid_rmsnorm"},
    {true, true, false, false, 0, false, "gate_resid"},
};
} // namespace

// Unified fused DiT pre-block op. Covers every AdaLN variant of the single underlying
// kernel (launchFusedDiTGateResidNormShiftScaleKernel) -- residual add, gate multiply, weightless RMSNorm,
// (single or dual) shift_scale affine modulation, and inline NVFP4 quant -- with all
// flags inferred from which optional arguments are provided. Replaces the former five
// ops (nine registrations) that each dispatched to this same kernel.
//
// Inferred flags:
//   has_residual    = attn.has_value()
//   has_gate        = gate_table.has_value()   (requires gate_ts; implies residual)
//   has_shift_scale = !scale_table.empty()     (scale/shift lists of length num_out)
//   has_quant       = !sf_scale.empty()        (sf_scale list of length num_out)
//   has_norm        = num_out >= 1             (num_out == 0 => gate-residual only)
//
// Each modulator is built inline by the kernel from its (table_fp32[D], ts_bf16[B, D]) pair:
//   m[b, d] = table[d].to(bf16) + ts[b, d]   (bf16 narrow first, bf16 hw add).
//
// Functional (no aliasing): for residual variants the kernel reads x (read-only) and writes the
// new residual stream x_new = x [+ attn [* gate]] into a SEPARATE fresh buffer (returned as
// output[0]); x is never mutated and never copied. This keeps the op alias-free (a mutable
// Tensor(a!) input + Tensor[] return is unsupported by torch.compile functionalization) with no
// clone overhead. Callers rebind x from the returned x_new.
//
// Returns (x_new prepended iff residual):
//   residual, num_out == 0  : {x_new}                          (gate-residual only)
//   residual, bf16          : {x_new, out_0, ..., out_{N-1}}
//   residual, quant         : {x_new, fp4_0, sf_0, ..., fp4_{N-1}, sf_{N-1}}  (128x4 SWIZZLED SF)
//   no-residual (KA), bf16  : {out_0}
//   no-residual (KA), quant : {fp4_0, sf_0}
std::vector<torch::Tensor> fused_dit_gate_resid_norm_shift_scale(torch::Tensor x,
    std::optional<torch::Tensor> const& attn, std::optional<torch::Tensor> const& gate_table,
    std::optional<torch::Tensor> const& gate_ts, std::optional<std::vector<torch::Tensor>> const& scale_table_opt,
    std::optional<std::vector<torch::Tensor>> const& scale_ts_opt,
    std::optional<std::vector<torch::Tensor>> const& shift_table_opt,
    std::optional<std::vector<torch::Tensor>> const& shift_ts_opt,
    std::optional<std::vector<torch::Tensor>> const& sf_scale_opt, double eps, int64_t num_out)
{
    // Optional list args default to None at the call site (callers pass only the features they
    // use); treat an absent list as empty here so the rest of the impl is unchanged.
    static std::vector<torch::Tensor> const kEmpty;
    auto const& scale_table = scale_table_opt.has_value() ? *scale_table_opt : kEmpty;
    auto const& scale_ts = scale_ts_opt.has_value() ? *scale_ts_opt : kEmpty;
    auto const& shift_table = shift_table_opt.has_value() ? *shift_table_opt : kEmpty;
    auto const& shift_ts = shift_ts_opt.has_value() ? *shift_ts_opt : kEmpty;
    auto const& sf_scale = sf_scale_opt.has_value() ? *sf_scale_opt : kEmpty;

    bool const has_residual = attn.has_value();
    bool const has_gate = gate_table.has_value();
    bool const has_shift_scale = !scale_table.empty();
    bool const has_quant = !sf_scale.empty();
    bool const has_norm = num_out >= 1;

    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims; got ", x.dim());
    CHECK_INPUT(x, torch::kBFloat16);
    int64_t const hidden_dim = x.size(-1);
    int64_t const num_tokens = x.numel() / hidden_dim;

    // Validate the inferred flag combination against the supported-variant table before dispatch.
    FusedDiTVariant const* variant = nullptr;
    for (auto const& v : kSupportedVariants)
    {
        if (v.residual == has_residual && v.gate == has_gate && v.norm == has_norm && v.shift_scale == has_shift_scale
            && v.num_out == num_out)
        {
            variant = &v;
            break;
        }
    }
    TORCH_CHECK(variant != nullptr,
        "fused_dit_gate_resid_norm_shift_scale: unsupported flag combination (residual=", has_residual,
        ", gate=", has_gate, ", norm=", has_norm, ", shift_scale=", has_shift_scale, ", num_out=", num_out,
        "). Supported: rmsnorm_shift_scale, resid_rmsnorm_shift_scale_dual, gate_resid_rmsnorm_shift_scale, "
        "gate_resid_rmsnorm, gate_resid.");
    TORCH_CHECK(!has_quant || variant->quant_ok, "fused_dit_gate_resid_norm_shift_scale: variant '", variant->name,
        "' has no NVFP4-quant instantiation");
    TORCH_CHECK(!has_gate || gate_ts.has_value(), "gate requires gate_ts");
    if (has_shift_scale)
    {
        TORCH_CHECK(static_cast<int64_t>(scale_table.size()) == num_out
                && static_cast<int64_t>(scale_ts.size()) == num_out
                && static_cast<int64_t>(shift_table.size()) == num_out
                && static_cast<int64_t>(shift_ts.size()) == num_out,
            "scale/shift table+ts lists must each have length num_out=", num_out);
    }
    if (has_quant)
    {
        TORCH_CHECK(
            static_cast<int64_t>(sf_scale.size()) == num_out, "sf_scale list must have length num_out=", num_out);
        TORCH_CHECK(hidden_dim % 16 == 0, "hidden_dim must be divisible by 16 (NVFP4 group size)");
    }

    // Row count comes from gate_ts (gated variants) or scale_ts[0] (no-gate variants);
    // every valid combination provides at least one of them.
    int64_t n_rows = 1;
    if (has_gate)
    {
        n_rows = gate_ts->numel() / hidden_dim;
    }
    else if (has_shift_scale)
    {
        n_rows = scale_ts[0].numel() / hidden_dim;
    }
    TORCH_CHECK(num_tokens % n_rows == 0, "num_tokens (", num_tokens, ") must be divisible by n_rows (", n_rows, ")");
    int64_t const tokens_per_batch = num_tokens / n_rows;

    // Functional (no Tensor(a!) alias): for residual variants the kernel reads x (read-only) and
    // writes the new residual stream x_new into a SEPARATE fresh buffer (params.x_out), returned as
    // output[0]. No clone/copy of x -- alias-free for torch.compile (a mutable input + Tensor[]
    // return is unsupported) AND strictly cheaper than an in-place op's auto_functionalize clone.
    // Callers rebind x from the returned x_new (they never relied on in-place).
    std::vector<torch::Tensor> outs;

    tensorrt_llm::kernels::AdaLNNormParams params;
    params.x = reinterpret_cast<__nv_bfloat16*>(x.data_ptr());
    if (has_residual)
    {
        torch::Tensor x_new = torch::empty_like(x);
        params.x_out = reinterpret_cast<__nv_bfloat16*>(x_new.data_ptr());
        outs.push_back(std::move(x_new));
    }
    params.num_tokens = static_cast<int>(num_tokens);
    params.tokens_per_batch = static_cast<int>(tokens_per_batch);
    params.eps = static_cast<float>(eps);

    if (has_residual)
    {
        torch::Tensor const& a = attn.value();
        TORCH_CHECK(a.dim() >= 2 && a.sizes() == x.sizes(), "x and attn shape mismatch");
        CHECK_INPUT(a, torch::kBFloat16);
        params.attn = reinterpret_cast<__nv_bfloat16 const*>(a.data_ptr());
    }
    if (has_gate)
    {
        params.gate_ts_stride = check_modulator(gate_table.value(), gate_ts.value(), hidden_dim, n_rows, "gate");
        params.gate_table = reinterpret_cast<float const*>(gate_table->data_ptr());
        params.gate_ts = reinterpret_cast<__nv_bfloat16 const*>(gate_ts->data_ptr());
    }
    if (has_shift_scale)
    {
        for (int64_t k = 0; k < num_out; ++k)
        {
            params.scale_ts_stride[k] = check_modulator(scale_table[k], scale_ts[k], hidden_dim, n_rows, "scale");
            params.shift_ts_stride[k] = check_modulator(shift_table[k], shift_ts[k], hidden_dim, n_rows, "shift");
            params.scale_table[k] = reinterpret_cast<float const*>(scale_table[k].data_ptr());
            params.scale_ts[k] = reinterpret_cast<__nv_bfloat16 const*>(scale_ts[k].data_ptr());
            params.shift_table[k] = reinterpret_cast<float const*>(shift_table[k].data_ptr());
            params.shift_ts[k] = reinterpret_cast<__nv_bfloat16 const*>(shift_ts[k].data_ptr());
        }
    }

    // Allocate norm outputs and wire them into params (NUM_OUT entries; appended after x_new).
    if (has_norm && !has_quant)
    {
        for (int64_t k = 0; k < num_out; ++k)
        {
            torch::Tensor o = torch::empty_like(x);
            params.out_bf16[k] = reinterpret_cast<__nv_bfloat16*>(o.data_ptr());
            outs.push_back(std::move(o));
        }
    }
    else if (has_norm && has_quant)
    {
        auto opt_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(x.device());
        int64_t const sf_cols = hidden_dim / 16;
        int64_t const padded_rows = (num_tokens + 127) / 128 * 128;
        int64_t const padded_cols = (sf_cols + 3) / 4 * 4;
        for (int64_t k = 0; k < num_out; ++k)
        {
            TORCH_CHECK(sf_scale[k].numel() >= 1, "sf_scale must contain at least 1 element");
            CHECK_TYPE(sf_scale[k], torch::kFloat32);
            torch::Tensor fp4 = torch::empty({num_tokens, hidden_dim / 2}, opt_u8);
            torch::Tensor sf = torch::empty({padded_rows * padded_cols}, opt_u8);
            params.out_fp4[k] = reinterpret_cast<uint32_t*>(fp4.data_ptr());
            params.out_sf[k] = reinterpret_cast<uint32_t*>(sf.data_ptr());
            params.sf_scale[k] = reinterpret_cast<float const*>(sf_scale[k].data_ptr());
            outs.push_back(std::move(fp4));
            outs.push_back(std::move(sf));
        }
    }

    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    int const hd = static_cast<int>(hidden_dim);
    // Hand the validated flags to the launcher, which maps them onto the kernel's compile-time
    // specialization (the runtime->template dispatch lives in the kernel module, not here).
    tensorrt_llm::kernels::launchFusedDiTGateResidNormShiftScaleKernel(
        params, has_residual, has_gate, has_norm, has_shift_scale, has_quant, hd, stream);

    return outs;
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_dit_gate_resid_norm_shift_scale("
        "Tensor x, Tensor? attn=None, Tensor? gate_table=None, Tensor? gate_ts=None, "
        "Tensor[]? scale_table=None, Tensor[]? scale_ts=None, Tensor[]? shift_table=None, Tensor[]? shift_ts=None, "
        "Tensor[]? sf_scale=None, float eps=1e-6, int num_out=1) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_dit_gate_resid_norm_shift_scale", &fused_dit_gate_resid_norm_shift_scale);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
