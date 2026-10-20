/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/fusedDiTQKNormRopeFp8Kernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstdint>
#include <tuple>
#include <utility>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

// The kernels move 16-byte vectors; check before any launch so a bad view fails cleanly.
void checkAligned(torch::Tensor const& t, char const* name)
{
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, " must be 16-byte aligned");
}

void checkQkv(torch::Tensor const& qkv, int64_t num_heads, int64_t head_dim)
{
    CHECK_INPUT(qkv, torch::kBFloat16);
    TORCH_CHECK(qkv.dim() == 2 && qkv.size(1) == 3 * num_heads * head_dim,
        "qkv must be [num_tokens, 3 * num_heads * head_dim]; got ", qkv.sizes());
    checkAligned(qkv, "qkv");
}

torch::Tensor vAmax(torch::Tensor const& qkv, int64_t num_heads, int64_t head_dim, cudaStream_t stream)
{
    int const device = qkv.get_device();
    int const partials = tensorrt_llm::kernels::getDiTVAmaxWorkspaceSize(device);
    auto const f32 = qkv.options().dtype(torch::kFloat32);
    auto workspace = torch::empty({partials}, f32);
    auto amax = torch::empty({1}, f32);
    tensorrt_llm::kernels::launchDiTVAmax(qkv.data_ptr(), static_cast<int>(qkv.size(0)), static_cast<int>(num_heads),
        static_cast<int>(head_dim), workspace.data_ptr<float>(), partials, amax.data_ptr<float>(), nullptr, nullptr,
        stream);
    return amax;
}

} // namespace

// |V| maximum of a packed bf16 QKV tensor, fp32 [1]. Sequence-parallel callers reduce it across
// ranks and pass it to fused_dit_qk_norm_rope_fp8 so every rank quantizes V with one scale.
torch::Tensor dit_v_amax(torch::Tensor const& qkv, int64_t num_heads, int64_t head_dim)
{
    checkQkv(qkv, num_heads, head_dim);
    return vAmax(qkv, num_heads, head_dim, at::cuda::getCurrentCUDAStream(qkv.get_device()));
}

// Full-dim QK RMSNorm + RoPE + FP8 quantization of a packed bf16 QKV tensor (Wan).
// Q and K are quantized as round_bf16(x) * qk_mul, so callers fold their descale (and optionally
// the softmax scale) into qk_mul. V uses a per-tensor scale from v_amax, or from the local |V|
// maximum when v_amax is not given. Returns e4m3 Q, K, V [num_tokens, num_heads, head_dim] and the
// fp32 [1] V descale.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> fused_dit_qk_norm_rope_fp8(
    torch::Tensor const& qkv, int64_t num_heads, int64_t head_dim, double eps, torch::Tensor const& q_weight,
    torch::Tensor const& k_weight, torch::Tensor const& cos_emb, torch::Tensor const& sin_emb, bool interleave,
    double qk_mul, c10::optional<torch::Tensor> const& v_amax)
{
    checkQkv(qkv, num_heads, head_dim);
    int64_t const n = num_heads * head_dim;
    CHECK_INPUT(q_weight, torch::kBFloat16);
    CHECK_INPUT(k_weight, torch::kBFloat16);
    TORCH_CHECK(q_weight.dim() == 1 && q_weight.size(0) == n && k_weight.sizes() == q_weight.sizes(),
        "q_weight and k_weight must be [num_heads * head_dim] (full-dim norm)");
    TORCH_CHECK(tensorrt_llm::kernels::fusedDiTQKNormRopeFp8Supported(
                    static_cast<int>(num_heads), static_cast<int>(head_dim), tensorrt_llm::common::getSMVersion()),
        "fused_dit_qk_norm_rope_fp8 requires SM90+, head_dim 128 and at most 64 heads");

    // cos/sin: [..., head_dim] shared across heads, or [..., num_heads, head_dim] per head.
    TORCH_CHECK(cos_emb.sizes() == sin_emb.sizes(), "sin_emb shape must match cos_emb");
    bool const per_head_cos = cos_emb.dim() >= 3 && cos_emb.size(-2) == num_heads && cos_emb.size(-1) == head_dim;
    TORCH_CHECK(per_head_cos || cos_emb.size(-1) == head_dim, "cos_emb last dim must be head_dim");
    auto const cos_2d = cos_emb.reshape({-1, per_head_cos ? n : head_dim}).to(torch::kFloat32).contiguous();
    auto const sin_2d = sin_emb.reshape({-1, per_head_cos ? n : head_dim}).to(torch::kFloat32).contiguous();
    int64_t const num_tokens = qkv.size(0);
    for (auto const& [t, name] : {std::pair{&q_weight, "q_weight"}, std::pair{&k_weight, "k_weight"},
             std::pair{&cos_2d, "cos_emb"}, std::pair{&sin_2d, "sin_emb"}})
    {
        checkAligned(*t, name);
    }
    int64_t const cos_rows = cos_2d.size(0);
    TORCH_CHECK(cos_rows > 0 && num_tokens % cos_rows == 0, "cos_emb rows (", cos_rows, ") must equal num_tokens (",
        num_tokens, ") or evenly divide it (broadcast over the batch)");
    int const cos_seq_per_batch = cos_rows == num_tokens ? 0 : static_cast<int>(cos_rows);

    auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());
    auto const f32 = qkv.options().dtype(torch::kFloat32);
    torch::Tensor amax;
    if (v_amax.has_value())
    {
        CHECK_INPUT(v_amax.value(), torch::kFloat32);
        TORCH_CHECK(v_amax.value().numel() == 1, "v_amax must have one element");
        amax = v_amax.value();
    }
    else
    {
        amax = vAmax(qkv, num_heads, head_dim, stream);
    }
    // [q_mul, k_mul, v_mul]: the kernel reads the multipliers from device memory.
    auto mul = torch::empty({3}, f32);
    mul.slice(0, 0, 2).fill_(qk_mul);
    auto v_descale = torch::empty({1}, f32);
    float* const mulPtr = mul.data_ptr<float>();
    tensorrt_llm::kernels::launchDiTVScaleFromAmax(
        amax.data_ptr<float>(), mulPtr + 2, v_descale.data_ptr<float>(), stream);

    auto const fp8 = qkv.options().dtype(torch::kFloat8_e4m3fn);
    auto q8 = torch::empty({num_tokens, num_heads, head_dim}, fp8);
    auto k8 = torch::empty_like(q8);
    auto v8 = torch::empty_like(q8);
    tensorrt_llm::kernels::launchFusedDiTQKNormRopeFullDimFp8(qkv.data_ptr(), q8.data_ptr(), k8.data_ptr(),
        v8.data_ptr(), static_cast<int>(num_tokens), static_cast<int>(num_heads), static_cast<int>(head_dim),
        static_cast<float>(eps), q_weight.data_ptr(), k_weight.data_ptr(), cos_2d.data_ptr<float>(),
        sin_2d.data_ptr<float>(), interleave, per_head_cos, cos_seq_per_batch, mulPtr, stream);
    return {q8, k8, v8, v_descale};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("dit_v_amax(Tensor qkv, int num_heads, int head_dim) -> Tensor");
    m.def(
        "fused_dit_qk_norm_rope_fp8(Tensor qkv, int num_heads, int head_dim, float eps, Tensor q_weight, "
        "Tensor k_weight, Tensor cos_emb, Tensor sin_emb, bool interleave, float qk_mul, Tensor? v_amax) "
        "-> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("dit_v_amax", &tensorrt_llm::torch_ext::dit_v_amax);
    m.impl("fused_dit_qk_norm_rope_fp8", &tensorrt_llm::torch_ext::fused_dit_qk_norm_rope_fp8);
}
