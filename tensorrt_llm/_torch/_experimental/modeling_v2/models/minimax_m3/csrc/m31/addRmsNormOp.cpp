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

#include "minimax_m3/csrc/m31/addRmsNormSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

bool aligned16(torch::Tensor const& t)
{
    return t.is_contiguous() && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
}

} // namespace

//! The residual add + RMSNorm of input / residual (contiguous 16-byte-aligned bf16 [M, 6144], both
//! written in place) with weight (bf16 [6144]), plus the MXFP8 form when q (E4M3 [M, 6144]) and sf (uint8, at least
//! mxfp8_rows.scale_bytes(M, 6144)) are given and n side-by-side copies when replicas (bf16 [M, n * 6144]) is given;
//! bit for bit the served CuTe DSL add_rmsnorm_mxfp8 / _replicated / _replicated_mxfp8 and FlashInfer's
//! fused_add_rmsnorm (neither). M <= 65535. NORM_L2PF: prefetch0 / prefetch1 (any device tensors this launch
//! does not write, e.g. the next GEMM's weight and weight scales) have their first prefetch*_bytes bytes pulled into L2
//! by the kernel's CTAs before its PDL wait; 0 bytes or no tensor = none. Outputs do not depend on them.
void minimaxM31AddRmsNorm(torch::Tensor& input, torch::Tensor& residual, torch::Tensor const& weight, double eps,
    std::optional<torch::Tensor> q, std::optional<torch::Tensor> sf, std::optional<torch::Tensor> replicas, bool pdl,
    std::optional<torch::Tensor> prefetch0, int64_t prefetch0_bytes, std::optional<torch::Tensor> prefetch1,
    int64_t prefetch1_bytes)
{
    constexpr int64_t kH = TRTLLM_MV2_NS::kernels::minimaxM31::kAddRmsNormHidden;
    TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.size(1) == kH && residual.sizes() == input.sizes()
            && weight.dim() == 1 && weight.size(0) == kH,
        "minimax_m31_add_rmsnorm: input / residual [M, 6144], weight [6144]");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16 && residual.scalar_type() == at::kBFloat16
            && weight.scalar_type() == at::kBFloat16,
        "minimax_m31_add_rmsnorm: bf16 input / residual / weight");
    TORCH_CHECK(aligned16(input) && aligned16(residual) && aligned16(weight),
        "minimax_m31_add_rmsnorm: contiguous 16-byte-aligned input / residual / weight");
    int64_t const rows = input.size(0);
    TORCH_CHECK(rows <= 65535, "minimax_m31_add_rmsnorm: at most 65535 rows");
    TORCH_CHECK(q.has_value() == sf.has_value(), "minimax_m31_add_rmsnorm: q and sf come together");
    TRTLLM_MV2_NS::kernels::minimaxM31::AddRmsNormParams p{};
    if (q.has_value())
    {
        int64_t const sfBytes = (rows + 127) / 128 * 128 * ((kH / 32 + 3) / 4 * 4);
        TORCH_CHECK(q->scalar_type() == at::kFloat8_e4m3fn && q->dim() == 2 && q->size(0) == rows && q->size(1) == kH
                && aligned16(*q),
            "minimax_m31_add_rmsnorm: q must be a contiguous 16-byte-aligned E4M3 [M, 6144]");
        TORCH_CHECK(sf->scalar_type() == at::kByte && sf->is_contiguous() && sf->numel() >= sfBytes
                && reinterpret_cast<uintptr_t>(sf->data_ptr()) % 4 == 0,
            "minimax_m31_add_rmsnorm: sf must be contiguous 4-byte-aligned uint8 of at least scale_bytes(M, 6144)");
        p.q = reinterpret_cast<uint64_t*>(q->data_ptr());
        p.sfWords = reinterpret_cast<uint32_t*>(sf->data_ptr());
    }
    if (replicas.has_value())
    {
        TORCH_CHECK(replicas->scalar_type() == at::kBFloat16 && replicas->dim() == 2 && replicas->size(0) == rows
                && replicas->size(1) >= kH && replicas->size(1) % kH == 0 && aligned16(*replicas),
            "minimax_m31_add_rmsnorm: replicas must be a contiguous 16-byte-aligned bf16 [M, n * 6144], n >= 1");
        p.replicas = reinterpret_cast<__nv_bfloat16*>(replicas->data_ptr());
        p.numReplicas = static_cast<int>(replicas->size(1) / kH);
    }
    TORCH_CHECK(residual.device() == input.device() && weight.device() == input.device()
            && (!q.has_value() || (q->device() == input.device() && sf->device() == input.device()))
            && (!replicas.has_value() || replicas->device() == input.device()),
        "minimax_m31_add_rmsnorm: every operand on input's device");
    auto const prefetchRange
        = [&](std::optional<torch::Tensor> const& t, int64_t bytes, char const*& base, uint64_t& size, char const* name)
    {
        TORCH_CHECK(bytes >= 0, "minimax_m31_add_rmsnorm: ", name, "_bytes must be >= 0");
        if (bytes == 0 || !t.has_value())
        {
            return;
        }
        TORCH_CHECK(t->device() == input.device() && t->is_contiguous()
                && bytes <= static_cast<int64_t>(t->numel() * t->element_size()),
            "minimax_m31_add_rmsnorm: ", name, " must be a contiguous tensor on input's device of at least ", name,
            "_bytes bytes");
        base = static_cast<char const*>(t->data_ptr());
        size = static_cast<uint64_t>(bytes);
    };
    prefetchRange(prefetch0, prefetch0_bytes, p.prefetch0, p.prefetch0Bytes, "prefetch0");
    prefetchRange(prefetch1, prefetch1_bytes, p.prefetch1, p.prefetch1Bytes, "prefetch1");
    p.input = reinterpret_cast<__nv_bfloat16*>(input.data_ptr());
    p.residual = reinterpret_cast<__nv_bfloat16*>(residual.data_ptr());
    p.weight = reinterpret_cast<__nv_bfloat16 const*>(weight.data_ptr());
    p.numRows = static_cast<int>(rows);
    p.eps = static_cast<float>(eps);
    at::cuda::CUDAGuard const guard(input.device());
    cudaError_t const e = TRTLLM_MV2_NS::kernels::minimaxM31::invokeAddRmsNorm(
        p, pdl, at::cuda::getCurrentCUDAStream(input.get_device()));
    TORCH_CHECK(e == cudaSuccess, "minimax_m31_add_rmsnorm launch: ", cudaGetErrorString(e));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m31_add_rmsnorm(Tensor(a!) input, Tensor(b!) residual, Tensor weight, float eps, Tensor(c!)? q=None, "
        "Tensor(d!)? sf=None, Tensor(e!)? replicas=None, bool pdl=True, Tensor? prefetch0=None, int prefetch0_bytes=0, "
        "Tensor? prefetch1=None, int prefetch1_bytes=0) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_add_rmsnorm", &TRTLLM_MV2_NS::torch_ext::minimaxM31AddRmsNorm);
}
