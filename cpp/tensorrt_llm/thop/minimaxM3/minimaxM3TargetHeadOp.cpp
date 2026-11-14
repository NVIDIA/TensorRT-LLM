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
#include "tensorrt_llm/kernels/minimaxM3/targetHead.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <initializer_list>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace th = tensorrt_llm::kernels::minimaxM3::targetHead;

void checkArch(torch::Tensor const& t)
{
    TORCH_CHECK(t.is_cuda(), "minimax_m3_target_head expects CUDA tensors");
    static int const sSmVersion = tensorrt_llm::common::getSMVersion();
    TORCH_CHECK(sSmVersion >= 100, "minimax_m3_target_head requires SM100 or newer");
}

} // namespace

//! out [M, n] (BF16) = x [M <= 8, K] (BF16) x W^T for the 12-bit coded BF16 shard W [n, K]: w the coded stream
//! (uint8 [tiles * K / 64 * 1536]), exc_off int32 [tiles + 1] / exc_pos int32 [E] / exc_w fp32 [E, 2] its exception
//! lists, exp_base its exponent window's base. part (fp32 [grid * 16 * 256]) and counter (int32 [tiles], zero) are
//! the split-tile scratch.
void minimaxM3TargetHead(torch::Tensor const& w, torch::Tensor const& x, torch::Tensor& out, torch::Tensor& part,
    torch::Tensor& counter, torch::Tensor const& excOff, torch::Tensor const& excPos, torch::Tensor const& excW,
    int64_t expBase, int64_t n, int64_t grid, bool pdl)
{
    checkArch(x);
    for (torch::Tensor const* t :
        std::initializer_list<torch::Tensor const*>{&w, &out, &part, &counter, &excOff, &excPos, &excW})
    {
        TORCH_CHECK(t->device() == x.device(), "target_head: every tensor must be on x's device");
    }
    c10::cuda::CUDAGuard const deviceGuard{x.device()};
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2, "target_head: x must be BF16 [M, K]");
    int const M = static_cast<int>(x.size(0));
    int const K = static_cast<int>(x.size(1));
    TORCH_CHECK(M >= 1 && M <= th::kMaxRows, "target_head: 1..", th::kMaxRows, " rows");
    TORCH_CHECK(K % 64 == 0 && K / 16 <= 0xFFFF, "target_head: K must be a multiple of 64");
    TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
        "target_head: x rows must be unit-stride and 16-byte aligned");
    TORCH_CHECK(n >= 16 && n % 16 == 0, "target_head: n must be a positive multiple of 16 (whole tiles)");
    int64_t const tiles = (n + 15) / 16;
    TORCH_CHECK(w.scalar_type() == at::kByte && w.is_contiguous() && w.numel() == tiles * 16 * K * 3 / 2,
        "target_head: w must be the contiguous uint8 coded stream of [ceil(n / 16) * 16, K]");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.size(0) >= M && out.size(1) >= n
            && out.stride(1) == 1,
        "target_head: out must be BF16 [M, >= n] with unit column stride");
    TORCH_CHECK(grid >= 1, "target_head: grid must be positive");
    TORCH_CHECK(part.scalar_type() == at::kFloat && part.is_contiguous() && part.numel() >= grid * th::kWarps * 2 * 128,
        "target_head: part must be fp32 [grid * 16 * 256]");
    TORCH_CHECK(counter.scalar_type() == at::kInt && counter.is_contiguous() && counter.numel() >= tiles,
        "target_head: counter must be int32 [tiles]");
    TORCH_CHECK(excOff.scalar_type() == at::kInt && excOff.is_contiguous() && excOff.numel() == tiles + 1,
        "target_head: exc_off must be int32 [tiles + 1]");
    TORCH_CHECK(excPos.scalar_type() == at::kInt && excPos.is_contiguous(), "target_head: exc_pos must be int32");
    TORCH_CHECK(excW.scalar_type() == at::kFloat && excW.is_contiguous() && excW.numel() == 2 * excPos.numel(),
        "target_head: exc_w must be fp32 [E, 2]");
    TORCH_CHECK(expBase >= 0 && expBase <= 240, "target_head: exponent base out of range");
    th::Params p{};
    p.w = reinterpret_cast<uint4 const*>(w.data_ptr());
    p.x = reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr());
    p.x_stride = x.stride(0);
    p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.out_stride = out.stride(0);
    p.part = part.data_ptr<float>();
    p.counter = counter.data_ptr<int>();
    p.exc_off = excOff.data_ptr<int>();
    p.exc_pos = reinterpret_cast<uint32_t const*>(excPos.data_ptr<int>());
    p.exc_w = reinterpret_cast<float2 const*>(excW.data_ptr<float>());
    p.exp_base = static_cast<uint32_t>((expBase << 7) | (expBase << 23));
    p.M = M;
    p.N = static_cast<int>(n);
    cudaError_t const err
        = th::invokeTargetHead(p, K, static_cast<int>(grid), pdl, at::cuda::getCurrentCUDAStream(x.get_device()));
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TORCH_CHECK(false, cudaGetErrorString(err));
    }
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m3_target_head(Tensor w, Tensor x, Tensor(a!) out, Tensor(b!) part, Tensor(c!) counter, "
        "Tensor exc_off, Tensor exc_pos, Tensor exc_w, int exp_base, int n, int grid, bool pdl) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_target_head", &tensorrt_llm::torch_ext::minimaxM3TargetHead);
}
