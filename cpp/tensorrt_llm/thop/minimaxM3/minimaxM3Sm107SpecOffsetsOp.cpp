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

#include "tensorrt_llm/kernels/minimaxM3Sm107/specOffsets.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

void check(torch::Tensor const& t, char const* name, char const* opName)
{
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kInt32 && t.dim() == 1 && t.stride(0) == 1, opName, ": ", name,
        " must be a contiguous 1-D int32 CUDA tensor");
}

} // namespace

//! pos += pos_off and kv += kv_off (equal lengths per pair) in one plain launch
//! (kernels/minimaxM3Sm107/specOffsets.h).
template <int kTp>
void minimaxM3Sm107SpecOffsets(
    torch::Tensor& pos, torch::Tensor const& posOff, torch::Tensor& kv, torch::Tensor const& kvOff)
{
    constexpr char const* kOpName = kTp == 2 ? "minimax_m3_sm107_tp2_spec_offsets"
        : kTp == 4                           ? "minimax_m3_sm107_tp4_spec_offsets"
                                             : "minimax_m3_sm107_tp8_spec_offsets";
    check(pos, "pos", kOpName);
    check(posOff, "pos_off", kOpName);
    check(kv, "kv", kOpName);
    check(kvOff, "kv_off", kOpName);
    TORCH_CHECK(pos.numel() == posOff.numel() && kv.numel() == kvOff.numel(), kOpName,
        ": each tensor and its offsets must have the same length");
    if (pos.numel() == 0 && kv.numel() == 0)
    {
        return;
    }
    c10::cuda::CUDAGuard const deviceGuard{pos.device()};
    auto const err = tensorrt_llm::kernels::minimaxM3Sm107::invokeSpecOffsets(pos.data_ptr<int32_t>(),
        posOff.data_ptr<int32_t>(), pos.numel(), kv.data_ptr<int32_t>(), kvOff.data_ptr<int32_t>(), kv.numel(),
        at::cuda::getCurrentCUDAStream(pos.get_device()));
    TORCH_CHECK(err == cudaSuccess, kOpName, " launch failed: ", cudaGetErrorString(err));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("minimax_m3_sm107_tp2_spec_offsets(Tensor(a!) pos, Tensor pos_off, Tensor(b!) kv, Tensor kv_off) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_sm107_tp2_spec_offsets", &tensorrt_llm::torch_ext::minimaxM3Sm107SpecOffsets<2>);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("minimax_m3_sm107_tp4_spec_offsets(Tensor(a!) pos, Tensor pos_off, Tensor(b!) kv, Tensor kv_off) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_sm107_tp4_spec_offsets", &tensorrt_llm::torch_ext::minimaxM3Sm107SpecOffsets<4>);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("minimax_m3_sm107_tp8_spec_offsets(Tensor(a!) pos, Tensor pos_off, Tensor(b!) kv, Tensor kv_off) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_sm107_tp8_spec_offsets", &tensorrt_llm::torch_ext::minimaxM3Sm107SpecOffsets<8>);
}
