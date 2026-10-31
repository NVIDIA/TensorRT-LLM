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

#include "tensorrt_llm/kernels/minimaxM3Sm107Tp8/stepPrologue.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! The decode graph's step-input prologue in one launch (see kernels/minimaxM3Sm107Tp8/stepPrologue.h).
//!
//! @param window The prologue's device window: contiguous uint8, at least the header and `maxDesc` descriptors (the
//! pack's descriptor area; the header's two counts add up to at most that). Its destinations are raw device addresses
//! the caller guarantees are live.
//! @param epoch Optional one-element int64 step counter, incremented once.
void minimaxM3Sm107Tp8StepPrologue(torch::Tensor& window, std::optional<torch::Tensor> epoch, int64_t maxDesc)
{
    TORCH_CHECK(window.is_cuda() && window.scalar_type() == torch::kUInt8 && window.is_contiguous(),
        "minimax_m3_sm107_tp8_step_prologue: window must be a contiguous uint8 CUDA tensor");
    TORCH_CHECK(maxDesc > 0 && window.numel() >= 16 + 24 * maxDesc,
        "minimax_m3_sm107_tp8_step_prologue: window is shorter than its header and max_desc descriptors");
    c10::cuda::CUDAGuard const deviceGuard{window.device()};
    int64_t* epochPtr = nullptr;
    if (epoch.has_value())
    {
        TORCH_CHECK(epoch->is_cuda() && epoch->scalar_type() == torch::kInt64 && epoch->numel() == 1
                && epoch->device() == window.device(),
            "minimax_m3_sm107_tp8_step_prologue: epoch must be a one-element int64 tensor on the window's device");
        epochPtr = epoch->data_ptr<int64_t>();
    }
    auto const err = tensorrt_llm::kernels::minimaxM3Sm107Tp8::invokeStepPrologue(
        window.data_ptr<uint8_t>(), epochPtr, maxDesc, at::cuda::getCurrentCUDAStream(window.get_device()));
    TORCH_CHECK(err == cudaSuccess, "minimax_m3_sm107_tp8_step_prologue launch failed: ", cudaGetErrorString(err));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("minimax_m3_sm107_tp8_step_prologue(Tensor(a!) window, Tensor(b!)? epoch, int max_desc) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_sm107_tp8_step_prologue", &tensorrt_llm::torch_ext::minimaxM3Sm107Tp8StepPrologue);
}
