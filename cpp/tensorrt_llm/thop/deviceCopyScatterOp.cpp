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

#include "tensorrt_llm/kernels/deviceCopyScatter.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! Issue a batch of packed host-to-device and device-to-device copies in one launch.
//!
//! @param stage Contiguous uint8 CUDA tensor laid out as described in kernels/deviceCopyScatter.h. Its
//! destinations are raw device addresses: the caller guarantees each one is live and that no other work on the stream
//! writes it between staging and this call.
void deviceCopyScatter(torch::Tensor const& stage)
{
    TORCH_CHECK(stage.is_cuda(), "device_copy_scatter expects a CUDA staging buffer");
    TORCH_CHECK(stage.scalar_type() == torch::kUInt8, "device_copy_scatter expects a uint8 staging buffer");
    TORCH_CHECK(stage.is_contiguous(), "device_copy_scatter expects a contiguous staging buffer");
    TORCH_CHECK(stage.numel() >= 16, "device_copy_scatter staging buffer is shorter than its header");
    c10::cuda::CUDAGuard const deviceGuard{stage.device()};
    tensorrt_llm::kernels::invokeDeviceCopyScatter(
        stage.data_ptr<uint8_t>(), at::cuda::getCurrentCUDAStream(stage.get_device()));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("device_copy_scatter(Tensor stage) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("device_copy_scatter", &tensorrt_llm::torch_ext::deviceCopyScatter);
}
