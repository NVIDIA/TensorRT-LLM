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

// NVFP4 quantization into caller-owned buffers: the same kernel as fp4_quantize, writing where a consumer reads
// (an MoE backend's activation staging), so the rows are not quantized into a fresh tensor and copied afterwards.

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/quantization.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/library.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

constexpr int64_t kSfVecSize = 16;

//! Launches the fp4_quantize kernel for inputs of type T, with UE4M3 scale factors in the given layout.
template <typename T>
void launchFp4QuantizeInto(at::Tensor const& input, at::Tensor const& globalScale, at::Tensor& out, at::Tensor& outSf,
    QuantizationSFLayout layout, int sms, cudaStream_t stream)
{
    kernels::invokeFP4Quantization<T, kSfVecSize>(1, static_cast<int>(input.size(0)), static_cast<int>(input.size(1)),
        reinterpret_cast<T const*>(input.data_ptr()), globalScale.data_ptr<float>(),
        reinterpret_cast<int64_t*>(out.data_ptr()), reinterpret_cast<int32_t*>(outSf.data_ptr()), false, layout, sms,
        stream);
}

} // namespace

// input: [M, K] half or bfloat16, K a multiple of 16; globalScale: [1] float32. out: [M, K / 2] uint8 (two E2M1
// per byte); out_sf: uint8 of at least the scale factors' size, M * K / 16 bytes in the linear layout (row stride
// K / 16) or the 128 x 4 swizzled layout's padded size. Both contiguous on the input's device.
void fp4QuantizeInto(
    at::Tensor const& input, at::Tensor const& globalScale, at::Tensor& out, at::Tensor& outSf, bool isSfSwizzledLayout)
{
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() == 2, "input must be a contiguous CUDA [M, K]");
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16, "input must be fp16 or bf16");
    TORCH_CHECK(globalScale.is_cuda() && globalScale.scalar_type() == at::kFloat && globalScale.numel() == 1,
        "globalScale must be one float32 on the device");
    auto const m = input.size(0);
    auto const k = input.size(1);
    TORCH_CHECK(k % kSfVecSize == 0, "K must be a multiple of 16");
    int64_t const sfBytes = isSfSwizzledLayout ? tensorrt_llm::computeSwizzledLayoutSFSize(m, k / kSfVecSize)
                                               : tensorrt_llm::computeLinearLayoutSFSize(m, k / kSfVecSize);
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kByte && out.device() == input.device()
            && out.numel() == m * (k / 2),
        "out must be contiguous uint8 [M, K / 2] on the input's device");
    TORCH_CHECK(outSf.is_cuda() && outSf.is_contiguous() && outSf.scalar_type() == at::kByte
            && outSf.device() == input.device() && outSf.numel() >= sfBytes,
        "out_sf must be contiguous uint8 of at least the scale factors' size on the input's device");
    TORCH_CHECK(
        reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0 && reinterpret_cast<uintptr_t>(outSf.data_ptr()) % 4 == 0,
        "out needs 8-byte and out_sf 4-byte alignment");
    if (m == 0)
    {
        return;
    }
    auto const layout = isSfSwizzledLayout ? QuantizationSFLayout::SWIZZLED : QuantizationSFLayout::LINEAR;
    int const sms = at::cuda::getDeviceProperties(input.get_device())->multiProcessorCount;
    auto const stream = at::cuda::getCurrentCUDAStream(input.get_device());
    auto const launch
        = input.scalar_type() == at::kHalf ? launchFp4QuantizeInto<half> : launchFp4QuantizeInto<__nv_bfloat16>;
    launch(input, globalScale, out, outSf, layout, sms, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fp4_quantize_into(Tensor input, Tensor globalScale, Tensor(a!) out, Tensor(b!) out_sf, "
        "bool isSfSwizzledLayout=False) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp4_quantize_into", &tensorrt_llm::torch_ext::fp4QuantizeInto);
}
