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

#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/csrc/samplingFromProbsMcFiSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

//! SFP_MC_FI (see minimax_m3_1_flash_nvfp4__sm_107__dep/csrc/samplingFromProbsMcFiSm107.h):
//! flashinfer.sampling.sampling_from_probs's int32 tokens [rows] (deterministic, tensor seed / offset), bitwise, on
//! rows x ctas_per_row CTAs. probs: float32 [rows, vocab] on the device, 16-byte aligned, with a row stride divisible
//! by 4 floats. seed / offset: int64 CUDA tensors with at least one entry (FlashInfer reads element 0 for every row).
//! workspace: a zero-initialized byte buffer of at least rows x minimax_m31_sampling_from_probs_mc_fi_ws_bytes(),
//! reused across calls; ctas_per_row divides 1024 (no co-residency limit).
torch::Tensor minimaxM31SamplingFromProbsMcFi(torch::Tensor const& probs, torch::Tensor const& seed,
    torch::Tensor const& offset, torch::Tensor& workspace, int64_t ctasPerRow)
{
    char const* name = "minimax_m31_sampling_from_probs_mc_fi";
    TORCH_CHECK(probs.dim() == 2 && probs.is_cuda() && probs.scalar_type() == torch::kFloat32 && probs.stride(1) == 1
            && probs.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(probs.data_ptr()) % 16 == 0,
        name,
        ": probs must be a float32 CUDA [rows, vocab] tensor, 16-byte aligned, with unit inner stride and a row "
        "stride divisible by 4");
    auto const rows = probs.size(0);
    auto const vocabSize = probs.size(1);
    TORCH_CHECK(vocabSize % 4 == 0 && vocabSize <= TRTLLM_MV2_NS::kernels::minimaxM31::kSamplingFromProbsMcMaxVocab,
        name, ": vocab must be divisible by 4 and at most ",
        TRTLLM_MV2_NS::kernels::minimaxM31::kSamplingFromProbsMcMaxVocab);
    for (auto const* t : {&seed, &offset})
    {
        TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->get_device() == probs.get_device()
                && t->scalar_type() == torch::kInt64 && t->numel() >= 1,
            name, ": seed and offset must be contiguous int64 CUDA tensors with at least one entry on probs' device");
    }
    TORCH_CHECK(rows <= 65535 && ctasPerRow >= 1 && ctasPerRow <= 1024 && 1024 % ctasPerRow == 0, name,
        ": rows <= 65535; ctas_per_row must divide 1024");
    auto const wsBytesPerRow
        = static_cast<int64_t>(TRTLLM_MV2_NS::kernels::minimaxM31::samplingFromProbsMcFiWorkspaceBytesPerRow());
    TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous() && workspace.get_device() == probs.get_device()
            && reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0
            && workspace.numel() * workspace.element_size() >= wsBytesPerRow * rows,
        name, ": workspace must be a contiguous, 16-byte aligned CUDA buffer of at least ", wsBytesPerRow * rows,
        " bytes on probs' device");

    c10::cuda::CUDAGuard const deviceGuard{probs.device()};
    auto tokens = torch::empty({rows}, probs.options().dtype(torch::kInt32));
    if (rows == 0)
    {
        return tokens;
    }
    auto const err = TRTLLM_MV2_NS::kernels::minimaxM31::invokeSamplingFromProbsMcFi(probs.const_data_ptr<float>(),
        probs.stride(0), seed.const_data_ptr<int64_t>(), offset.const_data_ptr<int64_t>(), tokens.data_ptr<int32_t>(),
        static_cast<int32_t>(rows), static_cast<int32_t>(vocabSize), static_cast<int>(ctasPerRow), workspace.data_ptr(),
        at::cuda::getCurrentCUDAStream(probs.get_device()));
    TORCH_CHECK(err == cudaSuccess, name, " launch failed: ", cudaGetErrorString(err));
    return tokens;
}

//! Workspace bytes per row of minimax_m31_sampling_from_probs_mc_fi.
int64_t minimaxM31SamplingFromProbsMcFiWsBytes()
{
    return static_cast<int64_t>(TRTLLM_MV2_NS::kernels::minimaxM31::samplingFromProbsMcFiWorkspaceBytesPerRow());
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m31_sampling_from_probs_mc_fi(Tensor probs, Tensor seed, Tensor offset, Tensor(a!) workspace, "
        "int ctas_per_row) -> Tensor");
    // No tensor arguments: a catch-all kernel (as moe_a2a_get_timeout), callable without a device.
    m.def("minimax_m31_sampling_from_probs_mc_fi_ws_bytes() -> int");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_sampling_from_probs_mc_fi", &TRTLLM_MV2_NS::torch_ext::minimaxM31SamplingFromProbsMcFi);
}

TRTLLM_MV2_LIBRARY_IMPL(CompositeExplicitAutograd, m)
{
    m.impl("minimax_m31_sampling_from_probs_mc_fi_ws_bytes",
        &TRTLLM_MV2_NS::torch_ext::minimaxM31SamplingFromProbsMcFiWsBytes);
}
