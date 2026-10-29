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

#include "tensorrt_llm/kernels/vaeWinoSpatialKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <optional>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{
tensorrt_llm::kernels::VaeWinoSpatialParams params(
    torch::Tensor const& frames, int64_t components, int64_t row0, int64_t row1, int64_t tile)
{
    TORCH_CHECK(frames.dim() == 4, "frames must be [N, H, W, C]");
    bool const half = frames.scalar_type() == torch::kHalf;
    CHECK_INPUT(frames, (half ? torch::kHalf : torch::kBFloat16));
    tensorrt_llm::kernels::VaeWinoSpatialParams p{};
    p.halfFrames = half ? 1 : 0;
    p.frames = static_cast<int>(frames.size(0));
    p.height = static_cast<int>(frames.size(1));
    p.width = static_cast<int>(frames.size(2));
    p.channels = static_cast<int>(frames.size(3));
    p.components = static_cast<int>(components);
    p.row0 = static_cast<int>(row0);
    p.row1 = static_cast<int>(row1);
    p.tile = static_cast<int>(tile);
    return p;
}
} // namespace

// V = B^T d B for tile rows [row0, row1): frames [N, H, W, C] bf16 or fp16 -> fp16
// [components*(tile+2)^2, (N/components)*ntiles, C], or with stack_frames (components == 1)
// [(tile+2)^2, ntiles, N*C]: the frames side by side along the GEMM's K axis.
torch::Tensor vae_wino_spatial_in(
    torch::Tensor const& frames, int64_t components, int64_t row0, int64_t row1, int64_t tile, bool stack_frames)
{
    auto p = params(frames, components, row0, row1, tile);
    p.stackFrames = stack_frames ? 1 : 0;
    int64_t const coords = (tile + 2) * (tile + 2);
    int64_t const ntiles = static_cast<int64_t>(p.row1 - p.row0) * ((p.width + tile - 1) / tile);
    auto v = stack_frames ? torch::empty({coords, ntiles, p.frames * p.channels}, frames.options().dtype(torch::kHalf))
                          : torch::empty({components * coords, (p.frames / components) * ntiles, p.channels},
                              frames.options().dtype(torch::kHalf));
    tensorrt_llm::kernels::launchVaeWinoSpatialInput(
        frames.data_ptr(), v.data_ptr(), p, at::cuda::getCurrentCUDAStream(frames.get_device()));
    return v;
}

// The stacked-frame input transform (components == 1) over frames held in separate buffers:
// ``zero_front`` zero frames, then the frames of each tensor in ``taps`` ([n_i, H, W, C], same
// dtype and shape), side by side along K: V [(tile+2)^2, ntiles, (zero_front + sum n_i) * C] fp16.
torch::Tensor vae_wino_spatial_in_taps(
    at::TensorList taps, int64_t zero_front, int64_t row0, int64_t row1, int64_t tile)
{
    TORCH_CHECK(!taps.empty(), "taps must hold at least one tensor");
    TORCH_CHECK(zero_front >= 0, "zero_front must be non-negative");
    auto p = params(taps[0], 1, row0, row1, tile);
    p.stackFrames = 1;
    p.segments = 0;
    if (zero_front > 0)
    {
        p.segPtr[p.segments] = nullptr;
        p.segFrames[p.segments++] = static_cast<int>(zero_front);
    }
    int frames = static_cast<int>(zero_front);
    for (auto const& tap : taps)
    {
        TORCH_CHECK(p.segments < 3, "at most three frame segments (zero frames, history, new frame)");
        TORCH_CHECK(tap.dim() == 4 && tap.size(1) == p.height && tap.size(2) == p.width && tap.size(3) == p.channels
                && tap.scalar_type() == taps[0].scalar_type(),
            "every tap must be [n, H, W, C] of the same shape and dtype");
        CHECK_INPUT(tap, taps[0].scalar_type());
        p.segPtr[p.segments] = tap.data_ptr();
        p.segFrames[p.segments++] = static_cast<int>(tap.size(0));
        frames += static_cast<int>(tap.size(0));
    }
    p.frames = frames;
    int64_t const coords = (tile + 2) * (tile + 2);
    int64_t const ntiles = static_cast<int64_t>(p.row1 - p.row0) * ((p.width + tile - 1) / tile);
    auto v = torch::empty({coords, ntiles, p.frames * p.channels}, taps[0].options().dtype(torch::kHalf));
    tensorrt_llm::kernels::launchVaeWinoSpatialInput(
        nullptr, v.data_ptr(), p, at::cuda::getCurrentCUDAStream(taps[0].get_device()));
    return v;
}

// y[:, tile*i+a, tile*j+b, :] = A^T M A (+ bias) for tile rows [row0, row1); writes into ``frames`` and
// returns it.
torch::Tensor vae_wino_spatial_out(torch::Tensor const& m, torch::Tensor const& frames, int64_t components,
    int64_t row0, int64_t row1, int64_t tile, std::optional<torch::Tensor> const& bias)
{
    auto const p = params(frames, components, row0, row1, tile);
    CHECK_INPUT(m, torch::kHalf);
    void const* biasPtr = nullptr;
    if (bias.has_value())
    {
        CHECK_INPUT(bias.value(), frames.scalar_type());
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == p.channels, "bias must be [C]");
        biasPtr = bias->data_ptr();
    }
    int64_t const coords = (tile + 2) * (tile + 2);
    int64_t const ntiles = static_cast<int64_t>(p.row1 - p.row0) * ((p.width + tile - 1) / tile);
    TORCH_CHECK(m.dim() == 3 && m.size(0) == components * coords && m.size(1) == (p.frames / components) * ntiles
            && m.size(2) == p.channels,
        "m must be [components*(tile+2)^2, (N/components)*ntiles, C]");
    tensorrt_llm::kernels::launchVaeWinoSpatialOutput(
        m.data_ptr(), frames.data_ptr(), biasPtr, p, at::cuda::getCurrentCUDAStream(frames.get_device()));
    return frames;
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "vae_wino_spatial_in(Tensor frames, int components, int row0, int row1, int tile=2, bool stack_frames=False)"
        " -> Tensor");
    m.def("vae_wino_spatial_in_taps(Tensor[] taps, int zero_front, int row0, int row1, int tile=2) -> Tensor");
    m.def(
        "vae_wino_spatial_out(Tensor m, Tensor(a!) frames, int components, int row0, int row1, int tile=2,"
        " Tensor? bias=None) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("vae_wino_spatial_in", &vae_wino_spatial_in);
    m.impl("vae_wino_spatial_in_taps", &vae_wino_spatial_in_taps);
    m.impl("vae_wino_spatial_out", &vae_wino_spatial_out);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
