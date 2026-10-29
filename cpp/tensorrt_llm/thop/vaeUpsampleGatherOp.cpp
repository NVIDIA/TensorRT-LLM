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

#include "tensorrt_llm/kernels/vaeUpsampleGatherKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// temp: [B*T, 4*Cout, H+1, W+1] channels-last (the packed 2x2 convolution output).
// shortcut: optional [B, Cin, Tr, H, W] channels_last_3d (DupUp3D input).
// Returns [B, Cout, T, 2H, 2W] in channels_last_3d.
torch::Tensor vae_upsample_gather(torch::Tensor const& temp, std::optional<torch::Tensor> const& shortcut,
    int64_t batch, int64_t frames, int64_t factor_t, int64_t repeats, int64_t crop,
    std::optional<torch::Tensor> const& bias)
{
    TORCH_CHECK(temp.dim() == 4, "temp must be [B*T, 4*Cout, H+1, W+1]");
    TORCH_CHECK(
        temp.scalar_type() == torch::kBFloat16 || temp.scalar_type() == torch::kHalf, "temp must be bf16 or fp16");
    TORCH_CHECK(temp.is_cuda() && temp.is_contiguous(at::MemoryFormat::ChannelsLast), "temp must be channels-last");
    TORCH_CHECK(temp.size(0) == batch * frames, "temp batch must be B*T");
    TORCH_CHECK(temp.size(1) % 4 == 0, "temp channels must be 4*Cout");
    int64_t const cout = temp.size(1) / 4;
    int64_t const height = temp.size(2) - 1;
    int64_t const width = temp.size(3) - 1;
    TORCH_CHECK(height >= 1 && width >= 1 && cout % 8 == 0, "unsupported temp shape");

    tensorrt_llm::kernels::VaeUpsampleGatherParams p{};
    p.batch = static_cast<int>(batch);
    p.frames = static_cast<int>(frames);
    p.height = static_cast<int>(height);
    p.width = static_cast<int>(width);
    p.cout = static_cast<int>(cout);
    p.cin = 0;
    p.factorT = static_cast<int>(factor_t);
    p.repeats = static_cast<int>(repeats);
    p.crop = static_cast<int>(crop);
    void const* sPtr = nullptr;
    if (shortcut.has_value())
    {
        auto const& s = shortcut.value();
        TORCH_CHECK(s.dim() == 5 && s.scalar_type() == temp.scalar_type() && s.is_cuda(), "bad shortcut tensor");
        TORCH_CHECK(s.is_contiguous(at::MemoryFormat::ChannelsLast3d), "shortcut must be channels_last_3d");
        TORCH_CHECK(s.size(0) == batch && s.size(3) == height && s.size(4) == width, "shortcut spatial mismatch");
        TORCH_CHECK(s.size(2) * factor_t == frames + crop, "shortcut frames do not match the output frames");
        TORCH_CHECK(s.size(1) * repeats == cout * factor_t * 4, "shortcut channel repeat mismatch");
        p.cin = static_cast<int>(s.size(1));
        sPtr = s.data_ptr();
    }

    void const* bPtr = nullptr;
    if (bias.has_value())
    {
        auto const& b = bias.value();
        TORCH_CHECK(b.scalar_type() == temp.scalar_type() && b.is_cuda() && b.is_contiguous() && b.numel() == cout,
            "bias must be a contiguous [Cout] vector of temp's dtype");
        bPtr = b.data_ptr();
    }
    auto out
        = torch::empty({batch, cout, frames, 2 * height, 2 * width}, temp.options(), at::MemoryFormat::ChannelsLast3d);
    auto stream = at::cuda::getCurrentCUDAStream(temp.get_device());
    tensorrt_llm::kernels::launchVaeUpsampleGather(
        temp.data_ptr(), sPtr, bPtr, out.data_ptr(), p, temp.scalar_type() == torch::kBFloat16, stream);
    return out;
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "vae_upsample_gather(Tensor temp, Tensor? shortcut, int batch, int frames, int factor_t, int repeats, "
        "int crop, Tensor? bias=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("vae_upsample_gather", &vae_upsample_gather);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
