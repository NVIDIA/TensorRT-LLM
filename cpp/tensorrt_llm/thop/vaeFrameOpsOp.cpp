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

#include "tensorrt_llm/kernels/vaeFrameOpsKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// y = x + AvgDown3D(r); x, y: [T', P', Cout]; r: [T, P, Cin]. See vaeFrameOpsKernel.h.
torch::Tensor vae_avg_down_add(torch::Tensor const& x, torch::Tensor const& r, int64_t out_height, int64_t out_width,
    int64_t in_width, int64_t factor_t, int64_t factor_s, int64_t group_size, int64_t pad_t,
    std::optional<torch::Tensor> const& bias)
{
    CHECK_INPUT(x, torch::kBFloat16);
    CHECK_INPUT(r, torch::kBFloat16);
    void const* biasPtr = nullptr;
    if (bias.has_value())
    {
        CHECK_INPUT(bias.value(), torch::kBFloat16);
        TORCH_CHECK(bias->numel() == x.size(2), "bias must have Cout values");
        biasPtr = bias->data_ptr();
    }
    TORCH_CHECK(x.dim() == 3 && r.dim() == 3, "frame tensors must be [T, P, C]");
    TORCH_CHECK(x.size(1) == out_height * out_width, "x positions must equal out_height * out_width");
    tensorrt_llm::kernels::VaeAvgDownParams p{};
    p.outPositions = x.size(1);
    p.outHeight = static_cast<int>(out_height);
    p.outWidth = static_cast<int>(out_width);
    p.inWidth = static_cast<int>(in_width);
    p.inFrames = static_cast<int>(r.size(0));
    p.outFrames = static_cast<int>(x.size(0));
    p.cin = static_cast<int>(r.size(2));
    p.cout = static_cast<int>(x.size(2));
    p.factorT = static_cast<int>(factor_t);
    p.factorS = static_cast<int>(factor_s);
    p.groupSize = static_cast<int>(group_size);
    p.padT = static_cast<int>(pad_t);
    TORCH_CHECK((p.inFrames + p.padT) == p.outFrames * p.factorT, "frame counts do not match the temporal factor");
    TORCH_CHECK(p.cin * p.factorT * p.factorS * p.factorS == p.cout * p.groupSize, "channel grouping mismatch");
    auto y = torch::empty_like(x);
    tensorrt_llm::kernels::launchVaeAvgDownAdd(
        x.data_ptr(), r.data_ptr(), biasPtr, y.data_ptr(), p, at::cuda::getCurrentCUDAStream(x.get_device()));
    return y;
}

// u [T, P, 2C] -> y [2T, P, C] with the time conv's bias; see vaeFrameOpsKernel.h.
torch::Tensor vae_interleave_bias(torch::Tensor const& u, std::optional<torch::Tensor> const& bias)
{
    TORCH_CHECK(u.dim() == 3 && u.size(2) % 16 == 0, "u must be [T, P, 2C] with C a multiple of 8");
    CHECK_INPUT(u, torch::kBFloat16);
    int64_t const T = u.size(0), P = u.size(1), C = u.size(2) / 2;
    void const* bPtr = nullptr;
    if (bias.has_value())
    {
        CHECK_INPUT(bias.value(), torch::kBFloat16);
        TORCH_CHECK(bias->numel() == 2 * C, "bias must have 2C values");
        bPtr = bias->data_ptr();
    }
    tensorrt_llm::kernels::VaeInterleaveParams p{};
    p.positions = P;
    p.channels = static_cast<int>(C);
    p.frames = static_cast<int>(T);
    auto y = torch::empty({2 * T, P, C}, u.options());
    tensorrt_llm::kernels::launchVaeInterleaveBias(
        u.data_ptr(), bPtr, y.data_ptr(), p, at::cuda::getCurrentCUDAStream(u.get_device()));
    return y;
}

// chunk [1, C*patch^2, t, H, W] -> out[:, :, frame0:frame0+t] of [1, C, T, H*patch, W*patch], clamped to
// [-1, 1]; both channels-last bf16. See vaeFrameOpsKernel.h.
torch::Tensor vae_unpatchify_clamp(torch::Tensor const& chunk, torch::Tensor const& out, int64_t frame0, int64_t patch)
{
    TORCH_CHECK(chunk.dim() == 5 && out.dim() == 5 && chunk.size(0) == 1 && out.size(0) == 1, "batch-1 5D tensors");
    TORCH_CHECK(chunk.scalar_type() == torch::kBFloat16 && out.scalar_type() == torch::kBFloat16, "bf16 only");
    TORCH_CHECK(
        chunk.is_contiguous(at::MemoryFormat::ChannelsLast3d) && out.is_contiguous(at::MemoryFormat::ChannelsLast3d),
        "chunk and out must be channels-last 3D");
    TORCH_CHECK(patch >= 1 && chunk.size(1) == out.size(1) * patch * patch, "chunk channels must be C * patch^2");
    TORCH_CHECK(chunk.size(3) * patch == out.size(3) && chunk.size(4) * patch == out.size(4), "spatial sizes");
    TORCH_CHECK(frame0 >= 0 && frame0 + chunk.size(2) <= out.size(2), "frame range");
    tensorrt_llm::kernels::VaeUnpatchifyParams p{};
    p.frames = static_cast<int>(chunk.size(2));
    p.height = static_cast<int>(chunk.size(3));
    p.width = static_cast<int>(chunk.size(4));
    p.channels = static_cast<int>(out.size(1));
    p.patch = static_cast<int>(patch);
    p.outFrames = static_cast<int>(out.size(2));
    p.frame0 = static_cast<int>(frame0);
    tensorrt_llm::kernels::launchVaeUnpatchifyClamp(
        chunk.data_ptr(), out.data_ptr(), p, at::cuda::getCurrentCUDAStream(chunk.get_device()));
    return out;
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("vae_unpatchify_clamp(Tensor chunk, Tensor(a!) out, int frame0, int patch) -> Tensor(a!)");
    m.def("vae_interleave_bias(Tensor u, Tensor? bias) -> Tensor");
    m.def(
        "vae_avg_down_add(Tensor x, Tensor r, int out_height, int out_width, int in_width, int factor_t, int factor_s, "
        "int group_size, int pad_t, Tensor? bias=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("vae_interleave_bias", &vae_interleave_bias);
    m.impl("vae_unpatchify_clamp", &vae_unpatchify_clamp);
    m.impl("vae_avg_down_add", &vae_avg_down_add);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
