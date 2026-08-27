/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorrt_llm/kernels/kimiK3AttnRes/attnResFwd.h"

#include <ATen/Functions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/library.h>

#include <tuple>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

bool is_sm100_family()
{
    int dev = 0;
    cudaGetDevice(&dev);
    static int cached_state[64] = {}; // 0 unknown, 1 false, 2 true
    if (dev >= 0 && dev < 64 && cached_state[dev] != 0)
    {
        return cached_state[dev] == 2;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    // The kernel binary is compiled for the sm_100 family only (it relies on
    // tcgen05/TMEM, which later architectures such as sm_120 do not support),
    // so require compute capability major == 10 rather than >= 10.
    bool const ok = prop.major == 10;
    if (dev >= 0 && dev < 64)
    {
        cached_state[dev] = ok ? 2 : 1;
    }
    return ok;
}

void check_attn_res_contract(int N, int T, int B, int H)
{
    TORCH_CHECK(is_sm100_family(), "attn_res_fwd requires an sm_100-family (datacenter Blackwell) GPU");
    TORCH_CHECK(B == 1, "attn_res_fwd: unsupported B=", B, " (only B=1 is supported)");
    TORCH_CHECK(N >= 1 && N <= 12, "attn_res_fwd: unsupported N=", N, " (must be in [1, 12])");
    TORCH_CHECK(T >= 1 && T <= 16384, "attn_res_fwd: unsupported T=", T, " (must be in [1, 16384])");
    TORCH_CHECK(H >= 4096 && H <= 8192 && H % 1024 == 0, "attn_res_fwd: unsupported H=", H,
        " (must be a multiple of 1024 in [4096, 8192])");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> attn_res_fwd(
    at::Tensor layer_residual, at::Tensor block_residual, at::Tensor res_weight, at::Tensor rms_weight, double rms_eps)
{
    TORCH_CHECK(layer_residual.dim() == 3, "attn_res_fwd: layer_residual must be [T, B, H]");
    TORCH_CHECK(block_residual.dim() == 4, "attn_res_fwd: block_residual must be [K, T, B, H]");

    int const T = static_cast<int>(layer_residual.size(0));
    int const B = static_cast<int>(layer_residual.size(1));
    int const H = static_cast<int>(layer_residual.size(2));
    int const N = static_cast<int>(block_residual.size(0)) + 1;
    TORCH_CHECK(layer_residual.is_cuda() && block_residual.is_cuda() && res_weight.is_cuda() && rms_weight.is_cuda(),
        "attn_res_fwd: all input tensors must be CUDA tensors");
    // Set the device before check_attn_res_contract: is_sm100_family() reads
    // the current device, which must match the tensors' device.
    c10::cuda::CUDAGuard device_guard(layer_residual.device());
    check_attn_res_contract(N, T, B, H);

    TORCH_CHECK(layer_residual.scalar_type() == at::kBFloat16, "attn_res_fwd: layer_residual must be bf16");
    TORCH_CHECK(block_residual.scalar_type() == at::kBFloat16, "attn_res_fwd: block_residual must be bf16");
    TORCH_CHECK(res_weight.scalar_type() == at::kBFloat16, "attn_res_fwd: res_weight must be bf16");
    TORCH_CHECK(rms_weight.scalar_type() == at::kBFloat16, "attn_res_fwd: rms_weight must be bf16");
    TORCH_CHECK(layer_residual.is_contiguous() && block_residual.is_contiguous() && res_weight.is_contiguous()
            && rms_weight.is_contiguous(),
        "attn_res_fwd: inputs must be contiguous");
    TORCH_CHECK(block_residual.sizes() == at::IntArrayRef({N - 1, T, B, H}),
        "attn_res_fwd: block_residual shape must match layer_residual");
    TORCH_CHECK(res_weight.numel() == H, "attn_res_fwd: res_weight must have H elements");
    TORCH_CHECK(rms_weight.numel() == H, "attn_res_fwd: rms_weight must have H elements");

    auto output = at::empty_like(layer_residual);
    auto float_options = layer_residual.options().dtype(at::kFloat);
    auto rsigma = at::empty({N, T, B}, float_options);
    auto probs = at::empty({N, T, B}, float_options);
    auto logits = at::empty({N, T, B}, float_options);

    kernels::kimiK3AttnRes::AttnResFwdParams params{};
    params.blockResidual = N > 1 ? reinterpret_cast<__nv_bfloat16 const*>(block_residual.const_data_ptr()) : nullptr;
    params.layerResidual = reinterpret_cast<__nv_bfloat16 const*>(layer_residual.const_data_ptr());
    params.resWeight = reinterpret_cast<__nv_bfloat16 const*>(res_weight.const_data_ptr());
    params.rmsWeight = reinterpret_cast<__nv_bfloat16 const*>(rms_weight.const_data_ptr());
    params.output = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
    params.rsigma = rsigma.data_ptr<float>();
    params.probs = probs.data_ptr<float>();
    params.logits = logits.data_ptr<float>();
    params.numCandidates = N;
    params.seqLen = T;
    params.batchSize = B;
    params.hiddenSize = H;
    params.rmsEps = static_cast<float>(rms_eps);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    kernels::kimiK3AttnRes::invokeAttnResFwd(params, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {output, rsigma, probs, logits};
}

} // namespace

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "attn_res_fwd(Tensor layer_residual, Tensor block_residual, "
        "Tensor res_weight, Tensor rms_weight, float rms_eps) "
        "-> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("attn_res_fwd", &tensorrt_llm::torch_ext::attn_res_fwd);
}
