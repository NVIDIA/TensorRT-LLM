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

// Dense (no mask, no paging) FP8 context attention through the trtllm-gen FMHA runner, for
// diffusion-transformer self-attention: packed tokens [T, H, D], equal Q and KV lengths, MHA.

#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunner.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <climits>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace tk = tensorrt_llm::kernels;

tk::Data_type toFmhaOutType(c10::ScalarType dtype)
{
    switch (dtype)
    {
    case c10::ScalarType::BFloat16: return tk::DATA_TYPE_BF16;
    case c10::ScalarType::Float8_e4m3fn: return tk::DATA_TYPE_E4M3;
    default: TORCH_CHECK(false, "fmha_dense_fp8: unsupported output dtype ", dtype);
    }
}

// One runner per (output dtype, device); the runner holds the loaded cubins.
tk::TllmGenFmhaRunner& runnerFor(tk::Data_type dtypeOut, int device)
{
    static std::mutex mutex;
    static std::map<std::pair<int, int>, std::unique_ptr<tk::TllmGenFmhaRunner>> runners;
    std::lock_guard<std::mutex> lock(mutex);
    auto const key = std::make_pair(static_cast<int>(dtypeOut), device);
    auto it = runners.find(key);
    if (it == runners.end())
    {
        it = runners
                 .emplace(key,
                     std::make_unique<tk::TllmGenFmhaRunner>(
                         tk::DATA_TYPE_E4M3, tk::DATA_TYPE_E4M3, tk::DATA_TYPE_E4M3, dtypeOut))
                 .first;
    }
    return *it->second;
}

// Shape-only parameters; callers fill the pointers and the stream before running.
tk::TllmGenFmhaRunnerParams makeParams(int64_t batch, int64_t maxSeqLen, int64_t numTokens, int64_t numHeads,
    int64_t headDim, bool fp16Softmax, int device)
{
    tk::TllmGenFmhaRunnerParams p{};
    p.mQkvLayout = tk::QkvLayout::SeparateQkv;
    p.mMaskType = tk::TrtllmGenAttentionMaskType::Dense;
    p.mKernelType = tk::FmhaKernelType::Context;
    p.mTileScheduler = tk::TileScheduler::Persistent;
    p.mMultiCtasKvMode = false;
    p.mHeadDimQk = static_cast<int>(headDim);
    p.mHeadDimV = static_cast<int>(headDim);
    p.mNumHeadsQ = static_cast<int>(numHeads);
    p.mNumHeadsKv = static_cast<int>(numHeads);
    p.mNumHeadsQPerKv = 1;
    p.mBatchSize = static_cast<int>(batch);
    p.mMaxSeqLenQ = static_cast<int>(maxSeqLen);
    p.mMaxSeqLenKv = static_cast<int>(maxSeqLen);
    p.mSumOfSeqLensQ = static_cast<int>(numTokens);
    p.mSumOfSeqLensKv = static_cast<int>(numTokens);
    p.mChunkedAttentionSize = INT_MAX;
    p.mLeftSlidingWindow = -1;
    p.mRightSlidingWindow = -1;
    p.mScaleQ = 1.0f;
    p.mNumPagesInMemPool = INT_MAX;
    int smCount = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
    p.mMultiProcessorCount = smCount;
    p.mFp16Softmax = fp16Softmax;
    return p;
}

} // namespace

// q, k, v: e4m3 [T, H, D] with T = sum of the sequence lengths in cu_seqlens (int32 [B + 1]).
// scale_bmm1: fp32 [2] on device, [1 / log2(e), softmax scale * log2(e) * q_descale * k_descale];
//   required without fp16_softmax and unused with it.
// scale_out: fp32 [1] on device, v_descale for BF16 output or v_descale / out_scale for FP8 output.
// With fp16_softmax the softmax scale must be folded into the Q/K quantization scales; the op runs
// the kernels with a unit scaleSoftmaxLog2 of its own.
torch::Tensor fmha_dense_fp8(torch::Tensor const& q, torch::Tensor const& k, torch::Tensor const& v,
    torch::Tensor const& cu_seqlens, int64_t max_seqlen, c10::optional<torch::Tensor> const& scale_bmm1,
    torch::Tensor const& scale_out, bool fp16_softmax, c10::ScalarType out_dtype)
{
    TORCH_CHECK(q.scalar_type() == c10::ScalarType::Float8_e4m3fn && k.scalar_type() == q.scalar_type()
            && v.scalar_type() == q.scalar_type(),
        "fmha_dense_fp8: q, k and v must be float8_e4m3fn");
    TORCH_CHECK(q.dim() == 3 && k.sizes() == q.sizes() && v.sizes() == q.sizes(),
        "fmha_dense_fp8: q, k and v must share one [T, H, D] shape");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "fmha_dense_fp8: q, k, v not contiguous");
    TORCH_CHECK(cu_seqlens.scalar_type() == c10::ScalarType::Int && cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2,
        "fmha_dense_fp8: cu_seqlens must be int32 [B + 1]");
    TORCH_CHECK(max_seqlen > 0 && max_seqlen <= q.size(0),
        "fmha_dense_fp8: max_seqlen must be in [1, num_tokens]; got ", max_seqlen);
    // The FP16-softmax kernels take the softmax scale folded into Q/K, so they always run with 1.0.
    auto const softmaxScale = fp16_softmax ? torch::ones({1}, q.options().dtype(torch::kFloat32)) : torch::Tensor();
    if (!fp16_softmax)
    {
        TORCH_CHECK(
            scale_bmm1.has_value() && scale_bmm1->scalar_type() == c10::ScalarType::Float && scale_bmm1->numel() >= 2,
            "fmha_dense_fp8: scale_bmm1 must be float32 [2]");
        TORCH_CHECK(scale_bmm1->device() == q.device(), "fmha_dense_fp8: all tensors must be on the device of q");
    }
    TORCH_CHECK(scale_out.scalar_type() == c10::ScalarType::Float && scale_out.numel() >= 1,
        "fmha_dense_fp8: scale_out must be float32 [1]");
    for (auto const* t : {&k, &v, &cu_seqlens, &scale_out})
    {
        TORCH_CHECK(t->device() == q.device(), "fmha_dense_fp8: all tensors must be on the device of q");
    }

    c10::cuda::CUDAGuard const deviceGuard(q.device());
    int const device = q.get_device();
    int64_t const batch = cu_seqlens.numel() - 1;
    auto const seqLens = (cu_seqlens.slice(0, 1) - cu_seqlens.slice(0, 0, batch)).contiguous();
    auto out = torch::empty(q.sizes(), q.options().dtype(out_dtype));

    auto& runner = runnerFor(toFmhaOutType(out_dtype), device);
    auto p = makeParams(batch, max_seqlen, q.size(0), q.size(1), q.size(2), fp16_softmax, device);
    p.qPtr = q.data_ptr();
    p.kPtr = k.data_ptr();
    p.vPtr = v.data_ptr();
    p.oPtr = out.data_ptr();
    p.cumSeqLensQPtr = cu_seqlens.data_ptr<int>();
    p.cumSeqLensKvPtr = cu_seqlens.data_ptr<int>();
    p.seqLensKvPtr = seqLens.data_ptr<int>();
    // The kernels read both scales from device memory, so the op stays valid under graph replay.
    p.scaleSoftmaxLog2Ptr
        = fp16_softmax ? softmaxScale.data_ptr<float>() : scale_bmm1->data_ptr<float>() + tk::kIdxScaleSoftmaxLog2Ptr;
    p.outputScalePtr = scale_out.data_ptr<float>();
    p.stream = at::cuda::getCurrentCUDAStream(device);

    auto const [supported, info] = runner.isSupportedWithInfo(p);
    TORCH_CHECK(supported, "fmha_dense_fp8: no trtllm-gen kernel for this configuration: ", info);
    runner.run(p);
    return out;
}

// Whether a kernel exists for this head layout on the current device; never throws.
bool fmha_dense_fp8_is_supported(int64_t head_dim, int64_t num_heads, c10::ScalarType out_dtype, bool fp16_softmax)
{
    if (out_dtype != c10::ScalarType::BFloat16 && out_dtype != c10::ScalarType::Float8_e4m3fn)
    {
        return false;
    }
    try
    {
        int const device = c10::cuda::current_device();
        auto& runner = runnerFor(toFmhaOutType(out_dtype), device);
        constexpr int64_t kProbeSeqLen = 128;
        auto const p = makeParams(1, kProbeSeqLen, kProbeSeqLen, num_heads, head_dim, fp16_softmax, device);
        return runner.isSupported(p);
    }
    catch (std::exception const&)
    {
        // No CUDA device, or an architecture the trtllm-gen runner does not support.
        return false;
    }
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fmha_dense_fp8(Tensor q, Tensor k, Tensor v, Tensor cu_seqlens, int max_seqlen, Tensor? scale_bmm1, "
        "Tensor scale_out, bool fp16_softmax, ScalarType out_dtype) -> Tensor");
    m.def("fmha_dense_fp8_is_supported(int head_dim, int num_heads, ScalarType out_dtype, bool fp16_softmax) -> bool");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fmha_dense_fp8", &tensorrt_llm::torch_ext::fmha_dense_fp8);
}

// No tensor arguments, so dispatch cannot pick a backend from the inputs.
TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("fmha_dense_fp8_is_supported", &tensorrt_llm::torch_ext::fmha_dense_fp8_is_supported);
}
