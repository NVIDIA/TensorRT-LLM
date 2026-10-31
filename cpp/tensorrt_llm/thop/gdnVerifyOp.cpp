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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/gdnVerifyKernels.h"

#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <optional>
#include <torch/extension.h>

namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

void checkCudaBf16(at::Tensor const& t, char const* name)
{
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16, name, " must be a CUDA bfloat16 tensor");
}

void checkCudaInt32(at::Tensor const& t, char const* name)
{
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kInt && t.is_contiguous(), name,
        " must be a contiguous CUDA int32 tensor");
}

} // namespace

//! Replay + gated RMSNorm of one MTP target-verify step of a gated DeltaNet layer (see gdnVerifyKernels.h).
//! Updates ssm_states (checkpoint commit), old_u, old_k and old_g in place and returns the norm output
//! [N*T, HV*128], FP8 E4M3 when fp8_scale is given, else bf16.
at::Tensor gdnVerify(at::Tensor const& qkv, at::Tensor const& z, at::Tensor const& a, at::Tensor const& b,
    at::Tensor const& logA, at::Tensor const& dtBias, at::Tensor const& normWeight,
    std::optional<at::Tensor> const& fp8Scale, double normEps, at::Tensor& ssmStates, at::Tensor const& stateIndices,
    at::Tensor& oldU, at::Tensor& oldK, at::Tensor& oldG, at::Tensor const& cacheBufIdx,
    at::Tensor const& prevNumAcceptedTokens)
{
    constexpr int64_t kDim = tk::kGdnVerifyHeadDim;
    constexpr int64_t kHist = tk::kGdnVerifyHistory;
    TORCH_CHECK(tensorrt_llm::common::getSMVersion() >= 90, "gdn_verify needs SM90+");

    checkCudaInt32(stateIndices, "state_indices");
    checkCudaInt32(cacheBufIdx, "cache_buf_idx");
    checkCudaInt32(prevNumAcceptedTokens, "prev_num_accepted_tokens");
    int64_t const numRequests = stateIndices.size(0);
    TORCH_CHECK(numRequests > 0 && qkv.dim() == 2 && qkv.size(0) % numRequests == 0,
        "qkv must be [num_requests * num_tokens, channels]");
    int64_t const numTokens = qkv.size(0) / numRequests;
    TORCH_CHECK(numTokens >= tk::kGdnVerifyMinTokens && numTokens <= tk::kGdnVerifyMaxTokens, "gdn_verify supports ",
        tk::kGdnVerifyMinTokens, " to ", tk::kGdnVerifyMaxTokens, " tokens per request, got ", numTokens);

    checkCudaBf16(oldU, "old_u");
    checkCudaBf16(oldK, "old_k");
    TORCH_CHECK(
        oldU.is_contiguous() && oldU.dim() == 5 && oldU.size(1) == 2 && oldU.size(2) == kHist && oldU.size(4) == kDim,
        "old_u must be a contiguous [slots, 2, 16, HV, 128] tensor");
    TORCH_CHECK(
        oldK.is_contiguous() && oldK.dim() == 5 && oldK.size(1) == 2 && oldK.size(2) == kHist && oldK.size(4) == kDim,
        "old_k must be a contiguous [slots, 2, 16, HK, 128] tensor");
    int64_t const numVHeads = oldU.size(3);
    int64_t const numKHeads = oldK.size(3);
    TORCH_CHECK(numVHeads % numKHeads == 0, "the value heads must be a multiple of the key heads");
    TORCH_CHECK(oldG.is_cuda() && oldG.scalar_type() == at::kFloat && oldG.is_contiguous()
            && oldG.sizes() == at::IntArrayRef({oldU.size(0), 2, numVHeads, kHist}),
        "old_g must be a contiguous float32 [slots, 2, HV, 16] tensor");

    checkCudaBf16(qkv, "qkv");
    TORCH_CHECK(qkv.size(1) >= (2 * numKHeads + numVHeads) * kDim && qkv.stride(1) == 1 && qkv.stride(0) % 8 == 0
            && reinterpret_cast<uintptr_t>(qkv.data_ptr()) % 16 == 0,
        "qkv must hold the q|k|v channels with 16-byte aligned rows");
    checkCudaBf16(z, "z");
    TORCH_CHECK(z.dim() == 3 && z.size(0) == qkv.size(0) && z.size(1) == numVHeads && z.size(2) == kDim
            && z.stride(2) == 1 && z.stride(1) == kDim,
        "z must be [num_requests * num_tokens, HV, 128] with packed heads");
    checkCudaBf16(a, "a");
    checkCudaBf16(b, "b");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(0) == qkv.size(0) && b.size(0) == qkv.size(0)
            && a.size(1) == numVHeads && b.size(1) == numVHeads && a.stride(1) == 1 && b.stride(1) == 1
            && a.stride(0) == b.stride(0),
        "a and b must be [num_requests * num_tokens, HV] with a common row stride");

    TORCH_CHECK(logA.is_cuda() && logA.scalar_type() == at::kFloat && logA.is_contiguous() && logA.numel() == numVHeads,
        "A_log must be a contiguous float32 [HV] tensor");
    TORCH_CHECK(
        dtBias.is_cuda() && dtBias.scalar_type() == at::kFloat && dtBias.is_contiguous() && dtBias.numel() == numVHeads,
        "dt_bias must be a contiguous float32 [HV] tensor");
    checkCudaBf16(normWeight, "norm_weight");
    TORCH_CHECK(normWeight.is_contiguous() && normWeight.numel() == kDim, "norm_weight must be contiguous [128]");
    if (fp8Scale.has_value())
    {
        TORCH_CHECK(fp8Scale->is_cuda() && fp8Scale->scalar_type() == at::kFloat && fp8Scale->numel() == 1,
            "fp8_scale must be a float32 scalar tensor");
    }

    checkCudaBf16(ssmStates, "ssm_states");
    TORCH_CHECK(ssmStates.dim() == 4 && ssmStates.size(1) == numVHeads && ssmStates.size(2) == kDim
            && ssmStates.size(3) == kDim && ssmStates.stride(3) == 1 && ssmStates.stride(2) == kDim
            && ssmStates.stride(1) == kDim * kDim && ssmStates.stride(0) % 8 == 0
            && reinterpret_cast<uintptr_t>(ssmStates.data_ptr()) % 16 == 0,
        "ssm_states must be [slots, HV, 128, 128] with packed, 16-byte aligned slots");

    auto const outDtype = fp8Scale.has_value() ? at::kFloat8_e4m3fn : at::kBFloat16;
    at::Tensor out = at::empty({qkv.size(0), numVHeads * kDim}, qkv.options().dtype(outDtype));

    tk::GdnVerifyParams params{};
    params.qkv = qkv.data_ptr();
    params.z = z.data_ptr();
    params.a = a.data_ptr();
    params.b = b.data_ptr();
    params.qkvStride = qkv.stride(0);
    params.zStride = z.stride(0);
    params.abStride = a.stride(0);
    params.logA = logA.data_ptr<float>();
    params.dtBias = dtBias.data_ptr<float>();
    params.normWeight = normWeight.data_ptr();
    params.fp8Scale = fp8Scale.has_value() ? fp8Scale->data_ptr<float>() : nullptr;
    params.normEps = static_cast<float>(normEps);
    params.ssmStates = ssmStates.data_ptr();
    params.ssmSlotStride = ssmStates.stride(0);
    params.stateIndices = stateIndices.data_ptr<int32_t>();
    params.oldU = oldU.data_ptr();
    params.oldK = oldK.data_ptr();
    params.oldG = oldG.data_ptr<float>();
    params.cacheBufIdx = cacheBufIdx.data_ptr<int32_t>();
    params.prevNumAcceptedTokens = prevNumAcceptedTokens.data_ptr<int32_t>();
    params.out = out.data_ptr();
    params.outStride = out.stride(0);
    params.numRequests = static_cast<int>(numRequests);
    params.numTokens = static_cast<int>(numTokens);
    params.numKHeads = static_cast<int>(numKHeads);
    params.numVHeads = static_cast<int>(numVHeads);
    tk::invokeGdnVerify(params, at::cuda::getCurrentCUDAStream(qkv.get_device()));
    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "gdn_verify(Tensor qkv, Tensor z, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, Tensor norm_weight, "
        "Tensor? fp8_scale, float norm_eps, Tensor(a!) ssm_states, Tensor state_indices, Tensor(b!) old_u, "
        "Tensor(c!) old_k, Tensor(d!) old_g, Tensor cache_buf_idx, Tensor prev_num_accepted_tokens) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("gdn_verify", &tensorrt_llm::torch_ext::gdnVerify);
}
