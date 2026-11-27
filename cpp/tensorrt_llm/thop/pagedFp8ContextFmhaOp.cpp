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

// Causal FP8 context attention over TRT-LLM's paged HND KV pool through the trtllm-gen FMHA runner, with a caller-fixed
// head dimension per K/V stage: a precompiled stage-KV kernel when one matches the selected config; otherwise the op
// declines (NVRTC only with the test-only allow_nvrtc).

#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunner.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <climits>
#include <map>
#include <memory>
#include <mutex>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace tk = tensorrt_llm::kernels;

// One E4M3-in / BF16-out runner per device; the runner holds the loaded cubins and the NVRTC kernel cache.
tk::TllmGenFmhaRunner& runnerFor(int device)
{
    static std::mutex mutex;
    static std::map<int, std::unique_ptr<tk::TllmGenFmhaRunner>> runners;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = runners.find(device);
    if (it == runners.end())
    {
        it = runners
                 .emplace(device,
                     std::make_unique<tk::TllmGenFmhaRunner>(
                         tk::DATA_TYPE_E4M3, tk::DATA_TYPE_E4M3, tk::DATA_TYPE_E4M3, tk::DATA_TYPE_BF16))
                 .first;
    }
    return *it->second;
}

// The shape-independent part of the runner params: layout, mask, kernel type, head dims, page size and stage.
tk::TllmGenFmhaRunnerParams makeParams(int64_t headDim, int64_t tokensPerPage, int64_t headDimPerStageKv)
{
    tk::TllmGenFmhaRunnerParams p{};
    p.mQkvLayout = tk::QkvLayout::PagedKv;
    p.mMaskType = tk::TrtllmGenAttentionMaskType::Causal;
    p.mKernelType = tk::FmhaKernelType::Context;
    p.mTileScheduler = tk::TileScheduler::Persistent;
    p.mHeadDimQk = static_cast<int>(headDim);
    p.mHeadDimV = static_cast<int>(headDim);
    p.mNumTokensPerPage = static_cast<int>(tokensPerPage);
    p.mHeadDimPerStageKv = static_cast<int>(headDimPerStageKv);
    return p;
}

} // namespace

// q: e4m3 [T, Hq, D]; kv_pool: e4m3 [pages, Hkv, tokens_per_page, D]; block_tables: int32 [B, 2, max_pages] (K and V
// page indices); seq_lens_kv: int32 [B]; cu_seqlens_q / cu_seqlens_kv: int32 [B + 1]; out: bf16 [T, Hq * D], written.
// scale_bmm1: fp32 [2] on device, [softmax scale, softmax scale * log2(e)] with the Q / K descales folded in.
// scale_out: fp32 [1] on device, the V descale. head_dim_per_stage_kv: the head dim per K/V stage (> 0).
// Returns false, launching nothing, when the kernel the runner selects for this call has no precompiled cubin and
// allow_nvrtc is false (the caller then runs another kernel instead of compiling one inside the step).
bool fmha_paged_context_fp8(torch::Tensor const& q, torch::Tensor const& kv_pool, torch::Tensor const& block_tables,
    torch::Tensor const& seq_lens_kv, torch::Tensor const& cu_seqlens_q, torch::Tensor const& cu_seqlens_kv,
    torch::Tensor& out, torch::Tensor const& scale_bmm1, torch::Tensor const& scale_out, int64_t max_q_len,
    int64_t max_kv_len, int64_t head_dim_per_stage_kv, bool allow_nvrtc)
{
    TORCH_CHECK(q.scalar_type() == c10::ScalarType::Float8_e4m3fn && kv_pool.scalar_type() == q.scalar_type(),
        "fmha_paged_context_fp8: q and kv_pool must be float8_e4m3fn");
    TORCH_CHECK(q.dim() == 3 && q.is_contiguous(), "fmha_paged_context_fp8: q must be a contiguous [T, H, D] tensor");
    TORCH_CHECK(kv_pool.dim() == 4 && kv_pool.is_contiguous() && kv_pool.size(3) == q.size(2),
        "fmha_paged_context_fp8: kv_pool must be a contiguous [pages, Hkv, tokens_per_page, D] tensor");
    // && short-circuits: the multiple check divides by the KV heads only once there are some.
    TORCH_CHECK(kv_pool.size(1) > 0 && q.size(1) % kv_pool.size(1) == 0,
        "fmha_paged_context_fp8: kv_pool must have KV heads and Q heads must be a multiple of them");
    TORCH_CHECK(out.scalar_type() == c10::ScalarType::BFloat16 && out.is_contiguous() && out.numel() == q.numel(),
        "fmha_paged_context_fp8: out must be a contiguous bf16 tensor with q's element count");
    TORCH_CHECK(block_tables.scalar_type() == c10::ScalarType::Int && block_tables.dim() == 3
            && block_tables.size(1) == 2 && block_tables.is_contiguous(),
        "fmha_paged_context_fp8: block_tables must be a contiguous int32 [B, 2, max_pages] tensor");
    int64_t const batch = block_tables.size(0);
    for (auto const* t : {&seq_lens_kv, &cu_seqlens_q, &cu_seqlens_kv})
    {
        TORCH_CHECK(t->scalar_type() == c10::ScalarType::Int && t->dim() == 1,
            "fmha_paged_context_fp8: sequence lengths must be int32 vectors");
    }
    TORCH_CHECK(seq_lens_kv.numel() >= batch && cu_seqlens_q.numel() > batch && cu_seqlens_kv.numel() > batch,
        "fmha_paged_context_fp8: sequence length vectors are shorter than the batch");
    TORCH_CHECK(scale_bmm1.scalar_type() == c10::ScalarType::Float && scale_bmm1.numel() >= 2,
        "fmha_paged_context_fp8: scale_bmm1 must be float32 [2]");
    TORCH_CHECK(scale_out.scalar_type() == c10::ScalarType::Float && scale_out.numel() >= 1,
        "fmha_paged_context_fp8: scale_out must be float32 [1]");
    TORCH_CHECK(max_q_len > 0 && max_kv_len >= max_q_len, "fmha_paged_context_fp8: invalid max sequence lengths");
    TORCH_CHECK(head_dim_per_stage_kv > 0, "fmha_paged_context_fp8: head_dim_per_stage_kv must be > 0");
    for (at::Tensor const* t : std::initializer_list<at::Tensor const*>{
             &kv_pool, &block_tables, &seq_lens_kv, &cu_seqlens_q, &cu_seqlens_kv, &out, &scale_bmm1, &scale_out})
    {
        TORCH_CHECK(t->device() == q.device(), "fmha_paged_context_fp8: all tensors must be on the device of q");
    }

    c10::cuda::CUDAGuard const deviceGuard(q.device());
    int const device = q.get_device();
    int64_t const numTokens = q.size(0);
    int64_t const numHeadsQ = q.size(1);
    int64_t const numHeadsKv = kv_pool.size(1);
    int64_t const headDim = q.size(2);

    tk::TllmGenFmhaRunnerParams p = makeParams(headDim, kv_pool.size(2), head_dim_per_stage_kv);
    p.mMultiCtasKvMode = false;
    p.qPtr = q.data_ptr();
    p.kvPtr = kv_pool.data_ptr();
    p.kvPageIdxPtr = block_tables.data_ptr<int>();
    p.seqLensKvPtr = seq_lens_kv.data_ptr<int>();
    p.cumSeqLensQPtr = cu_seqlens_q.data_ptr<int>();
    p.cumSeqLensKvPtr = cu_seqlens_kv.data_ptr<int>();
    p.oPtr = out.data_ptr();
    // No softmax stats: the kernels skip that store for a null pointer, as in the regular context dispatch.
    p.softmaxStatsPtr = nullptr;
    // The kernels read both scales from device memory.
    p.scaleSoftmaxLog2Ptr = scale_bmm1.data_ptr<float>() + tk::kIdxScaleSoftmaxLog2Ptr;
    p.outputScalePtr = scale_out.data_ptr<float>();
    p.mNumHeadsQ = static_cast<int>(numHeadsQ);
    p.mNumHeadsKv = static_cast<int>(numHeadsKv);
    p.mNumHeadsQPerKv = static_cast<int>(numHeadsQ / numHeadsKv);
    p.mBatchSize = static_cast<int>(batch);
    p.mMaxSeqLenQ = static_cast<int>(max_q_len);
    p.mMaxSeqLenKv = static_cast<int>(max_kv_len);
    p.mMaxSeqLenCacheKv = static_cast<int>(max_kv_len);
    p.mSumOfSeqLensQ = static_cast<int>(numTokens);
    p.mSumOfSeqLensKv = static_cast<int>(std::min<int64_t>(batch * max_kv_len, INT_MAX));
    p.mMaxNumPagesPerSeqKv = static_cast<int>(block_tables.size(2));
    p.mNumPagesInMemPool = static_cast<int>(std::min<int64_t>(kv_pool.size(0), INT_MAX));
    p.mChunkedAttentionSize = INT_MAX;
    p.mLeftSlidingWindow = -1;
    p.mRightSlidingWindow = -1;
    p.mScaleQ = 1.0f;
    int smCount = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
    p.mMultiProcessorCount = smCount;
    p.stream = at::cuda::getCurrentCUDAStream(device);
    tk::TllmGenFmhaRunner& runner = runnerFor(device);
    if (!allow_nvrtc && runner.usesNvrtc(p))
    {
        return false;
    }
    runner.run(p);
    return true;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fmha_paged_context_fp8(Tensor q, Tensor kv_pool, Tensor block_tables, Tensor seq_lens_kv, "
        "Tensor cu_seqlens_q, Tensor cu_seqlens_kv, Tensor(a!) out, Tensor scale_bmm1, Tensor scale_out, "
        "int max_q_len, int max_kv_len, int head_dim_per_stage_kv, bool allow_nvrtc=False) -> bool");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fmha_paged_context_fp8", &tensorrt_llm::torch_ext::fmha_paged_context_fp8);
}
