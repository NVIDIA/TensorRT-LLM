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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/minimaxM3/shardAccept.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace sa = tensorrt_llm::kernels::minimaxM3::shardAccept;

using OptTensor = std::optional<torch::Tensor>;

void checkArch(torch::Tensor const& t)
{
    TORCH_CHECK(t.is_cuda(), "minimax_m3_shard_accept expects CUDA tensors");
    static int const sSmVersion = tensorrt_llm::common::getSMVersion();
    TORCH_CHECK(sSmVersion >= 100, "minimax_m3_shard_accept requires SM100 or newer");
}

void check(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TORCH_CHECK(false, cudaGetErrorString(err));
    }
}

cudaStream_t currentStream(torch::Tensor const& t)
{
    return at::cuda::getCurrentCUDAStream(t.get_device());
}

//! Per-request accepted counts for fractional forced acceptance (int32 [nreq]), or nullptr.
int const* forceVecPtr(OptTensor const& forceVec, int nreq)
{
    if (!forceVec.has_value())
    {
        return nullptr;
    }
    TORCH_CHECK(forceVec->scalar_type() == at::kInt && forceVec->is_contiguous() && forceVec->numel() >= nreq,
        "accept: force_vec must be contiguous int32 [nreq]");
    return forceVec->data_ptr<int>();
}

} // namespace

//! Greedy verification of nreq * (draft_len + 1) <= 48 verify rows from this rank's BF16 logits shard [rows, V / tp]
//! (global ids vocabOffset..), exchanged across the TP ranks through the multicast slots wsMcPtr / wsUc, plus the
//! strict acceptance (accepted tokens, counts, first-pass gather rows, optional nextNew[:, 0]).
void minimaxM3ShardAccept(torch::Tensor const& logits, int64_t vocabOffset, torch::Tensor& keys,
    torch::Tensor& counters, torch::Tensor& wsUc, int64_t wsMcPtr, int64_t rank, int64_t world,
    torch::Tensor const& draft, torch::Tensor const& batchIdx, torch::Tensor& accTokens, torch::Tensor& numAcc,
    torch::Tensor& gatherIds, int64_t forceTotal, OptTensor const& nextNew, int64_t numCtas, OptTensor const& forceVec)
{
    checkArch(logits);
    c10::cuda::CUDAGuard const deviceGuard{logits.device()};
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16 && logits.dim() == 2 && logits.stride(1) == 1);
    int const M = logits.size(0);
    int const nreq = accTokens.size(0);
    int const k1 = accTokens.size(1);
    TORCH_CHECK(M >= 1 && M <= sa::kMaxRows && nreq * k1 == M && nreq <= sa::kThreads);
    TORCH_CHECK(draft.scalar_type() == at::kInt && batchIdx.scalar_type() == at::kInt && accTokens.is_contiguous());
    sa::AcceptArgs acc{draft.data_ptr<int>(), batchIdx.data_ptr<int>(), accTokens.data_ptr<int>(),
        numAcc.data_ptr<int>(), reinterpret_cast<long long*>(gatherIds.data_ptr<int64_t>()), k1 - 1, nreq,
        static_cast<int>(forceTotal), nextNew.has_value() ? nextNew->data_ptr<int>() : nullptr,
        nextNew.has_value() ? static_cast<int>(nextNew->stride(0)) : 0, forceVecPtr(forceVec, nreq)};
    check(sa::invokeShardAccept(reinterpret_cast<__nv_bfloat16 const*>(logits.data_ptr()), M,
        static_cast<int>(logits.size(1)), static_cast<long long>(logits.stride(0)), static_cast<int>(vocabOffset),
        reinterpret_cast<unsigned long long*>(keys.data_ptr()), reinterpret_cast<unsigned int*>(counters.data_ptr()),
        reinterpret_cast<unsigned long long*>(wsUc.data_ptr()), reinterpret_cast<unsigned long long*>(wsMcPtr),
        static_cast<int>(rank), static_cast<int>(world), acc, static_cast<int>(numCtas), currentStream(logits)));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

// Mutations through the raw multicast address ws_mc_ptr are not visible to the schema: that buffer is owned by the
// Python workspace and never aliased by the op's tensor arguments.
TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m3_shard_accept(Tensor logits, int vocab_offset, Tensor(a!) keys, Tensor(b!) counters, "
        "Tensor(c!) ws_uc, int ws_mc_ptr, int rank, int world, Tensor draft, Tensor batch_idx, "
        "Tensor(d!) acc_tokens, Tensor(e!) num_acc, Tensor(f!) gather_ids, int force_total, Tensor(g!)? next_new, "
        "int num_ctas, Tensor? force_vec) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_shard_accept", &tensorrt_llm::torch_ext::minimaxM3ShardAccept);
}
