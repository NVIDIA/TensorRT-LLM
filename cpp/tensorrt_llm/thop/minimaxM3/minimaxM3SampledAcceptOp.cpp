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

#include "tensorrt_llm/kernels/minimaxM3/sampledAccept.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace sa = tensorrt_llm::kernels::minimaxM3::sampledAccept;

//! uint64 words of the sampled-accept exchange workspace for `world` ranks.
int64_t minimaxM3SampledAcceptWsWords(int64_t world)
{
    return sa::sampledAcceptWsWords(static_cast<int>(world));
}

//! Sampled (temperature + top-p) verification from this rank's BF16 logits shard [rows, Vs], rows = nreq *
//! (draft_len + 1) <= 48: the sampled token of every row (tokens), then the strict acceptance against draft (accTokens,
//! numAcc, gatherIds, optional nextNew column 0). Per-row Philox (seeds, offsets); the exchange goes through the
//! multicast workspace wsMcPtr / wsUc with the (finished rows, call) counters.
void minimaxM3SampledAccept(torch::Tensor const& logits, int64_t vocabOffset, torch::Tensor const& temperatures,
    torch::Tensor const& topPs, torch::Tensor const& seeds, torch::Tensor const& offsets, torch::Tensor& wsUc,
    int64_t wsMcPtr, torch::Tensor& counters, torch::Tensor& tokens, int64_t rank, int64_t world,
    torch::Tensor const& draft, torch::Tensor const& batchIdx, torch::Tensor& accTokens, torch::Tensor& numAcc,
    torch::Tensor& gatherIds, int64_t forceTotal, std::optional<torch::Tensor> const& nextNew,
    std::optional<torch::Tensor> const& forceVec, std::optional<torch::Tensor> const& extraAttempts, bool usePdl)
{
    c10::cuda::CUDAGuard const deviceGuard{logits.device()};
    sa::Args a;
    a.logits = reinterpret_cast<__nv_bfloat16 const*>(logits.data_ptr());
    a.ld = logits.stride(0);
    a.rows = static_cast<int>(logits.size(0));
    a.Vs = static_cast<int>(logits.size(1));
    TORCH_CHECK(a.Vs <= sa::kMaxVs && a.rows <= sa::kMaxRows && world <= sa::kMaxWorld
        && static_cast<long long>(vocabOffset) + a.Vs <= (1LL << sa::kIdBits));
    a.vocab_offset = static_cast<int>(vocabOffset);
    a.temperatures = temperatures.data_ptr<float>();
    a.top_ps = topPs.data_ptr<float>();
    a.seeds = reinterpret_cast<long long const*>(seeds.data_ptr<int64_t>());
    a.offsets = reinterpret_cast<long long const*>(offsets.data_ptr<int64_t>());
    a.ws_uc = reinterpret_cast<unsigned long long*>(wsUc.data_ptr());
    a.ws_mc = reinterpret_cast<unsigned long long*>(wsMcPtr);
    a.counters = reinterpret_cast<unsigned int*>(counters.data_ptr());
    a.tokens = tokens.data_ptr<int>();
    a.extra_attempts = extraAttempts.has_value() ? extraAttempts->data_ptr<int>() : nullptr;
    a.rank = static_cast<int>(rank);
    a.world = static_cast<int>(world);
    a.draft = draft.data_ptr<int>();
    a.batch_idx = batchIdx.data_ptr<int>();
    a.acc_tokens = accTokens.data_ptr<int>();
    a.num_acc = numAcc.data_ptr<int>();
    a.gather_ids = reinterpret_cast<long long*>(gatherIds.data_ptr<int64_t>());
    a.draft_len = static_cast<int>(draft.size(1));
    a.nreq = static_cast<int>(draft.size(0));
    a.force_total = static_cast<int>(forceTotal);
    a.next_new = nextNew.has_value() ? nextNew->data_ptr<int>() : nullptr;
    a.next_stride = nextNew.has_value() ? static_cast<int>(nextNew->stride(0)) : 0;
    a.force_vec = forceVec.has_value() ? forceVec->data_ptr<int>() : nullptr;
    cudaError_t const err = sa::invokeSampledAccept(a, usePdl, at::cuda::getCurrentCUDAStream(logits.get_device()));
    TORCH_CHECK(err == cudaSuccess, "m3_sampled_accept launch failed: ", cudaGetErrorString(err));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

// ws_mc_ptr is the multicast address of ws_uc's mapping: the op writes through it, which the schema cannot show.
TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    // No tensor arguments, so no dispatch key: registered with the schema as a catch-all kernel.
    m.def("minimax_m3_sampled_accept_ws_words(int world) -> int",
        &tensorrt_llm::torch_ext::minimaxM3SampledAcceptWsWords);
    m.def(
        "minimax_m3_sampled_accept(Tensor logits, int vocab_offset, Tensor temperatures, Tensor top_ps, "
        "Tensor seeds, Tensor offsets, Tensor(a!) ws_uc, int ws_mc_ptr, Tensor(b!) counters, Tensor(c!) tokens, "
        "int rank, int world, Tensor draft, Tensor batch_idx, Tensor(d!) acc_tokens, Tensor(e!) num_acc, "
        "Tensor(f!) gather_ids, int force_total, Tensor(g!)? next_new, Tensor? force_vec, "
        "Tensor(h!)? extra_attempts, bool use_pdl) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_sampled_accept", &tensorrt_llm::torch_ext::minimaxM3SampledAccept);
}
