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
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/lmHeadArgmax/lmHeadArgmax.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace lha = tensorrt_llm::kernels::lm_head_argmax;

//! Greedy tokens (into tokens[i * tokens_stride]) of ``input @ weight.T`` over the tensor-parallel group's vocabulary,
//! weight being this rank's shard starting at global row vocab_offset. peer_buffers: every rank's exchange buffer
//! address, by rank (empty or one entry: a single rank); scratch: this rank's zero-initialized scratch.
void lmHeadArgmax(torch::Tensor const& input, torch::Tensor const& weight, int64_t vocabOffset, torch::Tensor& scratch,
    std::vector<int64_t> const& peerBuffers, int64_t rank, torch::Tensor& tokens, int64_t tokensStride)
{
    auto const smVersion = tensorrt_llm::common::getSMVersion();
    TORCH_CHECK(smVersion >= 90, "lm_head_argmax needs SM90 or newer");
    TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "input and weight must be 2D");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && weight.scalar_type() == torch::kBFloat16,
        "input and weight must be bf16");
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && scratch.is_cuda() && tokens.is_cuda(), "CUDA tensors expected");
    TORCH_CHECK(input.stride(1) == 1 && weight.stride(1) == 1, "input and weight need a unit inner stride");
    TORCH_CHECK(input.size(1) == weight.size(1), "input and weight disagree on the hidden size");
    TORCH_CHECK(
        scratch.scalar_type() == torch::kInt64 && scratch.is_contiguous() && scratch.numel() >= lha::kScratchWords,
        "scratch must be a contiguous int64 tensor of at least ", lha::kScratchWords, " elements");
    TORCH_CHECK(tokens.scalar_type() == torch::kInt32, "tokens must be int32");
    int const m = static_cast<int>(input.size(0));
    int const n = static_cast<int>(weight.size(0));
    int const k = static_cast<int>(input.size(1));
    TORCH_CHECK(lha::supports(m, n, k), "lm_head_argmax: unsupported shape m=", m, " n=", n, " k=", k);
    TORCH_CHECK(tokensStride >= 1 && tokens.numel() >= (m - 1) * tokensStride + 1, "tokens too small");
    int const ranks = std::max<int>(1, static_cast<int>(peerBuffers.size()));
    TORCH_CHECK(ranks <= lha::kMaxRanks && rank >= 0 && rank < ranks, "rank out of range");

    lha::Params params;
    params.input = input.data_ptr();
    params.inputStride = input.stride(0);
    params.weight = weight.data_ptr();
    params.weightStride = weight.stride(0);
    params.m = m;
    params.n = n;
    params.k = k;
    params.vocabOffset = static_cast<int>(vocabOffset);
    params.tokens = tokens.data_ptr<int32_t>();
    params.tokensStride = static_cast<int>(tokensStride);
    params.scratch = reinterpret_cast<uint64_t*>(scratch.data_ptr<int64_t>());
    for (int r = 0; r < static_cast<int>(peerBuffers.size()); ++r)
    {
        params.peers[r] = reinterpret_cast<uint64_t*>(peerBuffers[r]);
    }
    params.rank = static_cast<int>(rank);
    params.ranks = ranks;
    auto const device = input.get_device();
    int const ctas = at::cuda::getDeviceProperties(device)->multiProcessorCount;
    lha::launch(params, ctas, at::cuda::getCurrentCUDAStream(device), tensorrt_llm::common::getEnvEnablePDL());
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "lm_head_argmax(Tensor input, Tensor weight, int vocab_offset, Tensor(a!) scratch, int[] peer_buffers, "
        "int rank, Tensor(b!) tokens, int tokens_stride=1) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("lm_head_argmax", &tensorrt_llm::torch_ext::lmHeadArgmax);
}
