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

#include <limits>
#include <optional>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace lha = tensorrt_llm::kernels::lm_head_argmax;

namespace
{

bool isAligned16(torch::Tensor const& t)
{
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0 && t.stride(0) * t.element_size() % 16 == 0;
}

} // namespace

//! Rows one lm_head_argmax call takes for hidden size k on the current device (0: unsupported).
int64_t lmHeadArgmaxMaxRows(int64_t k)
{
    return lha::maxRows(static_cast<int>(k));
}

//! Greedy tokens (int32 [m]) of ``input @ weight.T`` over the tensor-parallel group's vocabulary, weight being this
//! rank's shard starting at global row vocab_offset. scratch: this rank's zero-initialized int64 scratch of
//! lha::kScratchWords words. With several ranks, the exchange runs on a TPKeyExchange channel: exchange_epoch is the
//! channel's int32 epoch counter, peer_buffers every rank's receive region of the channel (by rank) and
//! exchange_words the packets per rank and slot; peer_buffers empty (or of one entry) means a single rank.
void lmHeadArgmax(torch::Tensor const& input, torch::Tensor const& weight, int64_t vocabOffset, bool roundLogits,
    torch::Tensor& scratch, std::optional<torch::Tensor> const& exchangeEpoch, std::vector<int64_t> const& peerBuffers,
    int64_t exchangeWords, int64_t rank, torch::Tensor& tokens)
{
    TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "input and weight must be 2D");
    TORCH_CHECK((input.scalar_type() == torch::kBFloat16 || input.scalar_type() == torch::kHalf)
            && weight.scalar_type() == input.scalar_type(),
        "input and weight must both be bf16 or both fp16");
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && scratch.is_cuda() && tokens.is_cuda(), "CUDA tensors expected");
    TORCH_CHECK(input.stride(1) == 1 && weight.stride(1) == 1, "input and weight need a unit inner stride");
    TORCH_CHECK(isAligned16(input) && isAligned16(weight), "input and weight rows must be 16-byte aligned");
    TORCH_CHECK(input.size(1) == weight.size(1), "input and weight disagree on the hidden size");
    TORCH_CHECK(
        scratch.scalar_type() == torch::kInt64 && scratch.is_contiguous() && scratch.numel() >= lha::kScratchWords,
        "scratch must be a contiguous int64 tensor of at least ", lha::kScratchWords, " elements");
    int const m = static_cast<int>(input.size(0));
    int const n = static_cast<int>(weight.size(0));
    int const k = static_cast<int>(input.size(1));
    TORCH_CHECK(m >= 1 && m <= lha::maxRows(k) && n >= 1, "lm_head_argmax: unsupported shape m=", m, " n=", n, " k=", k,
        " (SM90+, k a multiple of ", lha::kHiddenGranularity, ", m <= lm_head_argmax_max_rows(k))");
    TORCH_CHECK(
        vocabOffset >= 0 && vocabOffset + n <= std::numeric_limits<int32_t>::max(), "vocab_offset out of range");
    TORCH_CHECK(tokens.scalar_type() == torch::kInt32 && tokens.is_contiguous() && tokens.numel() == m,
        "tokens must be a contiguous int32 tensor of ", m, " elements");
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
    params.fp16 = input.scalar_type() == torch::kHalf;
    params.roundLogits = roundLogits;
    params.tokens = tokens.data_ptr<int32_t>();
    params.scratch = reinterpret_cast<uint64_t*>(scratch.data_ptr<int64_t>());
    if (ranks > 1)
    {
        TORCH_CHECK(exchangeEpoch.has_value() && exchangeEpoch->is_cuda()
                && exchangeEpoch->scalar_type() == torch::kInt32 && exchangeEpoch->numel() >= 1,
            "several ranks need the exchange channel's int32 epoch counter");
        TORCH_CHECK(exchangeWords >= 2 * m, "exchange_words must hold two packets per row");
        params.exchangeEpoch = exchangeEpoch->data_ptr<int32_t>();
        params.exchangeWords = static_cast<int>(exchangeWords);
        for (int r = 0; r < ranks; ++r)
        {
            params.peers[r] = reinterpret_cast<int64_t*>(peerBuffers[r]);
        }
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
    m.def("lm_head_argmax_max_rows(int k) -> int");
    m.def(
        "lm_head_argmax(Tensor input, Tensor weight, int vocab_offset, bool round_logits, Tensor(a!) scratch, "
        "Tensor(b!)? exchange_epoch, int[] peer_buffers, int exchange_words, int rank, Tensor(c!) tokens) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("lm_head_argmax_max_rows", &tensorrt_llm::torch_ext::lmHeadArgmaxMaxRows);
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("lm_head_argmax", &tensorrt_llm::torch_ext::lmHeadArgmax);
}
