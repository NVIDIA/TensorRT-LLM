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

#include "minimax_m3/csrc/m31/markovSamplingSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace mk = TRTLLM_MV2_NS::kernels::minimaxM31;

//! Any vocab >= 1 in ceil(vocab / 2048) chunks (bit for bit the former Triton sampler on vocab % 16 == 0 with 65..128
//! chunks, the served class; every other width runs here too).
void checkChunks(int64_t vocab, int64_t numChunks, char const* op)
{
    TORCH_CHECK(vocab >= 1 && vocab < (int64_t{1} << 31) && numChunks == (vocab + 2047) / 2048 && numChunks <= 65535,
        op, ": vocab in [1, 2^31) in ceil(vocab / 2048) chunks");
}

bool isWeightDtype(at::ScalarType t)
{
    return t == at::kBFloat16 || t == at::kFloat;
}

} // namespace

void minimaxM31MarkovPick(torch::Tensor const& keys, torch::Tensor const& indices, torch::Tensor& tokens,
    torch::Tensor const& anchors, int64_t pos, int64_t numPos, torch::Tensor const& w1, torch::Tensor& emb,
    int64_t vocab, int64_t numChunks, bool fromDraw, bool pdl)
{
    checkChunks(vocab, numChunks, "minimax_m31_markov_pick");
    TORCH_CHECK(anchors.scalar_type() == at::kLong || anchors.scalar_type() == at::kInt,
        "minimax_m31_markov_pick: int32 / int64 anchors");
    TORCH_CHECK(isWeightDtype(w1.scalar_type()) && emb.scalar_type() == w1.scalar_type() && w1.dim() == 2
            && w1.stride(1) == 1 && emb.is_contiguous() && emb.size(1) == w1.size(1),
        "minimax_m31_markov_pick: bf16 / fp32 w1 [V, rank] (unit inner stride) and a contiguous emb [gens, rank]");
    TORCH_CHECK(keys.scalar_type() == at::kFloat && indices.scalar_type() == at::kInt
            && tokens.scalar_type() == at::kInt && anchors.is_contiguous() && tokens.is_contiguous(),
        "minimax_m31_markov_pick: fp32 keys, int32 indices / tokens");
    mk::MarkovPickArgs a{};
    a.keys = keys.data_ptr<float>();
    a.indices = indices.data_ptr<int32_t>();
    a.tokens = tokens.data_ptr<int32_t>();
    a.anchors = anchors.data_ptr();
    a.anchors64 = anchors.scalar_type() == at::kLong;
    a.pos = pos;
    a.numPos = numPos;
    a.w1 = w1.data_ptr();
    a.w1Bf16 = w1.scalar_type() == at::kBFloat16;
    a.w1Stride = w1.stride(0);
    a.emb = emb.data_ptr();
    a.rank = w1.size(1);
    a.vocab = vocab;
    a.numChunks = numChunks;
    a.numGens = emb.size(0);
    a.fromDraw = fromDraw;
    a.pdl = pdl;
    at::cuda::CUDAGuard const guard(emb.device());
    cudaError_t const e = mk::invokeMarkovPick(a, at::cuda::getCurrentCUDAStream(emb.get_device()));
    TORCH_CHECK(e == cudaSuccess, "minimax_m31_markov_pick launch: ", cudaGetErrorString(e));
}

void minimaxM31MarkovDraw(torch::Tensor const& logits, torch::Tensor& biased, torch::Tensor const& bias,
    torch::Tensor const& temps, torch::Tensor const& seeds, torch::Tensor const& offsets, torch::Tensor const& slots,
    torch::Tensor& keys, torch::Tensor& indices, torch::Tensor& runMax, torch::Tensor& runSum, int64_t pos,
    int64_t numPos, int64_t offsetStep, int64_t paddingSlot, int64_t vocab, int64_t numChunks,
    std::optional<torch::Tensor> arrivals, std::optional<torch::Tensor> tokens, std::optional<torch::Tensor> w1,
    std::optional<torch::Tensor> emb, bool pdl)
{
    checkChunks(vocab, numChunks, "minimax_m31_markov_draw");
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16 || logits.scalar_type() == at::kFloat,
        "minimax_m31_markov_draw: bf16 / fp32 logits");
    TORCH_CHECK(logits.is_contiguous() && bias.is_contiguous() && biased.is_contiguous()
            && biased.scalar_type() == at::kFloat && isWeightDtype(bias.scalar_type()),
        "minimax_m31_markov_draw: contiguous logits / bias / fp32 biased");
    TORCH_CHECK(
        seeds.scalar_type() == at::kLong && offsets.scalar_type() == at::kLong && slots.scalar_type() == at::kLong,
        "minimax_m31_markov_draw: int64 seeds / offsets / slots");
    TORCH_CHECK(temps.scalar_type() == at::kFloat && keys.scalar_type() == at::kFloat
            && indices.scalar_type() == at::kInt && runMax.scalar_type() == at::kFloat
            && runSum.scalar_type() == at::kFloat,
        "minimax_m31_markov_draw: fp32 temps / keys / statistics, int32 indices");
    bool const fused = arrivals.has_value();
    TORCH_CHECK(fused == tokens.has_value() && fused == w1.has_value() && fused == emb.has_value(),
        "minimax_m31_markov_draw: the fused pick needs arrivals, tokens, w1 and emb together");
    TORCH_CHECK(!fused
            || (arrivals->scalar_type() == at::kInt && tokens->scalar_type() == at::kInt
                && w1->scalar_type() == bias.scalar_type() && emb->scalar_type() == bias.scalar_type()
                && w1->stride(1) == 1 && emb->is_contiguous()),
        "minimax_m31_markov_draw: int32 arrivals / tokens; w1 / emb of the bias dtype");
    mk::MarkovDrawArgs a{};
    a.logits = logits.data_ptr();
    a.logitsBf16 = logits.scalar_type() == at::kBFloat16;
    a.biased = biased.data_ptr<float>();
    a.bias = bias.data_ptr();
    a.biasBf16 = bias.scalar_type() == at::kBFloat16;
    a.temps = temps.data_ptr<float>();
    a.seeds = seeds.data_ptr<int64_t>();
    a.offsets = offsets.data_ptr<int64_t>();
    a.slots = slots.data_ptr<int64_t>();
    a.keys = keys.data_ptr<float>();
    a.indices = indices.data_ptr<int32_t>();
    a.runMax = runMax.data_ptr<float>();
    a.runSum = runSum.data_ptr<float>();
    a.pos = pos;
    a.numPos = numPos;
    a.offsetStep = offsetStep;
    a.paddingSlot = paddingSlot;
    a.vocab = vocab;
    a.numChunks = numChunks;
    a.numGens = logits.size(0);
    if (fused)
    {
        a.arrivals = arrivals->data_ptr<int32_t>();
        a.tokens = tokens->data_ptr<int32_t>();
        a.w1 = w1->data_ptr();
        a.w1Stride = w1->stride(0);
        a.emb = emb->data_ptr();
        a.rank = w1->size(1);
    }
    a.pdl = pdl;
    at::cuda::CUDAGuard const guard(logits.device());
    cudaError_t const e = mk::invokeMarkovDraw(a, at::cuda::getCurrentCUDAStream(logits.get_device()));
    TORCH_CHECK(e == cudaSuccess, "minimax_m31_markov_draw launch: ", cudaGetErrorString(e));
}

void minimaxM31MarkovProposal(torch::Tensor const& biased, torch::Tensor const& temps, torch::Tensor const& runMax,
    torch::Tensor const& runSum, torch::Tensor const& slots, torch::Tensor& probs, int64_t vocab, int64_t numChunks,
    int64_t numPos, bool pdl)
{
    checkChunks(vocab, numChunks, "minimax_m31_markov_proposal");
    TORCH_CHECK(probs.scalar_type() == at::kFloat && probs.dim() == 3 && probs.stride(2) == 1 && probs.size(1) >= numPos
            && probs.size(2) >= vocab,
        "minimax_m31_markov_proposal: draft_probs must be fp32 [slots, >= K, >= vocab] with a unit last stride");
    TORCH_CHECK(biased.is_contiguous() && biased.scalar_type() == at::kFloat && slots.scalar_type() == at::kLong,
        "minimax_m31_markov_proposal: contiguous fp32 biased, int64 slots");
    mk::MarkovProposalArgs a{};
    a.biased = biased.data_ptr<float>();
    a.temps = temps.data_ptr<float>();
    a.runMax = runMax.data_ptr<float>();
    a.runSum = runSum.data_ptr<float>();
    a.slots = slots.data_ptr<int64_t>();
    a.probs = probs.data_ptr<float>();
    a.probsSlotStride = probs.stride(0);
    a.probsPosStride = probs.stride(1);
    a.vocab = vocab;
    a.numChunks = numChunks;
    a.numPos = numPos;
    a.numGens = biased.size(0);
    a.pdl = pdl;
    at::cuda::CUDAGuard const guard(probs.device());
    cudaError_t const e = mk::invokeMarkovProposal(a, at::cuda::getCurrentCUDAStream(probs.get_device()));
    TORCH_CHECK(e == cudaSuccess, "minimax_m31_markov_proposal launch: ", cudaGetErrorString(e));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m31_markov_pick(Tensor keys, Tensor indices, Tensor(a!) tokens, Tensor anchors, int pos, int num_pos, "
        "Tensor w1, Tensor(b!) emb, int vocab, int num_chunks, bool from_draw, bool pdl=False) -> ()");
    m.def(
        "minimax_m31_markov_draw(Tensor logits, Tensor(a!) biased, Tensor bias, Tensor temps, Tensor seeds, Tensor "
        "offsets, Tensor slots, Tensor(b!) keys, Tensor(c!) indices, Tensor(d!) run_max, Tensor(e!) run_sum, int pos, "
        "int num_pos, int offset_step, int padding_slot, int vocab, int num_chunks, Tensor(f!)? arrivals=None, "
        "Tensor(g!)? tokens=None, Tensor? w1=None, Tensor(h!)? emb=None, bool pdl=False) -> ()");
    m.def(
        "minimax_m31_markov_proposal(Tensor biased, Tensor temps, Tensor run_max, Tensor run_sum, Tensor slots, "
        "Tensor(a!) probs, int vocab, int num_chunks, int num_pos, bool pdl=False) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_markov_pick", &TRTLLM_MV2_NS::torch_ext::minimaxM31MarkovPick);
    m.impl("minimax_m31_markov_draw", &TRTLLM_MV2_NS::torch_ext::minimaxM31MarkovDraw);
    m.impl("minimax_m31_markov_proposal", &TRTLLM_MV2_NS::torch_ext::minimaxM31MarkovProposal);
}
