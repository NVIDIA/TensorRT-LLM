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
#include "tensorrt_llm/kernels/minimaxM3/draftStep.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <torch/extension.h>

#include <initializer_list>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace ds = tensorrt_llm::kernels::minimaxM3::draftStep;

using OptTensor = std::optional<torch::Tensor>;

void checkArch(torch::Tensor const& t)
{
    TORCH_CHECK(t.is_cuda(), "minimax_m3_draft_step expects CUDA tensors");
    static int const sSmVersion = tensorrt_llm::common::getSMVersion();
    TORCH_CHECK(sSmVersion >= 100, "minimax_m3_draft_step requires SM100 or newer");
}

void check(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TORCH_CHECK(false, cudaGetErrorString(err));
    }
}

void checkInt(torch::Tensor const& t, int64_t n, char const* name)
{
    TORCH_CHECK(t.scalar_type() == at::kInt && t.is_contiguous() && t.numel() >= n, "draft_step: ", name,
        " must be contiguous int32 with at least ", n, " elements");
}

} // namespace

//! One fused greedy draft pass over `rows` <= 8 requests: the BF16 head (this rank's fragment-ordered shard wf
//! [N / 16, K / 16, 256], or with excOff / excPos / excW / expBase its 12-bit coding, wf uint8 [N * K * 3 / 2];
//! global ids vocabOffset..) over x[xRows[b]] (or x[b]), the global argmax across the TP ranks through the multicast
//! slots wsMcPtr / wsUc, then draftOut[b, step] = idsNext[b] = token, posNext[b] = posSrc[xRows[b] or b] + 1 and the
//! KV-length bookkeeping (see draftStep.h).
void minimaxM3DraftStep(torch::Tensor const& x, OptTensor const& xRows, int64_t rows, torch::Tensor const& wf,
    int64_t vocabOffset, torch::Tensor& keys, torch::Tensor& counters, torch::Tensor& wsUc, int64_t wsMcPtr,
    int64_t rank, int64_t world, torch::Tensor& draftOut, int64_t step, torch::Tensor& idsNext,
    torch::Tensor const& posSrc, torch::Tensor& posNext, torch::Tensor& kvLens, OptTensor const& numAcc,
    OptTensor const& seqLens, int64_t draftLen, int64_t numCtas, bool pdl, OptTensor const& outKeys,
    OptTensor const& excOff, OptTensor const& excPos, OptTensor const& excW, int64_t expBase)
{
    checkArch(x);
    for (torch::Tensor const* t : std::initializer_list<torch::Tensor const*>{
             &wf, &keys, &counters, &wsUc, &draftOut, &idsNext, &posSrc, &posNext, &kvLens})
    {
        TORCH_CHECK(t->device() == x.device(), "draft_step: every tensor must be on x's device");
    }
    for (OptTensor const* t :
        std::initializer_list<OptTensor const*>{&xRows, &numAcc, &seqLens, &outKeys, &excOff, &excPos, &excW})
    {
        TORCH_CHECK(!t->has_value() || (*t)->device() == x.device(), "draft_step: every tensor must be on x's device");
    }
    c10::cuda::CUDAGuard const deviceGuard{x.device()};
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(),
        "draft_step: x must be contiguous BF16 [rows, K]");
    int const K = static_cast<int>(x.size(1));
    TORCH_CHECK(K % 16 == 0, "draft_step: K must be a multiple of 16");
    TORCH_CHECK(rows >= 1 && rows <= ds::kMaxTokens, "draft_step: 1..", ds::kMaxTokens, " rows");
    bool const coded = excOff.has_value();
    TORCH_CHECK(coded == excPos.has_value() && coded == excW.has_value(),
        "draft_step: exc_off, exc_pos and exc_w come together");
    int nTiles = 0;
    if (coded)
    {
        TORCH_CHECK(K % 64 == 0 && K / 16 <= 0xFFFF, "draft_step: a coded head needs K a multiple of 64");
        TORCH_CHECK(excOff->scalar_type() == at::kInt && excOff->is_contiguous() && excOff->numel() >= 2,
            "draft_step: exc_off must be int32 [tiles + 1]");
        nTiles = static_cast<int>(excOff->numel() - 1);
        TORCH_CHECK(wf.scalar_type() == at::kByte && wf.is_contiguous()
                && wf.numel() == static_cast<int64_t>(nTiles) * 16 * K * 3 / 2,
            "draft_step: a coded wf must be the contiguous uint8 stream of [tiles * 16, K]");
        TORCH_CHECK(excPos->scalar_type() == at::kInt && excPos->is_contiguous(), "draft_step: exc_pos must be int32");
        TORCH_CHECK(excW->scalar_type() == at::kFloat && excW->is_contiguous() && excW->numel() == 2 * excPos->numel(),
            "draft_step: exc_w must be fp32 [E, 2]");
        TORCH_CHECK(expBase >= 0 && expBase <= 240, "draft_step: exponent base out of range");
    }
    else
    {
        TORCH_CHECK(wf.scalar_type() == at::kBFloat16 && wf.dim() == 3 && wf.is_contiguous() && wf.size(1) == K / 16
                && wf.size(2) == 256,
            "draft_step: wf must be the fragment-ordered BF16 weight [N / 16, K / 16, 256]");
        nTiles = static_cast<int>(wf.size(0));
    }
    TORCH_CHECK(nTiles >= 1 && numCtas >= 1 && (nTiles + numCtas - 1) / numCtas <= ds::kMaxTilesPerCta,
        "draft_step: too many weight tiles per CTA");
    TORCH_CHECK(vocabOffset >= 0 && vocabOffset + 16LL * nTiles <= 0xFFFFFFFFLL, "draft_step: vocab offset");
    if (xRows.has_value())
    {
        TORCH_CHECK(xRows->scalar_type() == at::kLong && xRows->is_contiguous() && xRows->numel() >= rows,
            "draft_step: x_rows must be contiguous int64 [rows]");
    }
    else
    {
        TORCH_CHECK(x.size(0) >= rows, "draft_step: x must hold a row per request");
    }
    TORCH_CHECK(world >= 1 && rank >= 0 && rank < world, "draft_step: rank / world");
    TORCH_CHECK(keys.scalar_type() == at::kLong && keys.numel() >= rows, "draft_step: keys must be int64 [>= rows]");
    TORCH_CHECK(counters.scalar_type() == at::kInt && counters.numel() >= 2, "draft_step: counters must be int32 [2]");
    TORCH_CHECK(wsUc.scalar_type() == at::kLong && wsUc.numel() >= 2 * ds::kMaxTokens * world,
        "draft_step: ws_uc must hold [2][8][world] slots");
    TORCH_CHECK(draftOut.scalar_type() == at::kInt && draftOut.dim() == 2 && draftOut.stride(1) == 1
            && draftOut.size(0) >= rows && step >= 0 && step < draftOut.size(1),
        "draft_step: draft_out must be int32 [rows, > step] with unit column stride");
    checkInt(idsNext, rows, "ids_next");
    checkInt(posNext, rows, "pos_next");
    checkInt(kvLens, rows, "kv_lens");
    TORCH_CHECK(posSrc.scalar_type() == at::kInt && posSrc.is_contiguous(), "draft_step: pos_src must be int32");
    TORCH_CHECK(numAcc.has_value() == seqLens.has_value(), "draft_step: num_acc and seq_lens come together");
    TORCH_CHECK(!numAcc.has_value() || step == 0, "draft_step: the KV rollback (num_acc) is pass 0's");
    if (numAcc.has_value())
    {
        checkInt(*numAcc, rows, "num_acc");
        checkInt(*seqLens, rows, "seq_lens");
    }
    if (outKeys.has_value())
    {
        TORCH_CHECK(outKeys->scalar_type() == at::kLong && outKeys->is_contiguous() && outKeys->numel() >= rows,
            "draft_step: out_keys must be contiguous int64 [rows]");
    }
    ds::Params p{};
    p.x = reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr());
    p.x_rows = xRows.has_value() ? reinterpret_cast<long long const*>(xRows->data_ptr()) : nullptr;
    p.M = static_cast<int>(rows);
    p.wf = reinterpret_cast<uint4 const*>(wf.data_ptr());
    p.n_tiles = nTiles;
    p.vocab_offset = static_cast<int>(vocabOffset);
    p.keys = reinterpret_cast<unsigned long long*>(keys.data_ptr());
    p.counters = reinterpret_cast<unsigned int*>(counters.data_ptr());
    p.ws_uc = reinterpret_cast<unsigned long long*>(wsUc.data_ptr());
    p.ws_mc = reinterpret_cast<unsigned long long*>(wsMcPtr);
    p.rank = static_cast<int>(rank);
    p.world = static_cast<int>(world);
    p.out_keys = outKeys.has_value() ? reinterpret_cast<unsigned long long*>(outKeys->data_ptr()) : nullptr;
    p.st = ds::StepArgs{draftOut.data_ptr<int>(), static_cast<int>(draftOut.stride(0)), static_cast<int>(step),
        idsNext.data_ptr<int>(), posSrc.data_ptr<int>(), posNext.data_ptr<int>(), kvLens.data_ptr<int>(),
        numAcc.has_value() ? numAcc->data_ptr<int>() : nullptr,
        seqLens.has_value() ? seqLens->data_ptr<int>() : nullptr, static_cast<int>(draftLen)};
    p.cd = ds::CodedArgs{coded ? excOff->data_ptr<int>() : nullptr,
        coded ? reinterpret_cast<uint32_t const*>(excPos->data_ptr<int>()) : nullptr,
        coded ? reinterpret_cast<float2 const*>(excW->data_ptr<float>()) : nullptr,
        static_cast<uint32_t>((expBase << 7) | (expBase << 23))};
    check(ds::invokeDraftStep(p, K, static_cast<int>(numCtas), pdl, at::cuda::getCurrentCUDAStream(x.get_device())));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

// Mutations through the raw multicast address ws_mc_ptr are not visible to the schema: that buffer is owned by the
// Python workspace and never aliased by the op's tensor arguments.
TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m3_draft_step(Tensor x, Tensor? x_rows, int rows, Tensor wf, int vocab_offset, Tensor(a!) keys, "
        "Tensor(b!) counters, Tensor(c!) ws_uc, int ws_mc_ptr, int rank, int world, Tensor(d!) draft_out, int step, "
        "Tensor(e!) ids_next, Tensor pos_src, Tensor(f!) pos_next, Tensor(g!) kv_lens, Tensor? num_acc, "
        "Tensor(h!)? seq_lens, int draft_len, int num_ctas, bool pdl, Tensor(i!)? out_keys, Tensor? exc_off=None, "
        "Tensor? exc_pos=None, Tensor? exc_w=None, int exp_base=0) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_draft_step", &tensorrt_llm::torch_ext::minimaxM3DraftStep);
}
