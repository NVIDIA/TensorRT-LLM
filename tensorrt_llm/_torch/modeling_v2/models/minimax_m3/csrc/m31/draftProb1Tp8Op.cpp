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

#include "minimax_m3/csrc/m31/draftProb1Tp8.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <initializer_list>
#include <vector>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace dp = TRTLLM_MV2_NS::kernels::minimaxM3Sm107Tp8::draftProb1;

int64_t minimaxM3Sm107Tp8DraftProb1MaxGrid(int64_t rows)
{
    TORCH_CHECK(rows >= 1 && rows <= dp::kMaxRows, "draft_prob1_max_grid: 1 <= rows <= 4");
    return dp::maxGrid(static_cast<int>(rows));
}

//! One draft = prep -> the peer's dk_mk_prob1 at exchange world 8 (all k steps, one launch) -> the
//! cross-rank fault decision -> the draft_probs adapter. The checks are the peer steps()'
//! (dspark_markov_prob1_w8_ext.cu), plus the adapter's. ws = this rank's {part, part_ms, counter, epoch, exchange
//! slots, err, fault words} device pointers (MNNVL carve for the slots and fault words, which the peers write); peers /
//! fault_peers = every rank's exchange slots / fault words in rank order; caches = every rank's draft_logits cache in
//! rank order (the op hands the peer kernel its own first, then the others: the mirror).
void minimaxM3Sm107Tp8DraftProb1(torch::Tensor const& base, torch::Tensor const& w1, torch::Tensor const& w2s,
    int64_t voff, int64_t k, torch::Tensor const& prevIn, torch::Tensor& prev0, torch::Tensor& prev1,
    torch::Tensor& tokOut, std::vector<int64_t> const& ws, std::vector<int64_t> const& peers,
    std::vector<int64_t> const& caches, int64_t rank, int64_t world, int64_t grid, bool pdl,
    torch::Tensor const& slotIds, int64_t dummySlotRow, torch::Tensor const& offsets, int64_t rngBase, int64_t span,
    torch::Tensor& idxMap, torch::Tensor& samplePos, torch::Tensor const& temperature, torch::Tensor const& seeds,
    torch::Tensor const& cache, torch::Tensor& fault, torch::Tensor& faultCount, torch::Tensor& seq,
    std::vector<int64_t> const& faultPeers, int64_t capNs, torch::Tensor& stats, torch::Tensor& draftProbs,
    torch::Tensor& tokens32, int64_t vocab)
{
    TORCH_CHECK(ws.size() == 7, "draft_prob1: ws = {part, part_ms, counter, epoch, slots, err, fault words}");
    for (int64_t p : ws)
    {
        TORCH_CHECK(p != 0, "draft_prob1: workspace pointer");
    }
    TORCH_CHECK(world >= 2 && world <= dp::kMaxWorld && (world & (world - 1)) == 0 && rank >= 0 && rank < world,
        "draft_prob1: world 2 / 4 / 8");
    TORCH_CHECK(static_cast<int64_t>(peers.size()) == world && static_cast<int64_t>(faultPeers.size()) == world,
        "draft_prob1: one exchange slot / fault word base per rank");
    TORCH_CHECK(static_cast<int64_t>(caches.size()) == world, "draft_prob1: mirror needs every rank's cache");
    TORCH_CHECK(k >= 1 && k <= dp::kMaxK, "draft_prob1: 1 <= k <= 16");
    TORCH_CHECK(base.is_cuda() && base.scalar_type() == torch::kBFloat16 && base.dim() == 2 && base.stride(1) == 1,
        "draft_prob1: base bf16 [C*k][ld]");
    int64_t const rows = base.size(0) / k;
    TORCH_CHECK(rows * k == base.size(0) && rows >= 1 && rows <= dp::kMaxRows, "draft_prob1: 1 <= C <= 4");
    TORCH_CHECK(w1.scalar_type() == torch::kBFloat16 && w1.is_contiguous() && w1.dim() == 2 && w1.size(1) == dp::kRank,
        "draft_prob1: w1 bf16 [V][256]");
    TORCH_CHECK(w2s.scalar_type() == torch::kBFloat16 && w2s.is_contiguous() && w2s.dim() == 2
            && w2s.size(1) == dp::kRank && w2s.size(0) <= base.size(1),
        "draft_prob1: w2s bf16 [Vs][256], Vs <= ld");
    TORCH_CHECK(prevIn.scalar_type() == torch::kInt64 && prev0.scalar_type() == torch::kInt64
            && prev1.scalar_type() == torch::kInt64 && prevIn.numel() >= rows && prev0.numel() >= rows
            && prev1.numel() >= rows,
        "draft_prob1: prev int64 [>= C]");
    TORCH_CHECK(tokOut.scalar_type() == torch::kInt64 && tokOut.dim() == 2 && tokOut.stride(1) == 1
            && tokOut.size(0) >= rows && tokOut.size(1) >= k,
        "draft_prob1: tok_out int64 [>= C][>= k]");
    int64_t const mg = dp::maxGrid(static_cast<int>(rows));
    TORCH_CHECK(grid >= 1 && grid <= mg, "draft_prob1: grid ", grid, " not co-resident (max ", mg, ")");
    TORCH_CHECK(slotIds.scalar_type() == torch::kInt64 && slotIds.is_contiguous() && slotIds.numel() >= rows,
        "draft_prob1: slot_ids int64 [>= C]");
    TORCH_CHECK(offsets.scalar_type() == torch::kInt64 && offsets.is_contiguous() && offsets.numel() >= rows,
        "draft_prob1: offsets int64 [>= C]");
    TORCH_CHECK(idxMap.scalar_type() == torch::kInt32 && idxMap.is_contiguous() && idxMap.numel() >= rows * k,
        "draft_prob1: idx_map int32 [>= C*k]");
    TORCH_CHECK(samplePos.scalar_type() == torch::kInt64 && samplePos.is_contiguous() && samplePos.numel() >= rows * k,
        "draft_prob1: sample_pos int64 [>= C*k]");
    TORCH_CHECK(
        temperature.scalar_type() == torch::kFloat32 && temperature.is_contiguous() && temperature.numel() >= rows,
        "draft_prob1: temperature fp32 [>= C]");
    TORCH_CHECK(seeds.scalar_type() == torch::kInt64 && seeds.is_contiguous() && seeds.numel() >= rows,
        "draft_prob1: seeds int64 [>= C]");
    TORCH_CHECK(cache.scalar_type() == torch::kBFloat16 && cache.dim() == 3 && cache.stride(2) == 1
            && cache.size(0) >= rows && cache.size(1) >= k && cache.size(2) >= vocab
            && cache.size(2) >= voff + w2s.size(0),
        "draft_prob1: draft_logits bf16 [>= C][>= k][>= V]");
    TORCH_CHECK(reinterpret_cast<int64_t>(cache.data_ptr()) == caches[rank], "draft_prob1: caches[rank] is cache");
    TORCH_CHECK(fault.scalar_type() == torch::kInt32 && fault.numel() >= 1, "draft_prob1: fault int32 [1]");
    TORCH_CHECK(faultCount.scalar_type() == torch::kInt64 && faultCount.numel() >= 1, "draft_prob1: fault_count int64");
    TORCH_CHECK(seq.scalar_type() == torch::kInt64 && seq.numel() >= 1, "draft_prob1: seq int64 [1]");
    TORCH_CHECK(capNs > 0, "draft_prob1: cap_ns");
    TORCH_CHECK(stats.scalar_type() == torch::kFloat32 && stats.is_contiguous()
            && stats.numel() >= rows * k * dp::kStatSplit * 2,
        "draft_prob1: stats fp32 [>= C*k*8*2]");
    TORCH_CHECK(draftProbs.scalar_type() == torch::kFloat32 && draftProbs.dim() == 3 && draftProbs.stride(2) == 1
            && draftProbs.size(1) >= k && draftProbs.size(2) >= vocab,
        "draft_prob1: draft_probs fp32 [slots][>= k][>= V]");
    TORCH_CHECK(tokens32.scalar_type() == torch::kInt32 && tokens32.is_contiguous() && tokens32.numel() >= rows * k,
        "draft_prob1: tokens32 int32 [>= C*k]");
    TORCH_CHECK(vocab >= 1 && vocab <= (1 << 18), "draft_prob1: vocab (the exchange packs ids < 2^18)");
    // Every tensor operand is dereferenced on base's device: a CPU or other-device tensor would fault the context.
    for (torch::Tensor const* t : std::initializer_list<torch::Tensor const*>{&w1, &w2s, &prevIn, &prev0, &prev1,
             &tokOut, &slotIds, &offsets, &idxMap, &samplePos, &temperature, &seeds, &cache, &fault, &faultCount, &seq,
             &stats, &draftProbs, &tokens32})
    {
        TORCH_CHECK(t->is_cuda() && t->device() == base.device(),
            "draft_prob1: every tensor operand must be on base's "
            "device (",
            base.device(), "), got ", t->device());
    }

    at::cuda::CUDAGuard guard(base.device());
    cudaStream_t const stream = at::cuda::getCurrentCUDAStream(base.get_device());
    cudaError_t e = dp::launchPrep(static_cast<int>(rows), static_cast<int>(k), slotIds.data_ptr<int64_t>(),
        dummySlotRow, offsets.data_ptr<int64_t>(), rngBase, span, idxMap.data_ptr<int32_t>(),
        samplePos.data_ptr<int64_t>(), stream);
    TORCH_CHECK(e == cudaSuccess, "draft_prob1 prep: ", cudaGetErrorString(e));

    dp::Prob1Params a{};
    a.base = base.data_ptr();
    a.ld = base.stride(0);
    a.w1 = w1.data_ptr();
    a.w2s = w2s.data_ptr();
    a.vs = static_cast<int>(w2s.size(0));
    a.voff = static_cast<int>(voff);
    a.rows = static_cast<int>(rows);
    a.k = static_cast<int>(k);
    a.prevIn = prevIn.data_ptr<int64_t>();
    a.prev0 = prev0.data_ptr<int64_t>();
    a.prev1 = prev1.data_ptr<int64_t>();
    a.tokOut = tokOut.data_ptr<int64_t>();
    a.tokLd = static_cast<int>(tokOut.stride(0));
    a.idxMap = idxMap.data_ptr<int32_t>();
    a.samplePos = samplePos.data_ptr<int64_t>();
    a.temperature = temperature.data_ptr<float>();
    a.seeds = seeds.data_ptr<int64_t>();
    // the peer's caches[]: [0] own, then the other ranks in rank order
    a.caches[0] = cache.data_ptr();
    a.ncache = 1;
    for (int64_t s = 0; s < world; ++s)
    {
        if (s != rank)
        {
            TORCH_CHECK(caches[s] != 0, "draft_prob1: peer cache ", s, " not mapped");
            a.caches[a.ncache++] = reinterpret_cast<void*>(caches[s]);
        }
    }
    a.cs0 = cache.stride(0);
    a.cs1 = cache.stride(1);
    a.part = reinterpret_cast<unsigned long long*>(ws[0]);
    a.partMs = reinterpret_cast<float*>(ws[1]);
    a.counter = reinterpret_cast<unsigned int*>(ws[2]);
    a.epoch = reinterpret_cast<unsigned long long*>(ws[3]);
    a.mine = reinterpret_cast<unsigned long long*>(ws[4]);
    a.err = reinterpret_cast<int*>(ws[5]);
    for (int64_t s = 0; s < dp::kMaxWorld; ++s)
    {
        TORCH_CHECK(s >= world || peers[s] != 0, "draft_prob1: peer ", s, " exchange slots not mapped");
        a.peers[s] = s < world ? reinterpret_cast<unsigned long long*>(peers[s]) : nullptr;
    }
    TORCH_CHECK(peers[rank] == ws[4], "draft_prob1: peers[rank] is this rank's slots");
    a.rank = static_cast<int>(rank);
    a.world = static_cast<int>(world);
    e = dp::launchProb1(a, static_cast<int>(grid), pdl, stream);
    TORCH_CHECK(e == cudaSuccess, "dk_mk_prob1 launch: ", cudaGetErrorString(e));

    std::vector<unsigned long long*> fp(static_cast<size_t>(world));
    for (int64_t s = 0; s < world; ++s)
    {
        TORCH_CHECK(faultPeers[s] != 0, "draft_prob1: fault words ", s, " not mapped");
        fp[s] = reinterpret_cast<unsigned long long*>(faultPeers[s]);
    }
    TORCH_CHECK(faultPeers[rank] == ws[6], "draft_prob1: fault_peers[rank] is this rank's fault words");
    e = dp::launchAgree(static_cast<int>(rank), static_cast<int>(world), a.err, a.counter, a.epoch, a.mine,
        reinterpret_cast<unsigned long long*>(ws[6]), fp.data(), reinterpret_cast<unsigned long long*>(seq.data_ptr()),
        capNs, fault.data_ptr<int32_t>(), faultCount.data_ptr<int64_t>(), stream);
    TORCH_CHECK(e == cudaSuccess, "draft_prob1 agree: ", cudaGetErrorString(e));

    dp::ProbsParams q{};
    q.cache = cache.data_ptr();
    q.cs0 = cache.stride(0);
    q.cs1 = cache.stride(1);
    q.tok = tokOut.data_ptr<int64_t>();
    q.tokLd = static_cast<int>(tokOut.stride(0));
    q.temperature = temperature.data_ptr<float>();
    q.slotIds = slotIds.data_ptr<int64_t>();
    q.dummySlotRow = dummySlotRow;
    q.fault = fault.data_ptr<int32_t>();
    q.stats = stats.data_ptr<float>();
    q.draftProbs = draftProbs.data_ptr<float>();
    q.ds0 = draftProbs.stride(0);
    q.ds1 = draftProbs.stride(1);
    q.tokens32 = tokens32.data_ptr<int32_t>();
    q.rows = static_cast<int>(rows);
    q.k = static_cast<int>(k);
    q.vocab = static_cast<int>(vocab);
    e = dp::launchProbs(q, stream);
    TORCH_CHECK(e == cudaSuccess, "draft_prob1 probs: ", cudaGetErrorString(e));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def("minimax_m3_sm107tp8_draft_prob1_max_grid(int rows) -> int");
    m.def(
        "minimax_m3_sm107tp8_draft_prob1(Tensor base, Tensor w1, Tensor w2s, int voff, int k, Tensor prev_in, "
        "Tensor(a!) prev0, Tensor(b!) prev1, Tensor(c!) tok_out, int[] ws, int[] peers, int[] caches, int rank, "
        "int world, int grid, bool pdl, Tensor slot_ids, int dummy_slot_row, Tensor offsets, int rng_base, int span, "
        "Tensor(d!) idx_map, Tensor(e!) sample_pos, Tensor temperature, Tensor seeds, Tensor(f!) cache, "
        "Tensor(g!) fault, Tensor(h!) fault_count, Tensor(i!) seq, int[] fault_peers, int cap_ns, Tensor(j!) stats, "
        "Tensor(k!) draft_probs, Tensor(l!) tokens32, int vocab) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m3_sm107tp8_draft_prob1", &TRTLLM_MV2_NS::torch_ext::minimaxM3Sm107Tp8DraftProb1);
}

TRTLLM_MV2_LIBRARY_IMPL(CompositeExplicitAutograd, m)
{
    m.impl("minimax_m3_sm107tp8_draft_prob1_max_grid", &TRTLLM_MV2_NS::torch_ext::minimaxM3Sm107Tp8DraftProb1MaxGrid);
}
