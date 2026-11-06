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
#include "tensorrt_llm/kernels/glm5MtpStep/glm5MtpStep.h"
#include "tensorrt_llm/kernels/lmHeadArgmax/lmHeadArgmax.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <optional>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace mtp = tensorrt_llm::kernels::glm5_mtp_step;

namespace
{

struct Launch
{
    int ctas;
    cudaStream_t stream;
    bool pdl;
};

Launch launchContext(torch::Tensor const& t)
{
    TORCH_CHECK(tensorrt_llm::common::getSMVersion() >= 90, "glm5_mtp_step needs SM90 or newer");
    auto const device = t.get_device();
    return {at::cuda::getDeviceProperties(device)->multiProcessorCount, at::cuda::getCurrentCUDAStream(device),
        tensorrt_llm::common::getEnvEnablePDL()};
}

void checkBf16(torch::Tensor const& t, char const* name)
{
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kBFloat16, name, " must be a CUDA bf16 tensor");
    TORCH_CHECK(t.stride(-1) == 1, name, " needs a unit inner stride");
}

void checkContiguous(torch::Tensor const& t, char const* name)
{
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " must be a contiguous CUDA tensor");
}

//! The routed experts' fp32 intermediate between fc1 and fc2.
void checkInter(torch::Tensor const& inter)
{
    checkContiguous(inter, "inter");
    TORCH_CHECK(inter.scalar_type() == torch::kFloat32 && inter.numel() >= mtp::kInterElems, "inter must be fp32 [",
        mtp::kMaxSlots, ", ", mtp::kMaxRows, ", ", mtp::kMoeInter, "]");
}

mtp::AllReduce allReduce(std::vector<int64_t> const& peers, int64_t rank)
{
    TORCH_CHECK(!peers.empty() && peers.size() <= static_cast<size_t>(mtp::kMaxRanks),
        "all-reduce peers: one buffer per rank (own buffer included), at most ", mtp::kMaxRanks);
    mtp::AllReduce ar;
    for (size_t r = 0; r < peers.size(); ++r)
    {
        ar.peers[r] = reinterpret_cast<void*>(peers[r]);
    }
    ar.ranks = static_cast<int>(peers.size());
    ar.rank = static_cast<int>(rank);
    TORCH_CHECK(ar.rank >= 0 && ar.rank < ar.ranks, "all-reduce rank out of range");
    return ar;
}

mtp::Routing routing(torch::Tensor const& logits, torch::Tensor const& bias, int64_t topK, double scale,
    int64_t localStart, int64_t localExperts)
{
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32 && bias.scalar_type() == torch::kFloat32,
        "router logits and bias must be fp32");
    checkContiguous(logits, "logits");
    checkContiguous(bias, "bias");
    mtp::Routing r;
    r.logits = logits.data_ptr<float>();
    r.bias = bias.data_ptr<float>();
    r.experts = static_cast<int>(bias.numel());
    TORCH_CHECK(logits.size(-1) == r.experts, "logits and bias disagree on the expert count");
    r.topK = static_cast<int>(topK);
    r.scale = static_cast<float>(scale);
    r.localStart = static_cast<int>(localStart);
    r.localExperts = static_cast<int>(localExperts);
    TORCH_CHECK(r.localStart >= 0 && r.localStart + r.localExperts <= r.experts, "local experts out of range");
    return r;
}

//! The route table (int32, kRouteWords, zero-filled once; the router leaves its counter at zero).
int* routeTable(torch::Tensor const& route)
{
    checkContiguous(route, "route");
    TORCH_CHECK(route.scalar_type() == torch::kInt32 && route.numel() >= mtp::kRouteWords, "route must be int32 [",
        mtp::kRouteWords, "]");
    return route.data_ptr<int32_t>();
}

//! fc1 / fc2: the local experts and top-k (the routing itself is in the router's route table).
mtp::Routing localRouting(int64_t topK, int64_t localStart, int64_t localExperts)
{
    mtp::Routing r;
    r.topK = static_cast<int>(topK);
    r.localStart = static_cast<int>(localStart);
    r.localExperts = static_cast<int>(localExperts);
    TORCH_CHECK(r.topK >= 1 && r.topK <= mtp::kMaxTopK && r.localStart >= 0, "routing out of range");
    return r;
}

} // namespace

void glm5MtpPre(int64_t rows, torch::Tensor const& tokens, int64_t tokensStride, torch::Tensor const& hidden,
    torch::Tensor const& embed, bool useEmbed, torch::Tensor const& gamma, double eps, int64_t colStart,
    torch::Tensor const& weight, std::vector<int64_t> const& arPeers, int64_t rank)
{
    checkBf16(hidden, "hidden");
    checkBf16(embed, "embed");
    checkBf16(gamma, "gamma");
    checkBf16(weight, "weight");
    TORCH_CHECK(tokens.scalar_type() == torch::kInt32, "tokens must be int32");
    TORCH_CHECK(weight.size(0) == mtp::kHidden && weight.size(1) == mtp::kHidden / 2 && weight.is_contiguous(),
        "eh_proj shard must be a contiguous [", mtp::kHidden, ", ", mtp::kHidden / 2, "] tensor");
    mtp::PreParams p;
    p.m = static_cast<int>(rows);
    p.tokens = tokens.data_ptr<int32_t>();
    p.tokensStride = static_cast<int>(tokensStride);
    p.hidden = hidden.data_ptr();
    p.hiddenStride = hidden.stride(0);
    p.embed = embed.data_ptr();
    p.vocab = static_cast<int>(embed.size(0));
    p.useEmbed = useEmbed;
    p.gamma = gamma.data_ptr();
    p.eps = static_cast<float>(eps);
    p.colStart = static_cast<int>(colStart);
    p.weight = weight.data_ptr();
    p.ar = allReduce(arPeers, rank);
    auto const l = launchContext(weight);
    mtp::launchPre(p, l.ctas, l.stream, l.pdl);
}

void glm5MtpQkvA(int64_t rows, std::vector<int64_t> const& arPeers, int64_t rank, torch::Tensor const& gamma,
    double eps, torch::Tensor& residualOut, torch::Tensor const& qkvWeight, torch::Tensor const& idxWeight,
    int64_t idxRows, torch::Tensor& out)
{
    checkBf16(gamma, "gamma");
    checkBf16(qkvWeight, "qkv_weight");
    checkBf16(idxWeight, "idx_weight");
    checkContiguous(residualOut, "residual_out");
    checkContiguous(out, "out");
    TORCH_CHECK(qkvWeight.is_contiguous() && idxWeight.is_contiguous() && qkvWeight.size(1) == mtp::kHidden
            && idxWeight.size(1) == mtp::kHidden && idxWeight.size(0) >= idxRows,
        "qkv_a weights must be contiguous [*, ", mtp::kHidden, "]");
    TORCH_CHECK(out.size(1) == qkvWeight.size(0) + idxRows, "out must be [rows, qkv rows + idx rows]");
    mtp::QkvAParams p;
    p.m = static_cast<int>(rows);
    p.ar = allReduce(arPeers, rank);
    p.gamma = gamma.data_ptr();
    p.eps = static_cast<float>(eps);
    p.residualOut = residualOut.data_ptr();
    p.qkvWeight = qkvWeight.data_ptr();
    p.qkvRows = static_cast<int>(qkvWeight.size(0));
    p.idxWeight = idxWeight.data_ptr();
    p.idxRows = static_cast<int>(idxRows);
    p.out = out.data_ptr();
    auto const l = launchContext(out);
    mtp::launchQkvA(p, l.ctas, l.stream, l.pdl);
}

void glm5MtpQB(torch::Tensor const& qkv, int64_t qaRank, int64_t kvRank, int64_t idxDim, torch::Tensor const& qaGamma,
    torch::Tensor const& kvGamma, double eps, torch::Tensor const& lnWeight, torch::Tensor const& lnBias, double lnEps,
    torch::Tensor& latentOut, torch::Tensor& packedOut, torch::Tensor const& weight, torch::Tensor& queryOut,
    std::optional<torch::Tensor> const& idxQWeight, std::optional<torch::Tensor> const& idxQOut)
{
    TORCH_CHECK(qkv.is_cuda() && qkv.stride(1) == 1, "qkv needs a unit inner stride");
    checkContiguous(latentOut, "latent_out");
    checkContiguous(packedOut, "packed_out");
    checkContiguous(queryOut, "query_out");
    checkContiguous(weight, "weight");
    TORCH_CHECK(qkv.size(1) >= qaRank + kvRank + 2 * idxDim && weight.size(1) == qaRank, "q_b shapes disagree");
    mtp::QBParams p;
    p.m = static_cast<int>(qkv.size(0));
    p.qkv = qkv.data_ptr();
    p.qkvStride = static_cast<int>(qkv.stride(0));
    p.qaRank = static_cast<int>(qaRank);
    p.kvRank = static_cast<int>(kvRank);
    p.idxDim = static_cast<int>(idxDim);
    p.qaGamma = qaGamma.data_ptr();
    p.kvGamma = kvGamma.data_ptr();
    p.eps = static_cast<float>(eps);
    p.lnWeight = lnWeight.data_ptr();
    p.lnBias = lnBias.data_ptr();
    p.lnEps = static_cast<float>(lnEps);
    p.latentOut = latentOut.data_ptr();
    p.packedOut = packedOut.data_ptr();
    p.weight = weight.data_ptr();
    p.qbRows = static_cast<int>(weight.size(0));
    p.queryOut = queryOut.data_ptr();
    TORCH_CHECK(idxQWeight.has_value() == idxQOut.has_value(), "idx_q_weight and idx_q_out go together");
    if (idxQWeight.has_value())
    {
        checkContiguous(*idxQWeight, "idx_q_weight");
        checkContiguous(*idxQOut, "idx_q_out");
        TORCH_CHECK(
            idxQWeight->size(1) == qaRank && idxQOut->size(0) == qkv.size(0) && idxQOut->size(1) == idxQWeight->size(0),
            "indexer query shapes disagree");
        p.idxQWeight = idxQWeight->data_ptr();
        p.idxQRows = static_cast<int>(idxQWeight->size(0));
        p.idxQOut = idxQOut->data_ptr();
    }
    auto const l = launchContext(qkv);
    mtp::launchQB(p, l.ctas, l.stream, l.pdl);
}

void glm5MtpAbsorb(torch::Tensor const& query, torch::Tensor const& wkT, torch::Tensor& out)
{
    checkContiguous(query, "query");
    checkContiguous(wkT, "wk_t");
    checkContiguous(out, "out");
    TORCH_CHECK(wkT.dim() == 3, "wk_t must be [heads, kv_rank, nope]");
    mtp::AbsorbParams p;
    p.m = static_cast<int>(query.size(0));
    p.query = query.data_ptr();
    p.wkT = wkT.data_ptr();
    p.heads = static_cast<int>(wkT.size(0));
    p.kvRank = static_cast<int>(wkT.size(1));
    p.nope = static_cast<int>(wkT.size(2));
    TORCH_CHECK(
        query.numel() == p.m * p.heads * p.nope && out.numel() == p.m * p.heads * p.kvRank, "absorb shapes disagree");
    p.out = out.data_ptr();
    auto const l = launchContext(query);
    mtp::launchAbsorb(p, l.ctas, l.stream, l.pdl);
}

void glm5MtpVUp(torch::Tensor const& latent, torch::Tensor const& kvB, int64_t heads, int64_t nope, int64_t vDim,
    torch::Tensor& out)
{
    checkBf16(latent, "latent");
    checkContiguous(kvB, "kv_b");
    checkContiguous(out, "out");
    mtp::VUpParams p;
    p.m = static_cast<int>(latent.size(0));
    p.latent = latent.data_ptr();
    p.latentStride = latent.stride(0);
    p.kvB = kvB.data_ptr();
    p.heads = static_cast<int>(heads);
    p.nope = static_cast<int>(nope);
    p.vDim = static_cast<int>(vDim);
    p.kvRank = static_cast<int>(kvB.size(1));
    TORCH_CHECK(
        kvB.size(0) == heads * (nope + vDim) && latent.size(1) == heads * p.kvRank && out.numel() == p.m * heads * vDim,
        "v_up shapes disagree");
    p.out = out.data_ptr();
    auto const l = launchContext(latent);
    mtp::launchVUp(p, l.ctas, l.stream, l.pdl);
}

void glm5MtpOProj(
    torch::Tensor const& input, torch::Tensor const& weight, std::vector<int64_t> const& arPeers, int64_t rank)
{
    checkContiguous(input, "input");
    checkContiguous(weight, "weight");
    TORCH_CHECK(weight.size(0) == mtp::kHidden && weight.size(1) == input.size(1), "o_proj shapes disagree");
    mtp::OProjParams p;
    p.m = static_cast<int>(input.size(0));
    p.input = input.data_ptr();
    p.k = static_cast<int>(input.size(1));
    p.weight = weight.data_ptr();
    p.ar = allReduce(arPeers, rank);
    auto const l = launchContext(input);
    mtp::launchOProj(p, l.ctas, l.stream, l.pdl);
}

void glm5MtpRouter(std::vector<int64_t> const& arPeers, int64_t rank, torch::Tensor const& residualIn,
    torch::Tensor& residualOut, torch::Tensor const& gamma, double eps, torch::Tensor& normOut,
    torch::Tensor const& gateWeight, torch::Tensor& logitsOut, torch::Tensor const& sharedGateUp, double swigluLimit,
    torch::Tensor& sharedOut, torch::Tensor const& bias, int64_t topK, double scale, int64_t localStart,
    int64_t localExperts, torch::Tensor& route, std::optional<torch::Tensor> const& rows)
{
    checkContiguous(residualIn, "residual_in");
    checkContiguous(residualOut, "residual_out");
    checkContiguous(normOut, "norm_out");
    checkContiguous(gateWeight, "gate_weight");
    checkContiguous(logitsOut, "logits_out");
    checkContiguous(sharedGateUp, "shared_gate_up");
    checkContiguous(sharedOut, "shared_out");
    TORCH_CHECK(logitsOut.scalar_type() == torch::kFloat32, "logits_out must be fp32");
    mtp::RouterParams p;
    p.m = static_cast<int>(residualIn.size(0));
    p.ar = allReduce(arPeers, rank);
    p.residualIn = residualIn.data_ptr();
    p.residualOut = residualOut.data_ptr();
    p.gamma = gamma.data_ptr();
    p.eps = static_cast<float>(eps);
    p.normOut = normOut.data_ptr();
    p.gateWeight = gateWeight.data_ptr();
    p.experts = static_cast<int>(gateWeight.size(0));
    p.logitsOut = logitsOut.data_ptr<float>();
    p.sharedGateUp = sharedGateUp.data_ptr();
    p.sharedInter = static_cast<int>(sharedGateUp.size(0) / 2);
    p.swigluLimit = static_cast<float>(swigluLimit);
    p.sharedOut = sharedOut.data_ptr();
    TORCH_CHECK(logitsOut.size(1) == p.experts && sharedOut.size(1) == p.sharedInter, "router outputs disagree");
    p.routing = routing(logitsOut, bias, topK, scale, localStart, localExperts);
    p.route = routeTable(route);
    // residual_in / the all-reduce input hold more rows than the router takes (step 0): rows picks them.
    p.m = static_cast<int>(normOut.size(0));
    if (rows.has_value())
    {
        TORCH_CHECK(
            rows->is_cuda() && rows->scalar_type() == torch::kInt32 && rows->is_contiguous() && rows->numel() == p.m,
            "rows must be a contiguous int32 tensor of the router's row count");
        p.rowIdx = rows->data_ptr<int32_t>();
    }
    auto const l = launchContext(residualIn);
    mtp::launchRouter(p, l.ctas, l.stream, l.pdl);
}

void glm5MtpFc1(torch::Tensor const& route, int64_t topK, int64_t localStart, torch::Tensor const& input,
    torch::Tensor const& w13, torch::Tensor const& w13Scale, double swigluLimit, torch::Tensor& inter,
    torch::Tensor const& sharedIn, torch::Tensor const& sharedDown, torch::Tensor& sharedDownOut)
{
    checkContiguous(input, "input");
    checkContiguous(w13, "w13");
    checkContiguous(w13Scale, "w13_scale");
    checkInter(inter);
    checkContiguous(sharedIn, "shared_in");
    checkContiguous(sharedDown, "shared_down");
    checkContiguous(sharedDownOut, "shared_down_out");
    TORCH_CHECK(
        w13.dim() == 3 && w13.size(1) == 2 * mtp::kMoeInter && w13.size(2) == mtp::kHidden && w13.element_size() == 1,
        "w13 must be e4m3 [local experts, ", 2 * mtp::kMoeInter, ", ", mtp::kHidden, "]");
    mtp::Fc1Params p;
    p.m = static_cast<int>(input.size(0));
    p.routing = localRouting(topK, localStart, w13.size(0));
    p.route = routeTable(route);
    p.input = input.data_ptr();
    p.w13 = w13.data_ptr();
    p.w13Scale = w13Scale.data_ptr<float>();
    p.swigluLimit = static_cast<float>(swigluLimit);
    p.inter = inter.data_ptr<float>();
    p.sharedIn = sharedIn.data_ptr();
    p.sharedInter = static_cast<int>(sharedDown.size(1));
    p.sharedDown = sharedDown.data_ptr();
    p.sharedDownOut = sharedDownOut.data_ptr();
    auto const l = launchContext(input);
    mtp::launchFc1(p, l.ctas, l.stream, l.pdl);
}

void glm5MtpFc2(int64_t rows, torch::Tensor const& route, int64_t topK, int64_t localStart, torch::Tensor const& w2,
    torch::Tensor const& w2Scale, torch::Tensor const& inter, torch::Tensor const& shared,
    std::vector<int64_t> const& arPeers, int64_t rank)
{
    checkContiguous(w2, "w2");
    checkContiguous(w2Scale, "w2_scale");
    checkInter(inter);
    checkContiguous(shared, "shared");
    TORCH_CHECK(w2.dim() == 3 && w2.size(1) == mtp::kHidden && w2.size(2) == mtp::kMoeInter && w2.element_size() == 1,
        "w2 must be e4m3 [local experts, ", mtp::kHidden, ", ", mtp::kMoeInter, "]");
    mtp::Fc2Params p;
    p.m = static_cast<int>(rows);
    p.routing = localRouting(topK, localStart, w2.size(0));
    p.route = routeTable(route);
    p.w2 = w2.data_ptr();
    p.w2Scale = w2Scale.data_ptr<float>();
    p.inter = inter.data_ptr<float>();
    p.shared = shared.data_ptr();
    p.ar = allReduce(arPeers, rank);
    auto const l = launchContext(shared);
    mtp::launchFc2(p, l.ctas, l.stream, l.pdl);
}

void glm5MtpHead(std::vector<int64_t> const& arPeers, int64_t rank, torch::Tensor const& residualIn,
    torch::Tensor const& gamma, double eps, torch::Tensor& hiddenOut, torch::Tensor const& weight, int64_t vocabOffset,
    torch::Tensor& tokens, int64_t tokensStride, torch::Tensor& scratch, std::vector<int64_t> const& keyPeers,
    int64_t keyRank)
{
    namespace lha = tensorrt_llm::kernels::lm_head_argmax;
    checkContiguous(residualIn, "residual_in");
    checkBf16(hiddenOut, "hidden_out");
    checkContiguous(weight, "weight");
    TORCH_CHECK(weight.size(1) == mtp::kHidden, "LM head shard must be [n, ", mtp::kHidden, "]");
    TORCH_CHECK(tokens.scalar_type() == torch::kInt32, "tokens must be int32");
    TORCH_CHECK(
        scratch.scalar_type() == torch::kInt64 && scratch.is_contiguous() && scratch.numel() >= lha::kScratchWords,
        "scratch must be a contiguous int64 tensor of at least ", lha::kScratchWords, " elements");
    mtp::HeadParams p;
    p.m = static_cast<int>(residualIn.size(0));
    p.ar = allReduce(arPeers, rank);
    p.residualIn = residualIn.data_ptr();
    p.gamma = gamma.data_ptr();
    p.eps = static_cast<float>(eps);
    p.hiddenOut = hiddenOut.data_ptr();
    p.hiddenStride = hiddenOut.stride(0);
    p.weight = weight.data_ptr();
    p.n = static_cast<int>(weight.size(0));
    p.vocabOffset = static_cast<int>(vocabOffset);
    p.tokens = tokens.data_ptr<int32_t>();
    p.tokensStride = static_cast<int>(tokensStride);
    TORCH_CHECK(tokensStride >= 1 && tokens.numel() >= (p.m - 1) * tokensStride + 1, "tokens too small");
    p.scratch = reinterpret_cast<uint64_t*>(scratch.data_ptr<int64_t>());
    p.keyRanks = std::max<int>(1, static_cast<int>(keyPeers.size()));
    for (size_t r = 0; r < keyPeers.size(); ++r)
    {
        p.keyPeers[r] = reinterpret_cast<uint64_t*>(keyPeers[r]);
    }
    p.keyRank = static_cast<int>(keyRank);
    auto const l = launchContext(weight);
    mtp::launchHead(p, l.ctas, l.stream, l.pdl);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "glm5_mtp_pre(int rows, Tensor tokens, int tokens_stride, Tensor hidden, Tensor embed, bool use_embed, "
        "Tensor gamma, float eps, int col_start, Tensor weight, int[] ar_peers, int rank) -> ()");
    m.def(
        "glm5_mtp_qkv_a(int rows, int[] ar_peers, int rank, Tensor gamma, float eps, Tensor(a!) residual_out, "
        "Tensor qkv_weight, Tensor idx_weight, int idx_rows, Tensor(b!) out) -> ()");
    m.def(
        "glm5_mtp_q_b(Tensor qkv, int qa_rank, int kv_rank, int idx_dim, Tensor qa_gamma, Tensor kv_gamma, float eps, "
        "Tensor ln_weight, Tensor ln_bias, float ln_eps, Tensor(a!) latent_out, Tensor(b!) packed_out, Tensor weight, "
        "Tensor(c!) query_out, Tensor? idx_q_weight=None, Tensor(d!)? idx_q_out=None) -> ()");
    m.def("glm5_mtp_absorb(Tensor query, Tensor wk_t, Tensor(a!) out) -> ()");
    m.def("glm5_mtp_v_up(Tensor latent, Tensor kv_b, int heads, int nope, int v_dim, Tensor(a!) out) -> ()");
    m.def("glm5_mtp_o_proj(Tensor input, Tensor weight, int[] ar_peers, int rank) -> ()");
    m.def(
        "glm5_mtp_router(int[] ar_peers, int rank, Tensor residual_in, Tensor(a!) residual_out, Tensor gamma, "
        "float eps, Tensor(b!) norm_out, Tensor gate_weight, Tensor(c!) logits_out, Tensor shared_gate_up, "
        "float swiglu_limit, Tensor(d!) shared_out, Tensor bias, int top_k, float scale, int local_start, "
        "int local_experts, Tensor(e!) route, Tensor? rows=None) -> ()");
    m.def(
        "glm5_mtp_fc1(Tensor route, int top_k, int local_start, Tensor input, Tensor w13, "
        "Tensor w13_scale, float swiglu_limit, Tensor(a!) inter, Tensor shared_in, Tensor shared_down, "
        "Tensor(b!) shared_down_out) -> ()");
    m.def(
        "glm5_mtp_fc2(int rows, Tensor route, int top_k, int local_start, Tensor w2, "
        "Tensor w2_scale, Tensor inter, Tensor shared, int[] ar_peers, int rank) -> ()");
    m.def(
        "glm5_mtp_head(int[] ar_peers, int rank, Tensor residual_in, Tensor gamma, float eps, Tensor(a!) hidden_out, "
        "Tensor weight, int vocab_offset, Tensor(b!) tokens, int tokens_stride, Tensor(c!) scratch, int[] key_peers, "
        "int key_rank) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("glm5_mtp_pre", &tensorrt_llm::torch_ext::glm5MtpPre);
    m.impl("glm5_mtp_qkv_a", &tensorrt_llm::torch_ext::glm5MtpQkvA);
    m.impl("glm5_mtp_q_b", &tensorrt_llm::torch_ext::glm5MtpQB);
    m.impl("glm5_mtp_absorb", &tensorrt_llm::torch_ext::glm5MtpAbsorb);
    m.impl("glm5_mtp_v_up", &tensorrt_llm::torch_ext::glm5MtpVUp);
    m.impl("glm5_mtp_o_proj", &tensorrt_llm::torch_ext::glm5MtpOProj);
    m.impl("glm5_mtp_router", &tensorrt_llm::torch_ext::glm5MtpRouter);
    m.impl("glm5_mtp_fc1", &tensorrt_llm::torch_ext::glm5MtpFc1);
    m.impl("glm5_mtp_fc2", &tensorrt_llm::torch_ext::glm5MtpFc2);
    m.impl("glm5_mtp_head", &tensorrt_llm::torch_ext::glm5MtpHead);
}
