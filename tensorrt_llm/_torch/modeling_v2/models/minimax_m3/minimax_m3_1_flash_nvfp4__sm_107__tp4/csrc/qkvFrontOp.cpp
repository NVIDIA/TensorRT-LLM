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

#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp4/csrc/qkvFrontSm107.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <initializer_list>
#include <optional>
#include <vector>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace mk = TRTLLM_MV2_NS::kernels::minimaxM31;

constexpr int64_t kHeadDim = 128;
constexpr int64_t kPageSize = 128;

} // namespace

//! The sparse-layer front in one launch (csrc/qkvFrontSm107.cu): the MoE boundary all-reduce as the
//! prepushed 2 / 3 consumer of m3_mc_allreduce_norm's exchange (``ws_uc`` = the bucket's unicast view, ``flags`` its
//! cluster-8 call counters) writing ``out_residual`` (and ``out_normed``), the QKV MXFP8 GEMV and the producer
//! epilogue (minimax_m3_sm107_tp4_producer_pw's operands and outputs). Returns (q FP8 [T, heads_q, 128], index_q FP8
//! [T, 1, 128]) and writes the NVFP4 K / V and FP8 index-K of each token's cache slot. ``xq_ws`` / ``xsf_ws`` /
//! ``ctr`` are the front's hand-off workspace for this ``split``.
std::vector<torch::Tensor> minimaxM31QkvFront(torch::Tensor const& residual, torch::Tensor const& normWeight,
    double eps, torch::Tensor& outResidual, std::optional<torch::Tensor> const& outNormed, torch::Tensor& wsUc,
    torch::Tensor& flags, int64_t rank, int64_t world, int64_t maxt, int64_t prepushed, torch::Tensor const& weight,
    torch::Tensor const& weightScale, torch::Tensor& xqWs, torch::Tensor& xsfWs, torch::Tensor& ctr,
    torch::Tensor& kvData, torch::Tensor& kvScale, torch::Tensor& indexK, torch::Tensor const& outCacheLoc,
    torch::Tensor const& kvQuantScale, double peps, torch::Tensor const& qWeight, torch::Tensor const& kWeight,
    torch::Tensor const& indexQWeight, torch::Tensor const& indexKWeight, torch::Tensor const& rotaryCosSin,
    torch::Tensor const& positionIds, int64_t headsQ, int64_t split, bool pdl, bool early)
{
    TORCH_CHECK(residual.is_cuda(), "qkv_front: residual must be a CUDA tensor");
    for (torch::Tensor const* t : std::initializer_list<torch::Tensor const*>{&normWeight, &outResidual, &wsUc, &flags,
             &weight, &weightScale, &xqWs, &xsfWs, &ctr, &kvData, &kvScale, &indexK, &outCacheLoc, &kvQuantScale,
             &qWeight, &kWeight, &indexQWeight, &indexKWeight, &rotaryCosSin, &positionIds})
    {
        TORCH_CHECK(t->is_cuda() && t->get_device() == residual.get_device(),
            "qkv_front: every operand must be on residual's CUDA device");
    }
    c10::cuda::CUDAGuard const deviceGuard{residual.device()};
    static int const sSmVersion = tensorrt_llm::common::getSMVersion();
    TORCH_CHECK(sSmVersion >= 100 && sSmVersion < 110, "minimax_m31_qkv_front requires an SM10x GPU");
    TORCH_CHECK(mk::isQkvFrontSplitSupported(static_cast<int>(split)), "qkv_front: split 8 or 4");
    int64_t const hidden = mk::kQkvFrontHidden;
    int64_t const numTokens = residual.size(0);
    int64_t const heads = headsQ + 4;
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16 && residual.dim() == 2 && residual.size(1) == hidden
            && residual.is_contiguous() && numTokens >= 1 && numTokens <= mk::kQkvFrontMaxTokens,
        "qkv_front: residual must be contiguous BF16 [1..8, 6144]");
    TORCH_CHECK(outResidual.sizes() == residual.sizes() && outResidual.scalar_type() == at::kBFloat16
            && outResidual.is_contiguous(),
        "qkv_front: out_residual must be contiguous BF16 like residual");
    if (outNormed.has_value())
    {
        TORCH_CHECK(outNormed->sizes() == residual.sizes() && outNormed->scalar_type() == at::kBFloat16
                && outNormed->is_contiguous() && outNormed->is_cuda()
                && outNormed->get_device() == residual.get_device(),
            "qkv_front: out_normed must be contiguous BF16 like residual");
    }
    TORCH_CHECK(normWeight.scalar_type() == at::kBFloat16 && normWeight.is_contiguous() && normWeight.numel() == hidden,
        "qkv_front: norm_weight must be BF16 [6144]");
    TORCH_CHECK(wsUc.scalar_type() == at::kInt && wsUc.is_contiguous() && world >= 2 && world <= 8 && rank >= 0
            && rank < world && maxt >= numTokens && wsUc.numel() >= 4 * maxt * world * (hidden / 2),
        "qkv_front: ws_uc must be the bucket's int32 exchange [4 * maxt * world * 3072]");
    TORCH_CHECK(flags.scalar_type() == at::kInt && flags.is_contiguous() && flags.numel() >= maxt * 8,
        "qkv_front: flags must be the bucket's cluster-8 call counters [maxt * 8]");
    TORCH_CHECK(prepushed == 2 || prepushed == 3, "qkv_front: prepushed 2 or 3");
    TORCH_CHECK(weight.dim() == 2 && weight.size(0) == heads * kHeadDim && weight.size(1) == hidden
            && weight.element_size() == 1 && weight.is_contiguous()
            && reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
        "qkv_front: weight must be contiguous E4M3 [(heads_q + 4) * 128, 6144], 16-byte aligned");
    TORCH_CHECK(weightScale.element_size() == 1 && weightScale.is_contiguous()
            && weightScale.numel() == heads * kHeadDim * hidden / 32
            && reinterpret_cast<uintptr_t>(weightScale.data_ptr()) % 16 == 0,
        "qkv_front: weight_scale must be the 128x4-swizzled UE8M0 buffer [N * 6144 / 32], 16-byte aligned");
    TORCH_CHECK(xqWs.element_size() == 1 && xqWs.is_contiguous() && xqWs.numel() == mk::kQkvFrontXqBytes
            && xsfWs.element_size() == 1 && xsfWs.is_contiguous() && xsfWs.numel() == mk::kQkvFrontXsfBytes
            && ctr.scalar_type() == at::kInt && ctr.is_contiguous() && ctr.numel() >= mk::kQkvFrontMaxCtas
            && reinterpret_cast<uintptr_t>(xqWs.data_ptr()) % 16 == 0
            && reinterpret_cast<uintptr_t>(xsfWs.data_ptr()) % 16 == 0,
        "qkv_front: the hand-off workspace (xq, xsf, ctr) does not have its layout");
    // The producer's layout (minimax_m3_sm107_tp4_producer_pw's checks at one KV head).
    TORCH_CHECK(headsQ >= 1 && heads * split <= mk::kQkvFrontMaxCtas, "qkv_front: too many heads");
    TORCH_CHECK(kvData.scalar_type() == at::kByte && kvData.dim() == 5 && kvData.size(0) > 0 && kvData.size(1) == 2
            && kvData.size(2) == 1 && kvData.size(3) == kPageSize && kvData.size(4) == kHeadDim / 2,
        "qkv_front: kv_data must be the NVFP4 pool [pages, 2, 1, 128, 64] uint8");
    TORCH_CHECK(kvScale.scalar_type() == at::kByte && kvScale.dim() == 5 && kvScale.size(0) == kvData.size(0)
            && kvScale.size(1) == 2 && kvScale.size(2) == 1 && kvScale.size(3) == kPageSize
            && kvScale.size(4) == kHeadDim / 16,
        "qkv_front: kv_scale must be the E4M3 scale pool [pages, 2, 1, 128, 8] uint8");
    TORCH_CHECK(kvData.stride(4) == 1 && kvData.stride(3) == kvData.size(4)
            && kvData.stride(2) == kvData.size(3) * kvData.stride(3) && kvScale.stride(4) == 1
            && kvScale.stride(3) == kvScale.size(4) && kvScale.stride(2) == kvScale.size(3) * kvScale.stride(3),
        "qkv_front: the pools must have dense token rows and head blocks");
    TORCH_CHECK(kvData.stride(1) >= kvData.size(2) * kvData.stride(2) && kvData.stride(0) >= 2 * kvData.stride(1)
            && kvScale.stride(1) >= kvScale.size(2) * kvScale.stride(2) && kvScale.stride(0) >= 2 * kvScale.stride(1)
            && kvData.stride(0) % 2 == 0 && kvData.stride(1) % 2 == 0
            && reinterpret_cast<uintptr_t>(kvData.data_ptr()) % 2 == 0,
        "qkv_front: pool planes or pages overlap, or break the uint16 stores' alignment");
    TORCH_CHECK(indexK.scalar_type() == at::kFloat8_e4m3fn && indexK.dim() == 4 && indexK.size(0) == kvData.size(0)
            && indexK.size(1) == 1 && indexK.size(2) == kPageSize && indexK.size(3) == kHeadDim && indexK.stride(3) == 1
            && indexK.stride(2) == kHeadDim && indexK.stride(0) >= kPageSize * kHeadDim && indexK.stride(0) % 4 == 0
            && reinterpret_cast<uintptr_t>(indexK.data_ptr()) % 4 == 0,
        "qkv_front: index_k must be FP8 [pages, 1, 128, 128] keeping the 32-bit stores aligned");
    TORCH_CHECK(
        outCacheLoc.scalar_type() == at::kInt && outCacheLoc.is_contiguous() && outCacheLoc.numel() >= numTokens,
        "qkv_front: out_cache_loc must be int32 [>= T]");
    TORCH_CHECK(
        positionIds.scalar_type() == at::kInt && positionIds.is_contiguous() && positionIds.numel() >= numTokens,
        "qkv_front: position_ids must be int32 [>= T]");
    TORCH_CHECK(kvQuantScale.scalar_type() == at::kFloat && kvQuantScale.is_contiguous() && kvQuantScale.numel() >= 3,
        "qkv_front: kv_quant_scale must be fp32 [>= 3]");
    TORCH_CHECK(rotaryCosSin.scalar_type() == at::kFloat && rotaryCosSin.is_contiguous() && rotaryCosSin.dim() == 3
            && rotaryCosSin.size(1) == 2 && rotaryCosSin.size(2) == 32
            && reinterpret_cast<uintptr_t>(rotaryCosSin.data_ptr()) % 16 == 0,
        "qkv_front: rotary_cos_sin must be fp32 [positions, 2, 32], 16-byte aligned");
    for (auto const* w : {&qWeight, &kWeight, &indexQWeight, &indexKWeight})
    {
        TORCH_CHECK(w->scalar_type() == at::kBFloat16 && w->is_contiguous() && w->numel() == kHeadDim
                && reinterpret_cast<uintptr_t>(w->data_ptr()) % 8 == 0,
            "qkv_front: q / k / index norm weights must be BF16 [128], 8-byte aligned");
    }
    TORCH_CHECK(reinterpret_cast<uintptr_t>(residual.data_ptr()) % 16 == 0
            && reinterpret_cast<uintptr_t>(outResidual.data_ptr()) % 16 == 0
            && reinterpret_cast<uintptr_t>(normWeight.data_ptr()) % 16 == 0
            && reinterpret_cast<uintptr_t>(wsUc.data_ptr()) % 16 == 0,
        "qkv_front: the boundary operands must be 16-byte aligned");

    auto const fp8 = residual.options().dtype(at::kFloat8_e4m3fn);
    auto qOut = torch::empty({numTokens, headsQ, kHeadDim}, fp8);
    auto indexQOut = torch::empty({numTokens, 1, kHeadDim}, fp8);

    mk::QkvFrontParams p{};
    p.wsUc = wsUc.data_ptr<int32_t>();
    p.flags = flags.data_ptr<int32_t>();
    p.rank = static_cast<int32_t>(rank);
    p.world = static_cast<int32_t>(world);
    p.maxt = static_cast<int32_t>(maxt);
    p.prepushed = static_cast<int32_t>(prepushed);
    p.residual = reinterpret_cast<__nv_bfloat16 const*>(residual.data_ptr());
    p.normWeight = reinterpret_cast<__nv_bfloat16 const*>(normWeight.data_ptr());
    p.eps = static_cast<float>(eps);
    p.outResidual = reinterpret_cast<__nv_bfloat16*>(outResidual.data_ptr());
    p.outNormed = outNormed.has_value() ? reinterpret_cast<__nv_bfloat16*>(outNormed->data_ptr()) : nullptr;
    p.xq = xqWs.data_ptr<uint8_t>();
    p.xsf = xsfWs.data_ptr<uint8_t>();
    p.ctr = ctr.data_ptr<int32_t>();
    p.weight = weight.data_ptr();
    p.weightScale = reinterpret_cast<uint8_t const*>(weightScale.data_ptr());
    p.qOut = reinterpret_cast<__nv_fp8_e4m3*>(qOut.data_ptr());
    p.indexQOut = reinterpret_cast<__nv_fp8_e4m3*>(indexQOut.data_ptr());
    p.kvData = kvData.data_ptr<uint8_t>();
    p.kvScale = kvScale.data_ptr<uint8_t>();
    p.indexK = reinterpret_cast<__nv_fp8_e4m3*>(indexK.data_ptr());
    p.outCacheLoc = outCacheLoc.data_ptr<int32_t>();
    p.kvQuantScale = kvQuantScale.data_ptr<float>();
    p.dataPageStride = kvData.stride(0);
    p.dataPlaneStride = kvData.stride(1);
    p.dataHeadStride = kvData.stride(2);
    p.dataTokenStride = kvData.stride(3);
    p.scalePageStride = kvScale.stride(0);
    p.scalePlaneStride = kvScale.stride(1);
    p.scaleHeadStride = kvScale.stride(2);
    p.indexPageStride = indexK.stride(0);
    p.indexTokenStride = indexK.stride(2);
    p.numPages = kvData.size(0);
    p.peps = static_cast<float>(peps);
    p.qWeight = reinterpret_cast<__nv_bfloat16 const*>(qWeight.data_ptr());
    p.kWeight = reinterpret_cast<__nv_bfloat16 const*>(kWeight.data_ptr());
    p.indexQWeight = reinterpret_cast<__nv_bfloat16 const*>(indexQWeight.data_ptr());
    p.indexKWeight = reinterpret_cast<__nv_bfloat16 const*>(indexKWeight.data_ptr());
    p.rotaryCosSin = rotaryCosSin.data_ptr<float>();
    p.positionIds = positionIds.data_ptr<int32_t>();
    p.numTokens = static_cast<int32_t>(numTokens);
    p.headsQ = static_cast<int32_t>(headsQ);
    cudaError_t const err = mk::invokeQkvFront(
        p, static_cast<int>(split), pdl, early, at::cuda::getCurrentCUDAStream(residual.get_device()));
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TORCH_CHECK(false, "qkv_front launch: ", cudaGetErrorString(err),
            err == cudaErrorLaunchOutOfResources ? " (the device cannot hold every cluster of the grid at once)" : "");
    }
    return {qOut, indexQOut};
}

//! How many of the front's clusters at ``split`` the current device holds at once (-1: the kernel cannot run here).
int64_t minimaxM31QkvFrontMaxClusters(int64_t split)
{
    TORCH_CHECK(mk::isQkvFrontSplitSupported(static_cast<int>(split)), "qkv_front: split 8 or 4");
    return mk::qkvFrontMaxActiveClusters(static_cast<int>(split));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m31_qkv_front(Tensor residual, Tensor norm_weight, float eps, Tensor(a!) out_residual, "
        "Tensor(b!)? out_normed, Tensor(c!) ws_uc, Tensor(d!) flags, int rank, int world, int maxt, int prepushed, "
        "Tensor weight, Tensor weight_scale, Tensor(e!) xq_ws, Tensor(f!) xsf_ws, Tensor(g!) ctr, Tensor(h!) kv_data, "
        "Tensor(i!) kv_scale, Tensor(j!) index_k, Tensor out_cache_loc, Tensor kv_quant_scale, float peps, "
        "Tensor q_weight, Tensor k_weight, Tensor index_q_weight, Tensor index_k_weight, Tensor rotary_cos_sin, "
        "Tensor position_ids, int heads_q, int split, bool pdl, bool early) -> Tensor[]");
    m.def("minimax_m31_qkv_front_max_clusters(int split) -> int");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_qkv_front", &TRTLLM_MV2_NS::torch_ext::minimaxM31QkvFront);
}

// No tensor arguments, so no dispatch key: a host query registered for every backend (the fake raises under tracing).
TRTLLM_MV2_LIBRARY_IMPL(CompositeExplicitAutograd, m)
{
    m.impl("minimax_m31_qkv_front_max_clusters", &TRTLLM_MV2_NS::torch_ext::minimaxM31QkvFrontMaxClusters);
}
