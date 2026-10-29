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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/draftRoutedMoe.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cuda.h>
#include <cudaTypedefs.h>

#include <cstdint>
#include <initializer_list>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>
#include <vector>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace tk = TRTLLM_MV2_NS::draft_routed_moe;

#ifndef GLM_MOE_DSA_DRAFT_ROUTED_MOE_TIMELINE
#define GLM_MOE_DSA_DRAFT_ROUTED_MOE_TIMELINE 0
#endif
//! Probe builds compile the kernels with their event record (see draftRoutedMoe.cu).
constexpr bool kTimelineBuild = GLM_MOE_DSA_DRAFT_ROUTED_MOE_TIMELINE != 0;

void checkTensor(at::Tensor const& t, char const* name, at::ScalarType dtype, std::initializer_list<int64_t> shape,
    c10::DeviceIndex device)
{
    TORCH_CHECK(t.is_cuda() && t.get_device() == device, "[glm_moe_dsa_draft_routed_moe] ", name,
        " must be on the activation's device");
    TORCH_CHECK(t.scalar_type() == dtype, "[glm_moe_dsa_draft_routed_moe] ", name, " has dtype ", t.scalar_type());
    TORCH_CHECK(t.is_contiguous(), "[glm_moe_dsa_draft_routed_moe] ", name, " must be contiguous");
    TORCH_CHECK(t.sizes().vec() == std::vector<int64_t>(shape), "[glm_moe_dsa_draft_routed_moe] ", name, " has shape ",
        t.sizes());
}

//! ``cuTensorMapEncodeTiled`` through the runtime's driver entry point (no link-time driver dependency).
PFN_cuTensorMapEncodeTiled_v12000 encodeTiledFn()
{
    static PFN_cuTensorMapEncodeTiled_v12000 fn = []
    {
        void* ptr = nullptr;
        cudaDriverEntryPointQueryResult query{};
        cudaError_t const status
            = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &query);
        TORCH_CHECK(status == cudaSuccess && query == cudaDriverEntryPointSuccess && ptr != nullptr,
            "[glm_moe_dsa_draft_routed_moe] cuTensorMapEncodeTiled is not available");
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
    }();
    return fn;
}

//! Tensor map of a BlockMajorK weight buffer viewed as [slabs][slabRows][64] BF16 (a slab: one K chunk of one
//! expert) with boxes of 64 x ``boxRows`` x ``boxSlabs`` and the 128-byte swizzle. Cached per (device, address,
//! shape, box): the expert buffers are fixed model parameters.
CUtensorMap const& weightMap(
    int device, void const* base, uint64_t slabRows, uint64_t slabs, uint32_t boxRows, uint32_t boxSlabs)
{
    static std::mutex mutex;
    static std::map<std::tuple<int, void const*, uint64_t, uint64_t, uint32_t, uint32_t>, CUtensorMap> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto const key = std::make_tuple(device, base, slabRows, slabs, boxRows, boxSlabs);
    auto it = cache.find(key);
    if (it != cache.end())
    {
        return it->second;
    }
    CUtensorMap map{};
    cuuint64_t const rowBytes = static_cast<cuuint64_t>(tk::kKChunk) * 2;
    cuuint64_t const dims[3] = {static_cast<cuuint64_t>(tk::kKChunk), slabRows, slabs};
    cuuint64_t const strides[2] = {rowBytes, rowBytes * slabRows};
    cuuint32_t const box[3] = {static_cast<cuuint32_t>(tk::kKChunk), boxRows, boxSlabs};
    cuuint32_t const elementStrides[3] = {1, 1, 1};
    CUresult const result = encodeTiledFn()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(base), dims,
        strides, box, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(result == CUDA_SUCCESS,
        "[glm_moe_dsa_draft_routed_moe] cuTensorMapEncodeTiled failed: ", static_cast<int>(result));
    return cache.emplace(key, map).first->second;
}

} // namespace

//! Routed experts of the MTP draft layer (see draftRoutedMoe.h): native DeepSeek-V3
//! routing of ``router_logits`` with ``routing_bias``, FC1 + SwiGLU + FC2 + finalize on the FlashInfer TRT-LLM-Gen
//! BF16 layouts of ``w3_w1`` / ``w2``. Returns the finalized routed output ``[R, 6144]`` (BF16). ``h_scratch`` and
//! ``route_scratch`` are written (declared mutated in the schema) and reused by every call; calls sharing them must
//! be ordered on one stream. ``debug_fc2`` (optional, ``[R, 8, 6144]`` BF16)
//! receives the FC2 row of every (token, slot). ``timeline`` (optional, int64 ``[2, kTimelineCtas, kTimelineSlots]``)
//! receives the per-CTA event record of the TMA-fed kernels; only probe builds
//! (``GLM_MOE_DSA_DRAFT_ROUTED_MOE_TIMELINE=1``) accept it.
at::Tensor draftRoutedMoe(at::Tensor const& x, at::Tensor const& routerLogits, at::Tensor const& routingBias,
    at::Tensor const& w3w1, at::Tensor const& w2, at::Tensor const& hScratch, at::Tensor const& routeScratch,
    double routedScaling, int64_t fc1Ctas, int64_t fc1Warps, int64_t fc2Ctas, int64_t fc2Warps, int64_t flags,
    std::optional<at::Tensor> const& debugFc2, std::optional<at::Tensor> const& timeline)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2
            && x.size(1) == tk::kHidden,
        "[glm_moe_dsa_draft_routed_moe] x must be a contiguous CUDA bf16 tensor [R, ", tk::kHidden, "]");
    int64_t const rows = x.size(0);
    TORCH_CHECK(rows >= 1 && rows <= tk::kMaxRows, "[glm_moe_dsa_draft_routed_moe] rows ", rows, " outside [1, ",
        tk::kMaxRows, "]");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 16 == 0,
        "[glm_moe_dsa_draft_routed_moe] x must be 16-byte aligned");
    auto const device = x.get_device();
    checkTensor(routerLogits, "router_logits", at::kFloat, {rows, tk::kNumExperts}, device);
    checkTensor(routingBias, "routing_bias", at::kBFloat16, {tk::kNumExperts}, device);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(routerLogits.const_data_ptr()) % 16 == 0
            && reinterpret_cast<uintptr_t>(routingBias.const_data_ptr()) % 16 == 0,
        "[glm_moe_dsa_draft_routed_moe] router_logits and routing_bias must be 16-byte aligned");
    checkTensor(w3w1, "w3_w1", at::kBFloat16, {tk::kNumExperts, tk::kFc1Chunks, tk::kFc1Rows, tk::kKChunk}, device);
    checkTensor(w2, "w2", at::kBFloat16, {tk::kNumExperts, tk::kFc2Chunks, tk::kHidden, tk::kKChunk}, device);
    TORCH_CHECK(hScratch.is_cuda() && hScratch.get_device() == device && hScratch.scalar_type() == at::kBFloat16
            && hScratch.is_contiguous() && hScratch.numel() >= tk::kScratchHElems,
        "[glm_moe_dsa_draft_routed_moe] h_scratch must be a contiguous CUDA bf16 tensor with at least ",
        tk::kScratchHElems, " elements");
    TORCH_CHECK(routeScratch.is_cuda() && routeScratch.get_device() == device && routeScratch.scalar_type() == at::kInt
            && routeScratch.is_contiguous() && routeScratch.numel() >= tk::kRouteWords,
        "[glm_moe_dsa_draft_routed_moe] route_scratch must be a contiguous CUDA int32 tensor with at least ",
        tk::kRouteWords, " elements");
    TORCH_CHECK(
        fc1Warps == 1 || fc1Warps == 2 || fc1Warps == 4, "[glm_moe_dsa_draft_routed_moe] fc1_warps must be 1, 2 or 4");
    TORCH_CHECK(fc2Warps == 1 || fc2Warps == 2 || fc2Warps == 4 || (fc2Warps == 3 && (flags & tk::kFlagTma)),
        "[glm_moe_dsa_draft_routed_moe] fc2_warps must be 1, 2 or 4 (or 3 with the TMA flag)");
    TORCH_CHECK(fc1Ctas >= 1 && fc1Ctas <= 1024 && fc2Ctas >= 1 && fc2Ctas <= 1024,
        "[glm_moe_dsa_draft_routed_moe] grid sizes");
    if (debugFc2.has_value())
    {
        checkTensor(debugFc2.value(), "debug_fc2", at::kBFloat16, {rows, tk::kTopK, tk::kHidden}, device);
    }
    if (timeline.has_value())
    {
        TORCH_CHECK(kTimelineBuild,
            "[glm_moe_dsa_draft_routed_moe] timeline needs a build with GLM_MOE_DSA_DRAFT_ROUTED_MOE_TIMELINE=1");
        checkTensor(timeline.value(), "timeline", at::kLong, {2, tk::kTimelineCtas, tk::kTimelineSlots}, device);
    }

    c10::cuda::CUDAGuard const deviceGuard(x.device());
    at::Tensor out = at::empty({rows, tk::kHidden}, x.options());

    tk::DraftRoutedMoeParams params{};
    params.logits = routerLogits.const_data_ptr<float>();
    params.bias = reinterpret_cast<__nv_bfloat16 const*>(routingBias.const_data_ptr());
    params.x = reinterpret_cast<__nv_bfloat16 const*>(x.const_data_ptr());
    params.w13 = reinterpret_cast<__nv_bfloat16 const*>(w3w1.const_data_ptr());
    params.w2 = reinterpret_cast<__nv_bfloat16 const*>(w2.const_data_ptr());
    params.out = reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr());
    params.h = reinterpret_cast<__nv_bfloat16*>(hScratch.mutable_data_ptr());
    params.route = routeScratch.mutable_data_ptr<int32_t>();
    params.debugFc2 = debugFc2.has_value() ? reinterpret_cast<__nv_bfloat16*>(debugFc2->mutable_data_ptr()) : nullptr;
    params.timeline = timeline.has_value() ? timeline->mutable_data_ptr<int64_t>() : nullptr;
    params.rows = static_cast<int>(rows);
    params.routedScaling = static_cast<float>(routedScaling);
    params.fc1Ctas = static_cast<int>(fc1Ctas);
    params.fc1Warps = static_cast<int>(fc1Warps);
    params.fc2Ctas = static_cast<int>(fc2Ctas);
    params.fc2Warps = static_cast<int>(fc2Warps);
    params.flags = static_cast<int>(flags);

    tk::DraftRoutedMoeTensorMaps maps{};
    bool const tma = (flags & tk::kFlagTma) != 0;
    if (tma)
    {
        int const f = static_cast<int>(flags);
        int const w1 = static_cast<int>(fc1Warps);
        maps.w13 = weightMap(device, w3w1.const_data_ptr(), tk::kFc1Rows,
            static_cast<uint64_t>(tk::kNumExperts) * tk::kFc1Chunks, static_cast<uint32_t>(tk::fc1TmaBoxRows(f, w1)),
            static_cast<uint32_t>(tk::fc1TmaBoxSlabs(f, w1)));
        maps.w2 = weightMap(device, w2.const_data_ptr(), tk::kHidden,
            static_cast<uint64_t>(tk::kNumExperts) * tk::kFc2Chunks, static_cast<uint32_t>(16 * fc2Warps),
            static_cast<uint32_t>(tk::kFc2Chunks));
        // The triples path of the FC1 kernel (4 MMA warps) reads 64-row boxes; otherwise the map is not read.
        maps.w13Triple = ((f & tk::kFlagFc1Triples) && w1 == 4) ? weightMap(device, w3w1.const_data_ptr(), tk::kFc1Rows,
                             static_cast<uint64_t>(tk::kNumExperts) * tk::kFc1Chunks,
                             static_cast<uint32_t>(tk::kTripleRows), static_cast<uint32_t>(tk::kTripleStageChunks))
                                                                : maps.w13;
        // kFlagStreamX: the call's activation as [token][K chunk][64] (boxes of 8 tokens; rows beyond the call's are
        // zero-filled and only feed discarded MMA columns). Encoded per call (``x`` changes between calls).
        maps.x = maps.w13;
        if (f & tk::kFlagStreamX)
        {
            cuuint64_t const dims[3] = {static_cast<cuuint64_t>(tk::kKChunk), static_cast<cuuint64_t>(tk::kFc1Chunks),
                static_cast<cuuint64_t>(rows)};
            cuuint64_t const strides[2]
                = {static_cast<cuuint64_t>(tk::kKChunk) * 2, static_cast<cuuint64_t>(tk::kHidden) * 2};
            cuuint32_t const box[3] = {static_cast<cuuint32_t>(tk::kKChunk), static_cast<cuuint32_t>(tk::kXBoxChunks),
                static_cast<cuuint32_t>(tk::kMaxRows)};
            cuuint32_t const elementStrides[3] = {1, 1, 1};
            CUresult const result
                = encodeTiledFn()(&maps.x, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(x.const_data_ptr()),
                    dims, strides, box, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                    CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            TORCH_CHECK(result == CUDA_SUCCESS,
                "[glm_moe_dsa_draft_routed_moe] cuTensorMapEncodeTiled (x) failed: ", static_cast<int>(result));
        }
    }
    cudaError_t const status
        = tk::invokeDraftRoutedMoe(params, tma ? &maps : nullptr, at::cuda::getCurrentCUDAStream(device));
    TORCH_CHECK(
        status == cudaSuccess, "[glm_moe_dsa_draft_routed_moe] kernel launch failed: ", cudaGetErrorString(status));
    return out;
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "draft_routed_moe(Tensor x, Tensor router_logits, Tensor routing_bias, Tensor w3_w1, Tensor w2, "
        "Tensor(a!) h_scratch, Tensor(b!) route_scratch, float routed_scaling, int fc1_ctas, int fc1_warps, "
        "int fc2_ctas, int fc2_warps, int flags, Tensor(c!)? debug_fc2=None, Tensor(d!)? timeline=None) -> Tensor");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("draft_routed_moe", &TRTLLM_MV2_NS::torch_ext::draftRoutedMoe);
}
