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
#include "tensorrt_llm/kernels/minimaxM3/producerPw.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <initializer_list>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace pw = tensorrt_llm::kernels::minimaxM3::producerPw;

constexpr int64_t kHeadDim = 128;
constexpr int64_t kPageSize = 128;

void checkCuda(torch::Tensor const& t, char const* name)
{
    TORCH_CHECK(t.is_cuda(), "producer_pw: ", name, " must be a CUDA tensor");
}

} // namespace

//! The NVFP4 QKV + indexer norm/RoPE/KV-insert producer (same arguments and outputs as
//! minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert for 1 KV head, 1 index head, head_dim 128, rotary_dim 64):
//! returns (q FP8 [T, headsQ, 128], index_q FP8 [T, 1, 128]) and writes the NVFP4 K/V and FP8 index-K of each token's
//! cache slot. `early` triggers the dependents on entry.
std::vector<torch::Tensor> minimaxM3ProducerPw(torch::Tensor const& packed, torch::Tensor& kvData,
    torch::Tensor& kvScale, torch::Tensor& indexK, torch::Tensor const& outCacheLoc, torch::Tensor const& kvQuantScale,
    double eps, torch::Tensor const& qWeight, torch::Tensor const& kWeight, torch::Tensor const& indexQWeight,
    torch::Tensor const& indexKWeight, torch::Tensor const& rotaryCosSin, torch::Tensor const& positionIds,
    int64_t headsQ, bool pdl, bool early)
{
    checkCuda(packed, "packed");
    for (torch::Tensor const* t : std::initializer_list<torch::Tensor const*>{&kvData, &kvScale, &indexK, &outCacheLoc,
             &kvQuantScale, &qWeight, &kWeight, &indexQWeight, &indexKWeight, &rotaryCosSin, &positionIds})
    {
        TORCH_CHECK(t->is_cuda() && t->get_device() == packed.get_device(),
            "producer_pw: every operand must be on packed's CUDA device");
    }
    TORCH_CHECK(reinterpret_cast<uintptr_t>(packed.data_ptr()) % 8 == 0
            && reinterpret_cast<uintptr_t>(kvData.data_ptr()) % 2 == 0
            && reinterpret_cast<uintptr_t>(indexK.data_ptr()) % 4 == 0,
        "producer_pw: packed and the caches must start at the vector-load / store alignment");
    c10::cuda::CUDAGuard const deviceGuard{packed.device()};
    static int const sSmVersion = tensorrt_llm::common::getSMVersion();
    TORCH_CHECK(sSmVersion >= 100, "minimax_m3_producer_pw requires SM100 or newer");
    TORCH_CHECK(headsQ == 8 || headsQ == 16, "producer_pw: 8 or 16 query heads");
    int64_t const totalHeads = headsQ + 4;
    TORCH_CHECK(packed.scalar_type() == at::kBFloat16 && packed.dim() == 2 && packed.size(1) == totalHeads * kHeadDim
            && packed.is_contiguous(),
        "producer_pw: packed must be contiguous BF16 [T, (headsQ + 4) * 128]");
    // The kernel's layout (the native producer's checkMinimaxM3Nvfp4HndKVPools at one KV head): HND pools
    // [pages, 2 planes, 1 head, 128 tokens, 64 data / 8 scale bytes], dense token rows and head blocks, planes and
    // pages that do not overlap.
    TORCH_CHECK(kvData.scalar_type() == at::kByte && kvData.dim() == 5 && kvData.size(0) > 0 && kvData.size(1) == 2
            && kvData.size(2) == 1 && kvData.size(3) == kPageSize && kvData.size(4) == kHeadDim / 2,
        "producer_pw: kv_data must be the NVFP4 pool [pages, 2, 1, 128, 64] uint8");
    TORCH_CHECK(kvScale.scalar_type() == at::kByte && kvScale.dim() == 5 && kvScale.size(0) == kvData.size(0)
            && kvScale.size(1) == 2 && kvScale.size(2) == 1 && kvScale.size(3) == kPageSize
            && kvScale.size(4) == kHeadDim / 16,
        "producer_pw: kv_scale must be the E4M3 scale pool [pages, 2, 1, 128, 8] uint8");
    TORCH_CHECK(kvData.stride(4) == 1 && kvData.stride(3) == kvData.size(4)
            && kvData.stride(2) == kvData.size(3) * kvData.stride(3),
        "producer_pw: kv_data must have dense token rows and head blocks");
    TORCH_CHECK(kvScale.stride(4) == 1 && kvScale.stride(3) == kvScale.size(4)
            && kvScale.stride(2) == kvScale.size(3) * kvScale.stride(3),
        "producer_pw: kv_scale must have dense token rows and head blocks");
    TORCH_CHECK(kvData.stride(1) >= kvData.size(2) * kvData.stride(2) && kvData.stride(0) >= 2 * kvData.stride(1)
            && kvScale.stride(1) >= kvScale.size(2) * kvScale.stride(2) && kvScale.stride(0) >= 2 * kvScale.stride(1),
        "producer_pw: pool planes or pages overlap");
    TORCH_CHECK(kvData.stride(0) % 2 == 0 && kvData.stride(1) % 2 == 0,
        "producer_pw: kv_data page and plane strides must keep the uint16 stores aligned");
    TORCH_CHECK(indexK.scalar_type() == at::kFloat8_e4m3fn && indexK.dim() == 4 && indexK.size(0) == kvData.size(0)
            && indexK.size(1) == 1 && indexK.size(2) == kPageSize && indexK.size(3) == kHeadDim && indexK.stride(3) == 1
            && indexK.stride(2) == kHeadDim && indexK.stride(0) >= kPageSize * kHeadDim && indexK.stride(0) % 4 == 0,
        "producer_pw: index_k must be FP8 [pages, 1, 128, 128] with a page stride keeping the 32-bit stores aligned");
    int64_t const numTokens = packed.size(0);
    TORCH_CHECK(
        outCacheLoc.scalar_type() == at::kInt && outCacheLoc.is_contiguous() && outCacheLoc.numel() >= numTokens,
        "producer_pw: out_cache_loc must be int32 [>= T]");
    TORCH_CHECK(
        positionIds.scalar_type() == at::kInt && positionIds.is_contiguous() && positionIds.numel() >= numTokens,
        "producer_pw: position_ids must be int32 [>= T]");
    TORCH_CHECK(kvQuantScale.scalar_type() == at::kFloat && kvQuantScale.is_contiguous() && kvQuantScale.numel() >= 3,
        "producer_pw: kv_quant_scale must be fp32 [>= 3]");
    TORCH_CHECK(rotaryCosSin.scalar_type() == at::kFloat && rotaryCosSin.is_contiguous() && rotaryCosSin.dim() == 3
            && rotaryCosSin.size(1) == 2 && rotaryCosSin.size(2) == 32
            && reinterpret_cast<uintptr_t>(rotaryCosSin.data_ptr()) % 16 == 0,
        "producer_pw: rotary_cos_sin must be fp32 [positions, 2, 32], 16-byte aligned (read as float4)");
    for (auto const* w : {&qWeight, &kWeight, &indexQWeight, &indexKWeight})
    {
        TORCH_CHECK(w->scalar_type() == at::kBFloat16 && w->is_contiguous() && w->numel() == kHeadDim
                && reinterpret_cast<uintptr_t>(w->data_ptr()) % 8 == 0,
            "producer_pw: norm weights must be BF16 [128], 8-byte aligned (read as 4-value words)");
    }
    auto const fp8 = packed.options().dtype(at::kFloat8_e4m3fn);
    auto qOut = torch::empty({numTokens, headsQ, kHeadDim}, fp8);
    auto indexQOut = torch::empty({numTokens, 1, kHeadDim}, fp8);
    if (numTokens == 0)
    {
        return {qOut, indexQOut};
    }
    pw::Params p{};
    p.packed = reinterpret_cast<__nv_bfloat16 const*>(packed.data_ptr());
    p.qOut = reinterpret_cast<__nv_fp8_e4m3*>(qOut.data_ptr());
    p.indexQOut = reinterpret_cast<__nv_fp8_e4m3*>(indexQOut.data_ptr());
    p.kvData = kvData.data_ptr<uint8_t>();
    p.kvScale = kvScale.data_ptr<uint8_t>();
    p.indexK = reinterpret_cast<__nv_fp8_e4m3*>(indexK.data_ptr());
    p.outCacheLoc = outCacheLoc.data_ptr<int>();
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
    p.numTokens = static_cast<int>(numTokens);
    p.eps = static_cast<float>(eps);
    p.qWeight = reinterpret_cast<__nv_bfloat16 const*>(qWeight.data_ptr());
    p.kWeight = reinterpret_cast<__nv_bfloat16 const*>(kWeight.data_ptr());
    p.indexQWeight = reinterpret_cast<__nv_bfloat16 const*>(indexQWeight.data_ptr());
    p.indexKWeight = reinterpret_cast<__nv_bfloat16 const*>(indexKWeight.data_ptr());
    p.rotaryCosSin = rotaryCosSin.data_ptr<float>();
    p.positionIds = positionIds.data_ptr<int>();
    cudaError_t const err = pw::invokeProducerPw(p, static_cast<int>(numTokens), static_cast<int>(headsQ), pdl, early,
        at::cuda::getCurrentCUDAStream(packed.get_device()));
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TORCH_CHECK(false, "producer_pw launch: ", cudaGetErrorString(err));
    }
    return {qOut, indexQOut};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m3_producer_pw(Tensor packed, Tensor(a!) kv_data, Tensor(b!) kv_scale, Tensor(c!) index_k, "
        "Tensor out_cache_loc, Tensor kv_quant_scale, float eps, Tensor q_weight, Tensor k_weight, "
        "Tensor index_q_weight, Tensor index_k_weight, Tensor rotary_cos_sin, Tensor position_ids, int heads_q, "
        "bool pdl, bool early) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_producer_pw", &tensorrt_llm::torch_ext::minimaxM3ProducerPw);
}
