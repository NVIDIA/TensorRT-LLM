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
#include "tensorrt_llm/kernels/minimaxM3Sm107Tp4/drafterRopeFp8Kv.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <optional>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace drk = tensorrt_llm::kernels::minimaxM3Sm107Tp4::drafterRopeFp8Kv;

//! The MiniMax-M3 TP4 drafter layer's generation preprocess (kernels/minimaxM3Sm107Tp4/drafterRopeFp8Kv.h): RoPE on
//! the packed BF16 qkv rows, the E4M3 query into q_out, and the step's K / V rows into the paged E4M3 cache kv_pages
//! ([sub-pages, Hkv, tokens_per_block, 128]) at the sub-pages block_tables ([requests, 2, max blocks]) names.
void minimaxM3Sm107Tp4DrafterRopeFp8Kv(torch::Tensor const& qkv, torch::Tensor const& cosSin,
    torch::Tensor const& seqLens, std::optional<torch::Tensor> const& specPositionOffsets, torch::Tensor& kvPages,
    torch::Tensor const& blockTables, torch::Tensor& qOut, int64_t qLen, int64_t numHeadsQ, int64_t numHeadsKv,
    int64_t rotaryDim, int64_t tokensPerBlock, int64_t ropeMode, bool pdl, bool maxSharedCarveout)
{
    TORCH_CHECK(qkv.is_cuda() && qkv.is_contiguous() && qkv.dim() == 2 && qkv.scalar_type() == at::kBFloat16);
    TORCH_CHECK(qkv.size(1) == (numHeadsQ + 2 * numHeadsKv) * drk::kHeadDim, "qkv is not [tokens, (Hq + 2 Hkv) * 128]");
    TORCH_CHECK(qLen >= 1 && qkv.size(0) % qLen == 0, "the rows are not whole passes of q_len rows");
    TORCH_CHECK(rotaryDim > 0 && rotaryDim <= drk::kHeadDim && rotaryDim % 16 == 0, "unsupported rotary_dim");
    TORCH_CHECK(cosSin.is_cuda() && cosSin.is_contiguous() && cosSin.scalar_type() == at::kFloat
            && cosSin.numel() % rotaryDim == 0,
        "cos_sin must be the fp32 [positions, rotary_dim / 2] (cos, sin) table");
    int64_t const requests = qkv.size(0) / qLen;
    TORCH_CHECK(seqLens.is_cuda() && seqLens.scalar_type() == at::kInt && seqLens.numel() >= requests);
    if (specPositionOffsets.has_value())
    {
        TORCH_CHECK(specPositionOffsets->is_cuda() && specPositionOffsets->is_contiguous()
            && specPositionOffsets->scalar_type() == at::kInt && specPositionOffsets->numel() >= requests * qLen);
    }
    TORCH_CHECK(kvPages.is_cuda() && kvPages.is_contiguous() && kvPages.dim() == 4
            && kvPages.scalar_type() == at::kFloat8_e4m3fn && kvPages.size(1) == numHeadsKv
            && kvPages.size(2) == tokensPerBlock && kvPages.size(3) == drk::kHeadDim,
        "kv_pages must be the E4M3 [sub-pages, Hkv, tokens_per_block, 128] cache");
    TORCH_CHECK(blockTables.is_cuda() && blockTables.scalar_type() == at::kInt && blockTables.dim() == 3
            && blockTables.size(0) >= requests && blockTables.size(1) == 2 && blockTables.stride(2) == 1,
        "block_tables must be the int32 [requests, 2, max blocks] K / V sub-page table");
    TORCH_CHECK(qOut.is_cuda() && qOut.is_contiguous() && qOut.scalar_type() == at::kFloat8_e4m3fn
        && qOut.numel() == qkv.size(0) * numHeadsQ * drk::kHeadDim);
    c10::cuda::CUDAGuard const deviceGuard{qkv.device()};
    drk::Params params{};
    params.qkv = reinterpret_cast<__nv_bfloat16 const*>(qkv.data_ptr());
    params.cosSin = reinterpret_cast<float2 const*>(cosSin.data_ptr<float>());
    params.seqLens = seqLens.data_ptr<int>();
    params.specPositionOffsets = specPositionOffsets.has_value() ? specPositionOffsets->data_ptr<int>() : nullptr;
    params.qOut = reinterpret_cast<__nv_fp8_e4m3*>(qOut.data_ptr());
    params.pool = reinterpret_cast<__nv_fp8_e4m3*>(kvPages.data_ptr());
    params.blockTable = blockTables.data_ptr<int>();
    params.blockTableRequestStride = blockTables.stride(0);
    params.blockTableKvStride = blockTables.stride(1);
    params.numTokens = static_cast<int>(qkv.size(0));
    params.qLen = static_cast<int>(qLen);
    params.numHeadsQ = static_cast<int>(numHeadsQ);
    params.numHeadsKv = static_cast<int>(numHeadsKv);
    params.rotaryDim = static_cast<int>(rotaryDim);
    params.tokensPerBlock = static_cast<int>(tokensPerBlock);
    params.ropeMode = static_cast<int>(ropeMode);
    cudaError_t const err
        = drk::launch(params, pdl, maxSharedCarveout, at::cuda::getCurrentCUDAStream(qkv.get_device()));
    TORCH_CHECK(err == cudaSuccess, "minimax_m3_sm107tp4_drafter_rope_fp8_kv: ", cudaGetErrorString(err));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m3_sm107tp4_drafter_rope_fp8_kv(Tensor qkv, Tensor cos_sin, Tensor seq_lens, "
        "Tensor? spec_position_offsets, Tensor(a!) kv_pages, Tensor block_tables, Tensor(b!) q_out, int q_len, "
        "int num_heads_q, int num_heads_kv, int rotary_dim, int tokens_per_block, int rope_mode, bool pdl, bool "
        "max_shared_carveout=True) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_sm107tp4_drafter_rope_fp8_kv", &tensorrt_llm::torch_ext::minimaxM3Sm107Tp4DrafterRopeFp8Kv);
}
