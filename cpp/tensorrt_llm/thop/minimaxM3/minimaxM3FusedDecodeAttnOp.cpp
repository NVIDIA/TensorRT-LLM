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

#include "tensorrt_llm/kernels/minimaxM3/fusedDecodeAttn.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

using tensorrt_llm::kernels::MinimaxM3FusedDecodeAttnParams;

constexpr int64_t kNumHeads = 16;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kPackedWidth = (kNumHeads + 4) * kHeadDim;

void checkByteTensor(at::Tensor const& t, char const* name)
{
    TORCH_CHECK(t.is_cuda() && t.element_size() == 1, name, " must be a CUDA tensor of one-byte elements");
}

} // namespace

//! Fused per-head Gemma RMSNorm + partial NeoX RoPE + NVFP4 K/V and FP8 index-K cache insert + identity-selection
//! sparse decode attention over the paged NVFP4 cache + split-K merge, for the MiniMax-M3 MSA sparse layers in
//! short-sequence decode (every request attends at most 16 KV blocks of 128 tokens).
//!
//! @param packed bf16 [T, 2560] QKV+index projection rows ([Q x16 | K | V | idxQ | idxK] x 128), T = R * qlen.
//! @param kvData uint8 [pages, 2, 1, 128, 64] HND NVFP4 pool (K plane 0, V plane 1); the new tokens are written.
//! @param kvScale uint8 [pages, 2, 1, 128, 8] E4M3 block scales (K linear, V 4x4 token quads); written.
//! @param idxK one-byte [pages, 1, 128, 128] FP8 index-K cache; written.
//! @param output bf16 [T, 2048] attention output; mxfp8Data / mxfp8Scale its MXFP8 copy (E4M3 rows and 128x4-swizzled
//! UE8M0 scales) for the o_proj GEMM.
void minimaxM3FusedDecodeAttn(at::Tensor const& packed, at::Tensor& kvData, at::Tensor& kvScale, at::Tensor& idxK,
    at::Tensor const& outCacheLoc, at::Tensor const& positionIds, at::Tensor const& blockTable,
    at::Tensor const& seqLens, at::Tensor const& kvQuantScale, at::Tensor const& kGlobalScale,
    at::Tensor const& vGlobalScale, at::Tensor const& qNormWeight, at::Tensor const& kNormWeight,
    at::Tensor const& indexKNormWeight, at::Tensor const& rotaryCosSin, double eps, double smScale,
    int64_t decodeQueryLen, int64_t tokensPerCluster, int64_t blocksPerCta, at::Tensor& output, at::Tensor& mxfp8Data,
    at::Tensor& mxfp8Scale, bool usePdl)
{
    TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == at::kBFloat16 && packed.dim() == 2
            && packed.size(1) == kPackedWidth && packed.stride(1) == 1,
        "packed must be CUDA bf16 [T, 2560] with contiguous rows");
    checkByteTensor(kvData, "kv_data");
    checkByteTensor(kvScale, "kv_scale");
    checkByteTensor(idxK, "idx_k");
    TORCH_CHECK(
        kvData.dim() == 5 && kvScale.dim() == 5 && idxK.dim() == 4, "kv_data, kv_scale and idx_k must be paged");
    for (auto const* t : {&outCacheLoc, &positionIds, &blockTable, &seqLens})
    {
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kInt,
            "slot, position, block-table and length tensors "
            "must be CUDA int32");
    }
    TORCH_CHECK(blockTable.dim() == 2 && blockTable.stride(1) == 1, "block_table must be row-major 2-D");
    for (auto const* t : {&kvQuantScale, &kGlobalScale, &vGlobalScale, &rotaryCosSin})
    {
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kFloat, "scales and rotary_cos_sin must be CUDA fp32");
    }
    for (auto const* t : {&qNormWeight, &kNormWeight, &indexKNormWeight})
    {
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kBFloat16 && t->numel() == kHeadDim,
            "norm weights must be CUDA bf16 [128]");
    }
    TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16 && output.dim() == 2
            && output.size(1) == kNumHeads * kHeadDim && output.stride(1) == 1,
        "output must be CUDA bf16 [T, 2048] with contiguous rows");
    checkByteTensor(mxfp8Data, "mxfp8_data");
    checkByteTensor(mxfp8Scale, "mxfp8_scale");
    int64_t const numTokens = packed.size(0);
    int64_t const numReqs = seqLens.size(0);
    TORCH_CHECK(numTokens == numReqs * decodeQueryLen, "packed rows must equal num_requests * decode_query_len");
    TORCH_CHECK(output.size(0) >= numTokens && mxfp8Data.dim() == 2 && mxfp8Data.size(0) >= numTokens
            && mxfp8Data.size(1) == kNumHeads * kHeadDim && mxfp8Data.is_contiguous() && mxfp8Scale.is_contiguous(),
        "mxfp8_data must be contiguous [>= T, 2048]");
    TORCH_CHECK(tensorrt_llm::kernels::minimaxM3FusedDecodeAttnSupports(static_cast<int>(decodeQueryLen),
                    static_cast<int>(tokensPerCluster), static_cast<int>(blocksPerCta)),
        "minimax_m3_fused_decode_attn: geometry (", decodeQueryLen, ", ", tokensPerCluster, ", ", blocksPerCta,
        ") is not compiled");
    if (numTokens == 0)
    {
        return;
    }

    c10::cuda::CUDAGuard const deviceGuard{packed.device()};
    auto const* prop = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(prop->major == 10, "minimax_m3_fused_decode_attn requires an sm_100-family GPU, got sm_", prop->major,
        prop->minor);

    MinimaxM3FusedDecodeAttnParams prm = {};
    prm.packed = reinterpret_cast<__nv_bfloat16 const*>(packed.data_ptr());
    prm.packed_stride = packed.stride(0);
    prm.kv_data = reinterpret_cast<uint8_t*>(kvData.data_ptr());
    prm.d_page = kvData.stride(0);
    prm.d_plane = kvData.stride(1);
    prm.kv_scale = reinterpret_cast<uint8_t*>(kvScale.data_ptr());
    prm.s_page = kvScale.stride(0);
    prm.s_plane = kvScale.stride(1);
    prm.idx_k = reinterpret_cast<uint8_t*>(idxK.data_ptr());
    prm.i_page = idxK.stride(0);
    prm.i_tok = idxK.stride(2);
    prm.num_pages = kvData.size(0);
    prm.out_cache_loc = outCacheLoc.data_ptr<int>();
    prm.position_ids = positionIds.data_ptr<int>();
    prm.block_table = blockTable.data_ptr<int>();
    prm.bt_stride = blockTable.stride(0);
    prm.bt_cols = static_cast<int>(blockTable.size(1));
    prm.seq_lens = seqLens.data_ptr<int>();
    prm.kv_quant_scale = kvQuantScale.data_ptr<float>();
    prm.k_gs = kGlobalScale.data_ptr<float>();
    prm.v_gs = vGlobalScale.data_ptr<float>();
    prm.q_w = reinterpret_cast<__nv_bfloat16 const*>(qNormWeight.data_ptr());
    prm.k_w = reinterpret_cast<__nv_bfloat16 const*>(kNormWeight.data_ptr());
    prm.ik_w = reinterpret_cast<__nv_bfloat16 const*>(indexKNormWeight.data_ptr());
    prm.rope = rotaryCosSin.data_ptr<float>();
    prm.eps = static_cast<float>(eps);
    prm.sm_scale = static_cast<float>(smScale);
    prm.out = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
    prm.out_stride = output.stride(0);
    prm.q8 = reinterpret_cast<uint8_t*>(mxfp8Data.data_ptr());
    prm.sf = reinterpret_cast<uint8_t*>(mxfp8Scale.data_ptr());
    prm.num_sf_k_tiles = static_cast<int>(output.size(1) / 128);

    cudaError_t const err = tensorrt_llm::kernels::invokeMinimaxM3FusedDecodeAttn(prm, static_cast<int>(decodeQueryLen),
        static_cast<int>(tokensPerCluster), static_cast<int>(blocksPerCta), static_cast<int>(numReqs), usePdl,
        at::cuda::getCurrentCUDAStream(packed.get_device()));
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TORCH_CHECK(false, "minimax_m3_fused_decode_attn launch failed (qlen=", decodeQueryLen,
            " qg=", tokensPerCluster, " bpc=", blocksPerCta, "): ", cudaGetErrorString(err));
    }
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m3_fused_decode_attn(Tensor packed, Tensor(a!) kv_data, Tensor(b!) kv_scale, Tensor(c!) idx_k, "
        "Tensor out_cache_loc, Tensor position_ids, Tensor block_table, Tensor seq_lens, Tensor kv_quant_scale, "
        "Tensor k_global_scale, Tensor v_global_scale, Tensor q_norm_weight, Tensor k_norm_weight, "
        "Tensor index_k_norm_weight, Tensor rotary_cos_sin, float eps, float sm_scale, int decode_query_len, "
        "int tokens_per_cluster, int blocks_per_cta, Tensor(d!) output, Tensor(e!) mxfp8_data, "
        "Tensor(f!) mxfp8_scale, bool use_pdl) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_fused_decode_attn", &tensorrt_llm::torch_ext::minimaxM3FusedDecodeAttn);
}
