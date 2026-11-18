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

#include "tensorrt_llm/kernels/minimaxM3/attnC1.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace dims = tensorrt_llm::kernels::minimax_m3_attn_c1;
using tensorrt_llm::kernels::MinimaxM3AttnC1Params;

constexpr int64_t kHeadDim = 128;

void checkByteTensor(at::Tensor const& t, char const* name)
{
    TORCH_CHECK(t.is_cuda() && t.element_size() == 1, name, " must be a CUDA tensor of one-byte elements");
}

} // namespace

//! One launch for a one-request MiniMax-M3 MSA verify layer of 6 query tokens: the fused norm/RoPE/KV-insert +
//! identity-selection sparse decode + merge of minimax_m3_fused_decode_attn on the packed QKV rows, and the MXFP8
//! o_proj GEMV, 96 CTAs that are each an attention CTA and an o_proj unit (fed through the MXFP8 hand-off).
//!
//! @param packed bf16 [6, 2560] QKV+index projection rows.
//! @param flags int32 [>= 32 + 96]: the per-CTA call counters (zero-initialized once); written.
//! @param mxfp8Data / mxfp8Scale the two call-parity buffers of the attention's MXFP8 hand-off (0xFF at rest).
//! @param y bf16 [T, 6144] o_proj output. arWs / arFlags / arRank / arWorld / arMaxTokens / arMode: the next
//! multicast all-reduce's exchange (device addresses) to push y into when arMode is 2; 0 for none.
void minimaxM3AttnC1(at::Tensor const& packed, at::Tensor& flags, at::Tensor& kvData, at::Tensor& kvScale,
    at::Tensor& idxK, at::Tensor const& outCacheLoc, at::Tensor const& positionIds, at::Tensor const& blockTable,
    at::Tensor const& seqLens, at::Tensor const& kvQuantScale, at::Tensor const& kGlobalScale,
    at::Tensor const& vGlobalScale, at::Tensor const& qNormWeight, at::Tensor const& kNormWeight,
    at::Tensor const& indexKNormWeight, at::Tensor const& rotaryCosSin, double eps, double smScale,
    int64_t decodeQueryLen, int64_t tokensPerCluster, int64_t blocksPerCta, at::Tensor& output, at::Tensor& mxfp8Data,
    at::Tensor& mxfp8Scale, bool usePdl, at::Tensor const& oWeight, at::Tensor const& oWeightScale, at::Tensor& y,
    int64_t arWs, int64_t arFlags, int64_t arRank, int64_t arWorld, int64_t arMaxTokens, int64_t arMode)
{
    TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == at::kBFloat16 && packed.dim() == 2
            && packed.size(1) == 20 * kHeadDim && packed.stride(1) == 1,
        "packed must be CUDA bf16 [T, 2560] with contiguous rows");
    checkByteTensor(oWeight, "o_weight");
    checkByteTensor(oWeightScale, "o_weight_scale");
    checkByteTensor(kvData, "kv_data");
    checkByteTensor(kvScale, "kv_scale");
    checkByteTensor(idxK, "idx_k");
    checkByteTensor(mxfp8Data, "mxfp8_data");
    checkByteTensor(mxfp8Scale, "mxfp8_scale");
    TORCH_CHECK(
        oWeight.dim() == 2 && oWeight.size(0) == dims::kOpN && oWeight.size(1) == dims::kOpK && oWeight.is_contiguous(),
        "o_weight must be contiguous [6144, 2048]");
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
    TORCH_CHECK(flags.is_cuda() && flags.scalar_type() == at::kInt && flags.numel() >= 32 + dims::kCtas,
        "bad c1 flags: int32 [>= 128]");
    TORCH_CHECK(mxfp8Data.numel() == 2 * dims::kMaxTokens * dims::kOpK && mxfp8Scale.numel() == 2 * dims::kOpSfBytes
            && mxfp8Data.is_contiguous() && mxfp8Scale.is_contiguous(),
        "c1 MXFP8 hand-off");
    int64_t const numTokens = packed.size(0);
    int64_t const numReqs = seqLens.size(0);
    TORCH_CHECK(numTokens == numReqs * decodeQueryLen && numTokens <= dims::kMaxTokens, "token count");
    TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kBFloat16 && output.dim() == 2
            && output.size(0) >= numTokens && output.size(1) == 16 * kHeadDim && output.stride(1) == 1,
        "output must be CUDA bf16 [T, 2048] with contiguous rows");
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == at::kBFloat16 && y.dim() == 2 && y.size(0) == numTokens
            && y.size(1) == dims::kOpN && y.is_contiguous(),
        "y must be [M, 6144]");
    TORCH_CHECK(decodeQueryLen == 6 && tokensPerCluster == 1 && blocksPerCta == 1 && numReqs == 1,
        "minimax_m3_attn_c1 is compiled for one request with six query tokens (geometry (6, 1, 1)), got ", numReqs,
        " x (", decodeQueryLen, ", ", tokensPerCluster, ", ", blocksPerCta, ")");
    TORCH_CHECK(arMode == 0 || arMode == 2, "ar_mode must be 0 (none) or 2 (push)");

    c10::cuda::CUDAGuard const deviceGuard{packed.device()};
    auto const* prop = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(
        prop->major == 10, "minimax_m3_attn_c1 requires an sm_100-family GPU, got sm_", prop->major, prop->minor);

    MinimaxM3AttnC1Params prm = {};
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
    prm.packed = reinterpret_cast<__nv_bfloat16 const*>(packed.data_ptr());
    prm.packed_stride = packed.stride(0);
    prm.num_tokens = static_cast<int>(numTokens);
    prm.flags = reinterpret_cast<unsigned int*>(flags.data_ptr());
    prm.wsf_o = reinterpret_cast<uint8_t const*>(oWeightScale.data_ptr());
    prm.y_out = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
    prm.ar_ws = reinterpret_cast<int*>(arWs);
    prm.ar_flags = reinterpret_cast<int const*>(arFlags);
    prm.ar_rank = static_cast<int>(arRank);
    prm.ar_world = static_cast<int>(arWorld);
    prm.ar_maxt = static_cast<int>(arMaxTokens);
    prm.ar_mode = static_cast<int>(arMode);

    cudaError_t const err = tensorrt_llm::kernels::invokeMinimaxM3AttnC1(prm, oWeight.data_ptr(),
        static_cast<int>(decodeQueryLen), static_cast<int>(tokensPerCluster), static_cast<int>(blocksPerCta),
        static_cast<int>(numReqs), usePdl, at::cuda::getCurrentCUDAStream(packed.get_device()));
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        TORCH_CHECK(false, "minimax_m3_attn_c1 launch failed (qlen=", decodeQueryLen, " qg=", tokensPerCluster,
            " bpc=", blocksPerCta, "): ", cudaGetErrorString(err));
    }
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m3_attn_c1(Tensor packed, Tensor(a!) flags, Tensor(b!) kv_data, Tensor(c!) kv_scale, "
        "Tensor(d!) idx_k, Tensor out_cache_loc, Tensor position_ids, Tensor block_table, Tensor seq_lens, "
        "Tensor kv_quant_scale, Tensor k_global_scale, Tensor v_global_scale, Tensor q_norm_weight, "
        "Tensor k_norm_weight, Tensor index_k_norm_weight, Tensor rotary_cos_sin, float eps, float sm_scale, "
        "int decode_query_len, int tokens_per_cluster, int blocks_per_cta, Tensor(e!) output, Tensor(f!) mxfp8_data, "
        "Tensor(g!) mxfp8_scale, bool use_pdl, Tensor o_weight, Tensor o_weight_scale, Tensor(h!) y, "
        "int ar_workspace, int ar_flags, int ar_rank, int ar_world, int ar_max_tokens, int ar_mode) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_attn_c1", &tensorrt_llm::torch_ext::minimaxM3AttnC1);
}
