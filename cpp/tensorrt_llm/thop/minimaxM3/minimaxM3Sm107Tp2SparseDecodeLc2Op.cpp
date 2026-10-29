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

#include "tensorrt_llm/kernels/minimaxM3Sm107Tp2/sparseDecodeLc2.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

constexpr int64_t kHeadsPerKv = 16;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kTopK = 16;

void checkBytes(torch::Tensor const& t, char const* name)
{
    TORCH_CHECK(t.is_cuda() && t.element_size() == 1, "minimax_m3_sm107_tp2_sparse_decode_lc: ", name,
        " must be a one-byte CUDA view (uint8 / e4m3 / packed e2m1)");
}

} // namespace

//! One-launch MiniMax-M3 long-context sparse decode over the selected NVFP4 blocks, SM107 TP2 geometry (16 query
//! heads per KV head, 2 KV heads per rank, head_dim 128, page 128, top-k 16).
//!
//! @param q E4M3 bytes [T, 16 * H, 128], unit stride in the last dim.
//! @param k, v Packed E2M1 bytes [pages, H, 128, 64] (the per-layer HND views).
//! @param kBlockScale, vBlockScale E4M3 bytes [pages, H, 128, 8]: K token-major, V in 4x4 token-quad order.
//! @param kGlobalScale, vGlobalScale One-element float32 per-tensor dequant scales.
//! @param topk Int32 [H, T, 16] ascending selected block ids with -1 tail (the head-major selector table).
//! @param out BF16 [T, H * 2048], written. q8 / sf, when given, receive the MXFP8 copy and its swizzled UE8M0 scales.
//! @param n1Ids Optional int32 [H, rows, 16] (unit last-dim stride): this layer's selected block ids at its previous
//! decode step, read to prefetch those blocks into L2 before the grid dependency and overwritten with this step's;
//! query rows at or past `rows` skip it. Initialize to -1. A hint: `out` does not depend on it.
void minimaxM3Sm107Tp2SparseDecodeLc(torch::Tensor const& q, torch::Tensor const& k, torch::Tensor const& v,
    torch::Tensor const& kBlockScale, torch::Tensor const& vBlockScale, torch::Tensor const& kGlobalScale,
    torch::Tensor const& vGlobalScale, torch::Tensor const& topk, torch::Tensor const& blockTable,
    torch::Tensor const& seqLens, torch::Tensor& out, std::optional<torch::Tensor> q8, std::optional<torch::Tensor> sf,
    int64_t numSfKTiles, int64_t decodeQueryLen, double smScale, bool linearSoftmax, bool usePdl, bool prewait,
    std::optional<torch::Tensor> n1Ids)
{
    TORCH_CHECK(q.is_cuda(), "minimax_m3_sm107_tp2_sparse_decode_lc: q must be a CUDA tensor");
    c10::cuda::CUDAGuard const deviceGuard{q.device()};
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 10,
        "minimax_m3_sm107_tp2_sparse_decode_lc requires an SM100-family GPU");
    checkBytes(q, "q");
    checkBytes(k, "k");
    checkBytes(v, "v");
    checkBytes(kBlockScale, "k_block_scale");
    checkBytes(vBlockScale, "v_block_scale");
    TORCH_CHECK(k.dim() == 4 && k.size(2) == 128 && k.size(3) == kHeadDim / 2 && k.stride(3) == 1
            && v.sizes() == k.sizes() && v.strides() == k.strides(),
        "minimax_m3_sm107_tp2_sparse_decode_lc: k and v must be [pages, H, 128, 64] with matching strides");
    int64_t const kvHeads = k.size(1);
    TORCH_CHECK(kBlockScale.dim() == 4 && kBlockScale.size(1) == kvHeads && kBlockScale.size(3) == 8
            && kBlockScale.stride(3) == 1 && kBlockScale.stride(2) == 8 && vBlockScale.sizes() == kBlockScale.sizes()
            && vBlockScale.strides() == kBlockScale.strides(),
        "minimax_m3_sm107_tp2_sparse_decode_lc: block scales must be contiguous [pages, H, 128, 8] per page");
    TORCH_CHECK(q.dim() == 3 && q.size(1) == kHeadsPerKv * kvHeads && q.size(2) == kHeadDim && q.stride(2) == 1,
        "minimax_m3_sm107_tp2_sparse_decode_lc: q must be [T, 16 * H, 128] with unit last-dim stride");
    TORCH_CHECK(kGlobalScale.scalar_type() == torch::kFloat32 && vGlobalScale.scalar_type() == torch::kFloat32,
        "minimax_m3_sm107_tp2_sparse_decode_lc: global scales must be float32");
    TORCH_CHECK(topk.scalar_type() == torch::kInt32 && topk.dim() == 3 && topk.size(0) == kvHeads
            && topk.size(1) == q.size(0) && topk.size(2) == kTopK,
        "minimax_m3_sm107_tp2_sparse_decode_lc: topk must be int32 [H, T, 16]");
    TORCH_CHECK(blockTable.scalar_type() == torch::kInt32 && seqLens.scalar_type() == torch::kInt32,
        "minimax_m3_sm107_tp2_sparse_decode_lc: block_table and seq_lens must be int32");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.dim() == 2 && out.size(0) == q.size(0)
            && out.size(1) == kvHeads * kHeadsPerKv * kHeadDim && out.stride(1) == 1,
        "minimax_m3_sm107_tp2_sparse_decode_lc: out must be bfloat16 [T, H * 2048] with unit last-dim stride");
    TORCH_CHECK(
        q8.has_value() == sf.has_value(), "minimax_m3_sm107_tp2_sparse_decode_lc: pass both q8 and sf, or neither");
    bool const mxfp8 = q8.has_value();
    if (mxfp8)
    {
        checkBytes(*q8, "q8");
        checkBytes(*sf, "sf");
    }

    tensorrt_llm::kernels::MinimaxM3Sm107Tp2SparseDecodeLcParams prm{};
    prm.q = reinterpret_cast<uint8_t const*>(q.data_ptr());
    prm.q_tok = q.stride(0);
    prm.q_head = q.stride(1);
    prm.k = reinterpret_cast<uint8_t const*>(k.data_ptr());
    prm.v = reinterpret_cast<uint8_t const*>(v.data_ptr());
    prm.k_blk = k.stride(0);
    prm.v_blk = v.stride(0);
    prm.k_h = k.stride(1);
    prm.v_h = v.stride(1);
    prm.ksc = reinterpret_cast<uint8_t const*>(kBlockScale.data_ptr());
    prm.vsc = reinterpret_cast<uint8_t const*>(vBlockScale.data_ptr());
    prm.ksc_blk = kBlockScale.stride(0);
    prm.vsc_blk = vBlockScale.stride(0);
    prm.ksc_h = kBlockScale.stride(1);
    prm.vsc_h = vBlockScale.stride(1);
    prm.kgs = kGlobalScale.data_ptr<float>();
    prm.vgs = vGlobalScale.data_ptr<float>();
    prm.topk = topk.data_ptr<int>();
    prm.t_h = topk.stride(0);
    prm.t_tok = topk.stride(1);
    prm.t_k = topk.stride(2);
    prm.block_table = blockTable.data_ptr<int>();
    prm.bt_stride = blockTable.stride(0);
    prm.seq_lens = seqLens.data_ptr<int>();
    prm.dql = static_cast<int>(decodeQueryLen);
    prm.sm_scale = static_cast<float>(smScale);
    prm.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    prm.out_stride = out.stride(0);
    prm.q8 = mxfp8 ? reinterpret_cast<uint8_t*>(q8->data_ptr()) : nullptr;
    prm.sf = mxfp8 ? reinterpret_cast<uint8_t*>(sf->data_ptr()) : nullptr;
    prm.num_sf_k_tiles = static_cast<int>(numSfKTiles);
    prm.num_kv_heads = static_cast<int>(kvHeads);
    if (n1Ids.has_value())
    {
        TORCH_CHECK(n1Ids->is_cuda() && n1Ids->scalar_type() == torch::kInt32 && n1Ids->dim() == 3
                && n1Ids->size(0) == kvHeads && n1Ids->size(2) == kTopK && n1Ids->stride(2) == 1
                && n1Ids->stride(1) == kTopK,
            "minimax_m3_sm107_tp2_sparse_decode_lc: n1_ids must be int32 [H, rows, 16] with dense rows");
        prm.n1_ids = n1Ids->data_ptr<int>();
        prm.n1_h = n1Ids->stride(0);
        prm.n1_rows = static_cast<int>(n1Ids->size(1));
        prm.n1_pages = k.size(0);
    }

    auto const stream = at::cuda::getCurrentCUDAStream(q.get_device());
    cudaError_t const err = tensorrt_llm::kernels::invokeMinimaxM3Sm107Tp2SparseDecodeLc(
        prm, static_cast<int32_t>(q.size(0)), linearSoftmax, mxfp8, prewait, usePdl, stream);
    TORCH_CHECK(err == cudaSuccess, "minimax_m3_sm107_tp2_sparse_decode_lc launch failed: ", cudaGetErrorString(err));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m3_sm107_tp2_sparse_decode_lc(Tensor q, Tensor k, Tensor v, Tensor k_block_scale, "
        "Tensor v_block_scale, Tensor k_global_scale, Tensor v_global_scale, Tensor topk, Tensor block_table, "
        "Tensor seq_lens, Tensor(a!) out, Tensor(b!)? q8, Tensor(c!)? sf, int num_sf_k_tiles, int decode_query_len, "
        "float sm_scale, bool linear_softmax, bool use_pdl, bool prewait, Tensor(d!)? n1_ids=None) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_sm107_tp2_sparse_decode_lc", &tensorrt_llm::torch_ext::minimaxM3Sm107Tp2SparseDecodeLc);
}
