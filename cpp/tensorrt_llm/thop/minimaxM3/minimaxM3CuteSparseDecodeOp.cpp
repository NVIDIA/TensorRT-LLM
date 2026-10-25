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

#include "tensorrt_llm/kernels/minimaxM3/cuteSparseDecode.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

void checkBytes(torch::Tensor const& t, char const* name, int64_t innerStride)
{
    TORCH_CHECK(t.is_cuda() && t.element_size() == 1 && t.dim() == 4, "minimax_m3_cute_sparse_decode: ", name,
        " must be a one-byte 4-D CUDA view");
    TORCH_CHECK(t.stride(3) == 1 && t.stride(2) == innerStride && t.stride(1) % 16 == 0 && t.stride(0) % 16 == 0
            && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
        "minimax_m3_cute_sparse_decode: ", name, " (page, head) region must be contiguous and 16-byte aligned");
}

void checkPolicy(int64_t splits, int64_t stages, int64_t dequant)
{
    TORCH_CHECK((splits == 1 || splits == 2 || splits == 4 || splits == 8) && stages >= 2 && stages <= 4
            && (dequant == 0 || dequant == 1),
        "minimax_m3_cute_sparse_decode: unsupported policy (splits ", splits, ", stages ", stages, ", dequant ",
        dequant, ")");
}

} // namespace

//! MiniMax-M3 TP2 block-sparse verify decode over the selected NVFP4 pages, written into out in place.
//!
//! @param q E4M3 [total_q, H * 16, 128], unit last-dim stride.
//! @param k, v Packed E2M1 bytes [pages, H, 128, 64].
//! @param kBlockScale, vBlockScale E4M3 bytes [pages, H, 128, 8]: K token-major, V in 4x4 token-quad order.
//! @param kGlobalScale, vGlobalScale One-element float32 per-tensor dequant scales.
//! @param topk Int32 [H, total_q, k <= 32] selected block ids (any strides).
//! @param out BF16 [total_q, H * 16, 128].
void minimaxM3CuteSparseDecode(torch::Tensor const& q, torch::Tensor const& k, torch::Tensor const& v,
    torch::Tensor const& kBlockScale, torch::Tensor const& vBlockScale, torch::Tensor const& kGlobalScale,
    torch::Tensor const& vGlobalScale, torch::Tensor const& topk, torch::Tensor const& blockTable,
    torch::Tensor const& seqLens, torch::Tensor& out, int64_t decodeQueryLen, double smScale, int64_t splits,
    int64_t stages, int64_t dequant, bool usePdl)
{
    TORCH_CHECK(q.is_cuda(), "minimax_m3_cute_sparse_decode: q must be a CUDA tensor");
    c10::cuda::CUDAGuard const deviceGuard{q.device()};
    TORCH_CHECK(
        at::cuda::getCurrentDeviceProperties()->major == 10, "minimax_m3_cute_sparse_decode requires an SM10x GPU");
    checkPolicy(splits, stages, dequant);
    TORCH_CHECK(q.element_size() == 1 && q.dim() == 3 && q.size(2) == 128 && q.stride(2) == 1 && q.stride(1) % 16 == 0
            && q.stride(0) % 16 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0,
        "minimax_m3_cute_sparse_decode: q must be one-byte [total_q, H * 16, 128], 16-byte aligned");
    checkBytes(k, "k", 64);
    checkBytes(v, "v", 64);
    checkBytes(kBlockScale, "k_block_scale", 8);
    checkBytes(vBlockScale, "v_block_scale", 8);
    int64_t const numKvHeads = k.size(1);
    TORCH_CHECK(q.size(1) == 16 * numKvHeads && k.size(2) == 128,
        "minimax_m3_cute_sparse_decode: 16 q heads per KV head and page size 128 required");
    TORCH_CHECK(kGlobalScale.scalar_type() == torch::kFloat32 && vGlobalScale.scalar_type() == torch::kFloat32,
        "minimax_m3_cute_sparse_decode: global scales must be float32");
    TORCH_CHECK(topk.scalar_type() == torch::kInt32 && topk.dim() == 3 && topk.size(2) <= 32,
        "minimax_m3_cute_sparse_decode: topk must be int32 [H, total_q, k <= 32]");
    TORCH_CHECK(blockTable.scalar_type() == torch::kInt32 && blockTable.stride(1) == 1
            && seqLens.scalar_type() == torch::kInt32,
        "minimax_m3_cute_sparse_decode: block_table (inner stride 1) and seq_lens must be int32");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.dim() == 3 && out.stride(2) == 1 && out.stride(1) % 8 == 0
            && out.stride(0) % 8 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
        "minimax_m3_cute_sparse_decode: out must be bfloat16 [total_q, H * 16, 128], 16-byte aligned");
    int64_t const totalQ = q.size(0);
    TORCH_CHECK(decodeQueryLen > 0 && totalQ % decodeQueryLen == 0 && totalQ / decodeQueryLen <= seqLens.size(0),
        "minimax_m3_cute_sparse_decode: total_q must be a whole number of decode spans");

    tensorrt_llm::kernels::MinimaxM3CuteSparseDecodeParams p{};
    p.q = reinterpret_cast<uint8_t const*>(q.data_ptr());
    p.k = reinterpret_cast<uint8_t const*>(k.data_ptr());
    p.v = reinterpret_cast<uint8_t const*>(v.data_ptr());
    p.ksc = reinterpret_cast<uint8_t const*>(kBlockScale.data_ptr());
    p.vsc = reinterpret_cast<uint8_t const*>(vBlockScale.data_ptr());
    p.kgs = kGlobalScale.data_ptr<float>();
    p.vgs = vGlobalScale.data_ptr<float>();
    p.topk = topk.data_ptr<int32_t>();
    p.bt = blockTable.data_ptr<int32_t>();
    p.seq_lens = seqLens.data_ptr<int32_t>();
    p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.total_q = static_cast<int>(totalQ);
    p.num_kv_heads = static_cast<int>(numKvHeads);
    p.max_topk = static_cast<int>(topk.size(2));
    p.dql = static_cast<int>(decodeQueryLen);
    p.sm_scale = static_cast<float>(smScale);
    p.sq_n = q.stride(0);
    p.sq_h = q.stride(1);
    p.sk_blk = k.stride(0);
    p.sk_h = k.stride(1);
    p.sv_blk = v.stride(0);
    p.sv_h = v.stride(1);
    p.sks_blk = kBlockScale.stride(0);
    p.sks_h = kBlockScale.stride(1);
    p.svs_blk = vBlockScale.stride(0);
    p.svs_h = vBlockScale.stride(1);
    p.st_h = topk.stride(0);
    p.st_n = topk.stride(1);
    p.st_k = topk.stride(2);
    p.so_n = out.stride(0);
    p.so_h = out.stride(1);
    p.sbt_b = blockTable.stride(0);

    auto const stream = at::cuda::getCurrentCUDAStream(q.get_device());
    cudaError_t const err = tensorrt_llm::kernels::invokeMinimaxM3CuteSparseDecode(
        p, static_cast<int>(splits), static_cast<int>(stages), static_cast<int>(dequant), usePdl, stream);
    TORCH_CHECK(err == cudaSuccess, "minimax_m3_cute_sparse_decode launch failed: ", cudaGetErrorString(err));
}

//! [registers, local bytes, dynamic smem bytes, CTAs per SM, binary version] of one policy's kernel.
std::vector<int64_t> minimaxM3CuteSparseDecodeAttrs(int64_t splits, int64_t stages, int64_t dequant)
{
    checkPolicy(splits, stages, dequant);
    int out[5] = {0, 0, 0, 0, 0};
    cudaError_t const err = tensorrt_llm::kernels::minimaxM3CuteSparseDecodeAttrs(
        static_cast<int>(splits), static_cast<int>(stages), static_cast<int>(dequant), out);
    TORCH_CHECK(err == cudaSuccess, "minimax_m3_cute_sparse_decode_attrs failed: ", cudaGetErrorString(err));
    return std::vector<int64_t>(out, out + 5);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m3_cute_sparse_decode(Tensor q, Tensor k, Tensor v, Tensor k_block_scale, Tensor v_block_scale, "
        "Tensor k_global_scale, Tensor v_global_scale, Tensor topk, Tensor block_table, Tensor seq_lens, "
        "Tensor(a!) out, int decode_query_len, float sm_scale, int splits, int stages, int dequant, bool use_pdl) "
        "-> ()");
    m.def("minimax_m3_cute_sparse_decode_attrs(int splits, int stages, int dequant) -> int[]",
        &tensorrt_llm::torch_ext::minimaxM3CuteSparseDecodeAttrs);
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_cute_sparse_decode", &tensorrt_llm::torch_ext::minimaxM3CuteSparseDecode);
}
