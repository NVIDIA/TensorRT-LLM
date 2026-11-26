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

#include "tensorrt_llm/kernels/minimaxM3/msaKvUpdate.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

void checkInt32Cuda(torch::Tensor const& t, char const* name)
{
    TORCH_CHECK(t.is_cuda(), "minimax_m3_msa_kv_update: ", name, " must be a CUDA tensor");
    TORCH_CHECK(
        t.scalar_type() == torch::kInt32, "minimax_m3_msa_kv_update: ", name, " must be int32, got ", t.scalar_type());
}

} // namespace

//! MiniMax-M3 MSA KV-length correction after the overlap scheduler moved kv_lens (see invokeMinimaxM3MsaKvUpdate).
//!
//! Every tensor is int32 on one CUDA device. block_table is [rows, ncols] with unit column stride; every other
//! operand is contiguous with at least `batch` (per request) or `total_q` (per query token) elements, and cu_kv_lens
//! at least `batch + 1`. Writes msa_seq_lens, out_cache_loc and, when given, cu_kv_lens and n_valid_blocks.
void minimaxM3MsaKvUpdate(torch::Tensor const& kvLens, torch::Tensor const& kvLensStaged, torch::Tensor const& seqLens,
    torch::Tensor const& qBatchRow, torch::Tensor const& qIntra, torch::Tensor const& blockTable,
    int64_t tokensPerBlock, int64_t batch, int64_t totalQ, torch::Tensor& msaSeqLens,
    std::optional<torch::Tensor> cuKvLens, torch::Tensor& outCacheLoc, std::optional<torch::Tensor> nValidBlocks)
{
    checkInt32Cuda(kvLens, "kv_lens");
    checkInt32Cuda(kvLensStaged, "kv_lens_staged");
    checkInt32Cuda(seqLens, "seq_lens");
    checkInt32Cuda(qBatchRow, "q_batch_row");
    checkInt32Cuda(qIntra, "q_intra");
    checkInt32Cuda(blockTable, "block_table");
    checkInt32Cuda(msaSeqLens, "msa_seq_lens");
    checkInt32Cuda(outCacheLoc, "out_cache_loc");
    TORCH_CHECK(blockTable.dim() == 2 && blockTable.stride(1) == 1,
        "minimax_m3_msa_kv_update: block_table must be 2-D with unit column stride");
    TORCH_CHECK(tokensPerBlock > 0, "minimax_m3_msa_kv_update: tokens_per_block must be positive");
    TORCH_CHECK(batch >= 0 && totalQ >= 0, "minimax_m3_msa_kv_update: batch and total_q must be non-negative");
    TORCH_CHECK(kvLens.numel() >= batch && kvLensStaged.numel() >= batch && seqLens.numel() >= batch
            && msaSeqLens.numel() >= batch,
        "minimax_m3_msa_kv_update: a per-request operand has fewer than batch elements");
    TORCH_CHECK(qBatchRow.numel() >= totalQ && qIntra.numel() >= totalQ && outCacheLoc.numel() >= totalQ,
        "minimax_m3_msa_kv_update: a per-token operand has fewer than total_q elements");
    int32_t* cuKvPtr = nullptr;
    if (cuKvLens.has_value())
    {
        checkInt32Cuda(*cuKvLens, "cu_kv_lens");
        TORCH_CHECK(cuKvLens->numel() >= batch + 1, "minimax_m3_msa_kv_update: cu_kv_lens needs batch + 1 elements");
        cuKvPtr = cuKvLens->data_ptr<int32_t>();
    }
    int32_t* nValidPtr = nullptr;
    if (nValidBlocks.has_value())
    {
        checkInt32Cuda(*nValidBlocks, "n_valid_blocks");
        TORCH_CHECK(nValidBlocks->numel() >= totalQ, "minimax_m3_msa_kv_update: n_valid_blocks needs total_q elements");
        nValidPtr = nValidBlocks->data_ptr<int32_t>();
    }

    c10::cuda::CUDAGuard const deviceGuard{kvLens.device()};
    auto const stream = at::cuda::getCurrentCUDAStream(kvLens.get_device());
    cudaError_t const err = tensorrt_llm::kernels::invokeMinimaxM3MsaKvUpdate(kvLens.data_ptr<int32_t>(),
        kvLensStaged.data_ptr<int32_t>(), seqLens.data_ptr<int32_t>(), qBatchRow.data_ptr<int32_t>(),
        qIntra.data_ptr<int32_t>(), blockTable.data_ptr<int32_t>(), blockTable.stride(0),
        static_cast<int32_t>(blockTable.size(1)), static_cast<int32_t>(tokensPerBlock), static_cast<int32_t>(batch),
        static_cast<int32_t>(totalQ), msaSeqLens.data_ptr<int32_t>(), cuKvPtr, outCacheLoc.data_ptr<int32_t>(),
        nValidPtr, stream);
    TORCH_CHECK(err == cudaSuccess, "minimax_m3_msa_kv_update launch failed: ", cudaGetErrorString(err));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m3_msa_kv_update(Tensor kv_lens, Tensor kv_lens_staged, Tensor seq_lens, Tensor q_batch_row, "
        "Tensor q_intra, Tensor block_table, int tokens_per_block, int batch, int total_q, Tensor(a!) msa_seq_lens, "
        "Tensor(b!)? cu_kv_lens, Tensor(c!) out_cache_loc, Tensor(d!)? n_valid_blocks) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_msa_kv_update", &tensorrt_llm::torch_ext::minimaxM3MsaKvUpdate);
}
