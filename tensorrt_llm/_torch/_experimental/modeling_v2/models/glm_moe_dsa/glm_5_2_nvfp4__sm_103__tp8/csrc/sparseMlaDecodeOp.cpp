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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/sparseMlaDecode.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include <torch/types.h>

#include <optional>

namespace tk = TRTLLM_MV2_NS::sparse_mla_decode;

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

bool isFp8ByteTensor(torch::Tensor const& t)
{
    return t.scalar_type() == torch::kUInt8 || t.scalar_type() == torch::kFloat8_e4m3fn;
}

} // namespace

// Sparse MLA generation attention; see sparseMlaDecode.h.
void sparseMlaDecode(torch::Tensor const& q, int64_t kvPoolPtr, torch::Tensor const& indices,
    torch::Tensor const& bmm1Scale, torch::Tensor const& bmm2Scale, torch::Tensor const& seqLensKv, int64_t seqLenQ,
    torch::Tensor& out)
{
    TORCH_CHECK(q.is_cuda() && indices.is_cuda() && bmm1Scale.is_cuda() && bmm2Scale.is_cuda() && out.is_cuda(),
        "glm_moe_dsa_sparse_mla_decode expects CUDA tensors");
    TORCH_CHECK(isFp8ByteTensor(q), "q must hold e4m3 bytes (uint8 or float8_e4m3fn)");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    int64_t const numRows = q.size(0);
    TORCH_CHECK(q.numel() == numRows * tk::kNumHeads * tk::kHeadDimQk, "q must be [rows, ", tk::kNumHeads, ", ",
        tk::kHeadDimQk, "]");
    TORCH_CHECK(indices.scalar_type() == torch::kInt32 && indices.dim() == 2, "indices must be a 2-D int32 tensor");
    TORCH_CHECK(indices.size(0) == numRows, "indices must have one row per query row");
    TORCH_CHECK(indices.size(1) == tk::kTopK, "indices must hold ", tk::kTopK, " entries per row");
    TORCH_CHECK(indices.stride(1) == 1, "indices rows must be contiguous");
    TORCH_CHECK(bmm1Scale.scalar_type() == torch::kFloat32 && bmm1Scale.numel() >= 2, "bmm1_scale must be float32[2]");
    TORCH_CHECK(bmm2Scale.scalar_type() == torch::kFloat32 && bmm2Scale.numel() >= 1, "bmm2_scale must be float32[1]");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.is_contiguous(), "out must be contiguous bfloat16");
    TORCH_CHECK(out.numel() == numRows * tk::kNumHeads * tk::kHeadDimV, "out must be [rows, ",
        tk::kNumHeads * tk::kHeadDimV, "]");
    TORCH_CHECK(kvPoolPtr != 0, "kv_pool_ptr must be a valid device address");

    tk::SparseMlaDecodeParams params{};
    params.q = q.data_ptr();
    params.kvPool = reinterpret_cast<void const*>(kvPoolPtr);
    params.indices = indices.data_ptr<int32_t>();
    params.indicesRowStride = indices.stride(0);
    params.bmm1Scale = bmm1Scale.data_ptr<float>();
    params.bmm2Scale = bmm2Scale.data_ptr<float>();
    TORCH_CHECK(seqLensKv.is_cuda() && seqLensKv.scalar_type() == torch::kInt32 && seqLensKv.is_contiguous(),
        "seq_lens_kv must be a contiguous int32 CUDA tensor");
    TORCH_CHECK(seqLenQ > 0 && numRows % seqLenQ == 0, "rows must be a multiple of seq_len_q");
    TORCH_CHECK(seqLensKv.numel() >= numRows / seqLenQ, "seq_lens_kv must cover every sequence");
    params.seqLensKv = seqLensKv.data_ptr<int32_t>();
    params.seqLenQ = static_cast<int32_t>(seqLenQ);
    params.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    params.numRows = static_cast<int32_t>(numRows);

    auto const stream = at::cuda::getCurrentCUDAStream(q.get_device());
    tk::invokeSparseMlaDecode(params, stream);
}

bool sparseMlaDecodeSupported(int64_t device)
{
    return tk::isSparseMlaDecodeDeviceSupported(static_cast<int>(device));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "sparse_mla_decode(Tensor q, int kv_pool_ptr, Tensor indices, Tensor bmm1_scale, "
        "Tensor bmm2_scale, Tensor seq_lens_kv, int seq_len_q, Tensor(a!) out) -> ()");
    m.def("sparse_mla_decode_supported(int device) -> bool");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("sparse_mla_decode", &TRTLLM_MV2_NS::torch_ext::sparseMlaDecode);
}

TRTLLM_MV2_LIBRARY_IMPL(CompositeExplicitAutograd, m)
{
    m.impl("sparse_mla_decode_supported", &TRTLLM_MV2_NS::torch_ext::sparseMlaDecodeSupported);
}
