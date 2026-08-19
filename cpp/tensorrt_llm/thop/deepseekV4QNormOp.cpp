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

#include "tensorrt_llm/kernels/deepseekV4QNormKernel.h"

#include <ATen/cuda/CUDAContext.h>
#include <limits>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

torch::Tensor deepseekV4QNorm(torch::Tensor q, int64_t numHeads, int64_t headDim, double eps)
{
    TORCH_CHECK(q.is_cuda(), "deepseek_v4_q_norm expects a CUDA tensor");
    TORCH_CHECK(q.is_contiguous(), "deepseek_v4_q_norm expects a contiguous tensor");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 || q.scalar_type() == torch::kBFloat16,
        "deepseek_v4_q_norm expects fp16/bf16 input, got ", q.scalar_type());
    TORCH_CHECK(headDim == 512, "deepseek_v4_q_norm only supports head_dim=512, got ", headDim);

    TORCH_CHECK(q.dim() == 2, "deepseek_v4_q_norm expects 2D q [num_tokens, num_heads * head_dim], got ", q.dim(), "D");
    TORCH_CHECK(q.size(1) == numHeads * headDim, "q.shape[1] must equal num_heads * head_dim");

    auto output = torch::empty_like(q);
    int64_t const totalRows64 = q.size(0) * numHeads;
    TORCH_CHECK(totalRows64 <= std::numeric_limits<int>::max(), "deepseek_v4_q_norm total rows exceed int range");
    int const totalRows = static_cast<int>(totalRows64);
    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());

    if (q.scalar_type() == torch::kFloat16)
    {
        tensorrt_llm::kernels::invokeDeepseekV4QNorm(q.data_ptr(), output.data_ptr(), totalRows,
            static_cast<int>(headDim), false, static_cast<float>(eps), stream);
    }
    else
    {
        tensorrt_llm::kernels::invokeDeepseekV4QNorm(q.data_ptr(), output.data_ptr(), totalRows,
            static_cast<int>(headDim), true, static_cast<float>(eps), stream);
    }
    return output;
}

// Writes the FP8 nope segment of Q into `quantQOut` and the bf16/fp16 rope
// segment into `qPeOut` in one fused kernel. The caller pre-allocates both
// output tensors; the FP8 row stride is inferred from `quantQOut.size(1)`
// (either `numHeads * headDim` for an interleaved Q buffer that
// `applyMLARopeAndAssignQKVKernelOptContext` will fill the rope slot of, or
// `numHeads * nopeDim` for a packed nope-only layout).
void deepseekV4QNormFusedFp8(torch::Tensor q, torch::Tensor quantQOut, torch::Tensor qPeOut, int64_t numHeads,
    int64_t headDim, int64_t nopeDim, double eps, torch::optional<torch::Tensor> quantScaleQkv,
    torch::optional<torch::Tensor> rotaryCosSin, torch::optional<torch::Tensor> cacheSeqLens, int64_t seqLen,
    torch::optional<torch::Tensor> cuQSeqLens)
{
    TORCH_CHECK(q.is_cuda(), "deepseek_v4_q_norm_fused_fp8 expects a CUDA tensor");
    TORCH_CHECK(q.is_contiguous(), "deepseek_v4_q_norm_fused_fp8 expects a contiguous tensor");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16 || q.scalar_type() == torch::kBFloat16,
        "deepseek_v4_q_norm_fused_fp8 expects fp16/bf16 input, got ", q.scalar_type());
    TORCH_CHECK(headDim == 512, "deepseek_v4_q_norm_fused_fp8 only supports head_dim=512, got ", headDim);
    TORCH_CHECK(nopeDim == 448, "deepseek_v4_q_norm_fused_fp8 only supports nope_dim=448, got ", nopeDim);

    TORCH_CHECK(q.dim() == 2, "deepseek_v4_q_norm_fused_fp8 expects 2D q [num_tokens, num_heads * head_dim], got ",
        q.dim(), "D");
    TORCH_CHECK(q.size(1) == numHeads * headDim, "q.shape[1] must equal num_heads * head_dim");

    int64_t const numTokens = q.size(0);
    int64_t const ropeDim = headDim - nopeDim;

    TORCH_CHECK(quantQOut.is_cuda() && quantQOut.scalar_type() == torch::kFloat8_e4m3fn,
        "quant_q_out must be a CUDA Float8_e4m3fn tensor");
    TORCH_CHECK(quantQOut.dim() == 2 && quantQOut.size(0) == numTokens,
        "quant_q_out must be [num_tokens, num_heads * stride] with matching num_tokens");
    int64_t const quantQStridePerHead = quantQOut.size(1) / numHeads;
    TORCH_CHECK(quantQOut.size(1) == numHeads * quantQStridePerHead, "quant_q_out.shape[1] (", quantQOut.size(1),
        ") must be a multiple of num_heads (", numHeads, ")");
    TORCH_CHECK(quantQStridePerHead == nopeDim || quantQStridePerHead == headDim,
        "quant_q_out per-head stride must be nope_dim (", nopeDim, ") for packed or head_dim (", headDim,
        ") for interleaved, got ", quantQStridePerHead);

    TORCH_CHECK(qPeOut.is_cuda() && qPeOut.scalar_type() == q.scalar_type(),
        "q_pe_out must be a CUDA tensor with the same dtype as q");
    TORCH_CHECK(qPeOut.dim() == 2 && qPeOut.size(0) == numTokens && qPeOut.size(1) == numHeads * ropeDim,
        "q_pe_out must be [num_tokens, num_heads * rope_dim]");

    int64_t const totalRows64 = numTokens * numHeads;
    TORCH_CHECK(
        totalRows64 <= std::numeric_limits<int>::max(), "deepseek_v4_q_norm_fused_fp8 total rows exceed int range");
    int const totalRows = static_cast<int>(totalRows64);

    void const* quantScalePtr = nullptr;
    if (quantScaleQkv.has_value() && quantScaleQkv->defined())
    {
        auto const& s = quantScaleQkv.value();
        TORCH_CHECK(s.is_cuda(), "quant_scale_qkv must be on CUDA");
        TORCH_CHECK(s.scalar_type() == torch::kFloat32, "quant_scale_qkv must be float32");
        TORCH_CHECK(s.numel() >= 1, "quant_scale_qkv must have at least 1 element");
        quantScalePtr = s.data_ptr();
    }

    int const quantQNopeRowStrideBytes = static_cast<int>(quantQStridePerHead);

    // Fusing the Q RoPE writes into this row's rope slots, so it needs the
    // interleaved layout and a position per row.
    void const* cosSinPtr = nullptr;
    int const* cacheSeqLensPtr = nullptr;
    int const* cuQSeqLensPtr = nullptr;
    int numSeqs = 0;
    if (rotaryCosSin.has_value() && rotaryCosSin->defined() && cacheSeqLens.has_value() && cacheSeqLens->defined())
    {
        TORCH_CHECK(rotaryCosSin->is_cuda(), "rotary_cos_sin must be on CUDA");
        TORCH_CHECK(rotaryCosSin->scalar_type() == torch::kFloat32,
            "rotary_cos_sin must be float32 (the kernel reads it as float2), got ", rotaryCosSin->scalar_type());
        TORCH_CHECK(cacheSeqLens->is_cuda(), "cache_seq_lens must be on CUDA");
        TORCH_CHECK(quantQStridePerHead == headDim,
            "fused Q RoPE needs the interleaved quant_q_out layout (per-head stride == head_dim)");
        TORCH_CHECK(cacheSeqLens->scalar_type() == torch::kInt32, "cache_seq_lens must be int32");
        cosSinPtr = rotaryCosSin->data_ptr();
        cacheSeqLensPtr = cacheSeqLens->data_ptr<int>();
        if (cuQSeqLens.has_value() && cuQSeqLens->defined())
        {
            // Ragged context form: positions come from cu_q_seqlens (in tokens).
            TORCH_CHECK(cuQSeqLens->is_cuda(), "cu_q_seqlens must be on CUDA");
            TORCH_CHECK(cuQSeqLens->scalar_type() == torch::kInt32, "cu_q_seqlens must be int32");
            TORCH_CHECK(cuQSeqLens->numel() >= 2, "cu_q_seqlens must hold num_seqs + 1 entries");
            cuQSeqLensPtr = cuQSeqLens->data_ptr<int>();
            numSeqs = static_cast<int>(cuQSeqLens->numel()) - 1;
            TORCH_CHECK(cacheSeqLens->numel() >= numSeqs, "cache_seq_lens holds ", cacheSeqLens->numel(),
                " entries but cu_q_seqlens implies ", numSeqs, " sequences");
        }
        else
        {
            TORCH_CHECK(seqLen > 0, "fused Q RoPE needs a positive seq_len without cu_q_seqlens");
        }
    }

    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    bool const isBfloat16 = (q.scalar_type() == torch::kBFloat16);
    tensorrt_llm::kernels::invokeDeepseekV4QNormFusedFp8(q.data_ptr(), quantQOut.data_ptr(), qPeOut.data_ptr(),
        quantScalePtr, totalRows, static_cast<int>(headDim), static_cast<int>(nopeDim), quantQNopeRowStrideBytes,
        isBfloat16, static_cast<float>(eps), cosSinPtr, cacheSeqLensPtr, static_cast<int>(numHeads),
        static_cast<int>(seqLen), cuQSeqLensPtr, numSeqs, stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("deepseek_v4_q_norm(Tensor q, int num_heads, int head_dim, float eps) -> Tensor");
    m.def(
        "deepseek_v4_q_norm_fused_fp8(Tensor q, Tensor(a!) quant_q_out, Tensor(b!) q_pe_out, "
        "int num_heads, int head_dim, int nope_dim, float eps, Tensor? quant_scale_qkv, "
        "Tensor? rotary_cos_sin=None, Tensor? cache_seq_lens=None, int seq_len=0, "
        "Tensor? cu_q_seqlens=None) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("deepseek_v4_q_norm", &tensorrt_llm::torch_ext::deepseekV4QNorm);
    m.impl("deepseek_v4_q_norm_fused_fp8", &tensorrt_llm::torch_ext::deepseekV4QNormFusedFp8);
}
