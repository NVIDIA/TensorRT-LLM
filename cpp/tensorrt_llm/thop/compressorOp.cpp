/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/compressorKernels/compressorKernels.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace tk = tensorrt_llm::kernels::compressor;

namespace
{

// Decode kernel: write tokens to paged cache + conditional compression
void compressorPagedKvCompressOp(torch::Tensor kv_score, // [m, 2*state_dim] bf16/fp32
    torch::Tensor ape,                                   // [compress_ratio, state_dim] fp32
    torch::Tensor paged_kv,                              // [num_blocks, page_size, state_dim] bf16/fp32
    torch::Tensor paged_score,                           // [num_blocks, page_size, state_dim] bf16/fp32
    torch::Tensor block_table_kv,                        // [bsz, max_blocks] int32
    torch::Tensor block_table_score,                     // [bsz, max_blocks] int32
    torch::Tensor output,                                // [total_outputs, head_dim] bf16
    torch::Tensor kv_lens,                               // [bsz] int32
    torch::Tensor cu_seq_lens,                           // [bsz+1] int32
    torch::Tensor cu_kv_comp,                            // [bsz+1] int32
    int64_t batch_size, int64_t page_size, int64_t head_dim, int64_t compress_ratio, int64_t next_n)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    int kv_score_eb = static_cast<int>(kv_score.element_size());
    int state_eb = static_cast<int>(paged_kv.element_size());
    int out_eb = static_cast<int>(output.element_size());
    TORCH_CHECK(
        paged_score.element_size() == paged_kv.element_size(), "paged_kv and paged_score must use the same dtype");

    tk::pagedKvCompressLaunch(kv_score.data_ptr(), ape.data_ptr<float>(), paged_kv.data_ptr(), paged_score.data_ptr(),
        block_table_kv.data_ptr<int32_t>(), block_table_score.data_ptr<int32_t>(), output.data_ptr(),
        kv_lens.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(), cu_kv_comp.data_ptr<int32_t>(),
        static_cast<int>(batch_size), static_cast<int>(page_size), static_cast<int>(block_table_kv.size(1)),
        static_cast<int>(head_dim), static_cast<int>(compress_ratio), static_cast<int>(next_n), kv_score_eb, state_eb,
        out_eb, stream);
}

// Prefill kernel: bulk compression with state update
void compressorPrefillReductionOp(torch::Tensor kv_score, torch::Tensor ape, torch::Tensor paged_kv,
    torch::Tensor paged_score, torch::Tensor block_table_kv, torch::Tensor block_table_score, torch::Tensor output,
    torch::Tensor kv_lens, torch::Tensor start_pos, torch::Tensor cu_seq_lens, torch::Tensor cu_kv_comp,
    int64_t batch_size, int64_t page_size, int64_t head_dim, int64_t compress_ratio, int64_t max_outputs)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    int kv_score_eb = static_cast<int>(kv_score.element_size());
    int state_eb = static_cast<int>(paged_kv.element_size());
    int out_eb = static_cast<int>(output.element_size());
    TORCH_CHECK(
        paged_score.element_size() == paged_kv.element_size(), "paged_kv and paged_score must use the same dtype");

    tk::prefillReductionLaunch(kv_score.data_ptr(), ape.data_ptr<float>(), paged_kv.data_ptr(), paged_score.data_ptr(),
        block_table_kv.data_ptr<int32_t>(), block_table_score.data_ptr<int32_t>(), output.data_ptr(),
        kv_lens.data_ptr<int32_t>(), start_pos.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(),
        cu_kv_comp.data_ptr<int32_t>(), static_cast<int>(batch_size), static_cast<int>(page_size),
        static_cast<int>(block_table_kv.size(1)), static_cast<int>(head_dim), static_cast<int>(compress_ratio),
        static_cast<int>(max_outputs), kv_score_eb, state_eb, out_eb, stream);
}

// Fused postprocess + scatter: RMSNorm + RoPE + Hadamard + paged scatter in one kernel
void compressorPostProcessScatterOp(torch::Tensor kv_comp, // [total_tokens, head_dim] input
    std::optional<torch::Tensor> kv_out,                   // [total_tokens, head_dim] output (optional)
    torch::Tensor rms_weight,                              // [head_dim]
    double rms_eps,
    torch::Tensor cos_sin_table,                           // [max_pos, 2, rope_dim/2]
    torch::Tensor position_ids,                            // [total_tokens]
    int64_t nope_dim, int64_t rope_dim,
    torch::Tensor kv_cache,                                // paged cache buffer
    torch::Tensor num_outputs,                             // [bsz] int32
    torch::Tensor cu_kv_comp,                              // [bsz+1] int32
    torch::Tensor start_pos,                               // [bsz] int32
    torch::Tensor block_offsets,                           // [bsz, max_blocks] int32
    torch::Tensor compressed_mask,                         // [total_tokens] bool — per-token mask
    int64_t tokens_per_block, int64_t cache_scale_type, bool rotate_activation,
    std::optional<torch::Tensor> quant_output, std::optional<torch::Tensor> scale_output)
{
    auto stream = at::cuda::getCurrentCUDAStream();

    TORCH_CHECK(
        cos_sin_table.scalar_type() == at::kFloat, "cos_sin_table must be float32, got ", cos_sin_table.scalar_type());
    TORCH_CHECK(cos_sin_table.is_contiguous(), "cos_sin_table must be contiguous");
    TORCH_CHECK(position_ids.scalar_type() == at::kInt, "position_ids must be int32, got ", position_ids.scalar_type());
    TORCH_CHECK(position_ids.is_contiguous(), "position_ids must be contiguous");
    TORCH_CHECK(compressed_mask.scalar_type() == at::kBool, "compressed_mask must be bool, got ",
        compressed_mask.scalar_type());

    tk::postProcessScatterLaunch(kv_comp.data_ptr(), kv_out.has_value() ? kv_out->data_ptr() : nullptr,
        rms_weight.data_ptr(), static_cast<float>(rms_eps), cos_sin_table.data_ptr<float>(),
        position_ids.data_ptr<int32_t>(), static_cast<int>(nope_dim), static_cast<int>(rope_dim), kv_cache.data_ptr(),
        num_outputs.data_ptr<int32_t>(), cu_kv_comp.data_ptr<int32_t>(), start_pos.data_ptr<int32_t>(),
        block_offsets.data_ptr<int32_t>(), reinterpret_cast<bool const*>(compressed_mask.data_ptr()),
        static_cast<int>(num_outputs.size(0)),    // batch_size
        static_cast<int>(tokens_per_block),
        static_cast<int>(kv_comp.size(1)),        // head_dim
        static_cast<int>(block_offsets.size(1)),  // max_blocks
        static_cast<int>(kv_comp.element_size()), // elem_bytes
        static_cast<int>(kv_comp.size(0)),        // total_tokens
        static_cast<int>(cache_scale_type), rotate_activation,
        quant_output.has_value() ? quant_output->data_ptr() : nullptr,
        scale_output.has_value() ? scale_output->data_ptr() : nullptr, stream);
}

} // anonymous namespace

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "compressor_paged_kv_compress("
        "Tensor kv_score, Tensor ape, "
        "Tensor(a!) paged_kv, Tensor(b!) paged_score, "
        "Tensor block_table_kv, Tensor block_table_score, "
        "Tensor(c!) output, "
        "Tensor kv_lens, "
        "Tensor cu_seq_lens, Tensor cu_kv_comp, "
        "int batch_size, int page_size, "
        "int head_dim, int compress_ratio, "
        "int next_n) -> ()");

    m.def(
        "compressor_prefill_reduction("
        "Tensor kv_score, Tensor ape, "
        "Tensor(a!) paged_kv, Tensor(b!) paged_score, "
        "Tensor block_table_kv, Tensor block_table_score, "
        "Tensor(c!) output, "
        "Tensor kv_lens, Tensor start_pos, "
        "Tensor cu_seq_lens, Tensor cu_kv_comp, "
        "int batch_size, int page_size, "
        "int head_dim, int compress_ratio, "
        "int max_outputs) -> ()");

    m.def(
        "compressor_postprocess_scatter("
        "Tensor kv_comp, Tensor(a!)? kv_out, "
        "Tensor rms_weight, float rms_eps, "
        "Tensor cos_sin_table, Tensor position_ids, "
        "int nope_dim, int rope_dim, "
        "Tensor(b!) kv_cache, "
        "Tensor num_outputs, Tensor cu_kv_comp, "
        "Tensor start_pos, Tensor block_offsets, "
        "Tensor compressed_mask, "
        "int tokens_per_block, int cache_scale_type, "
        "bool rotate_activation, "
        "Tensor(c!)? quant_output, Tensor(d!)? scale_output) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("compressor_paged_kv_compress", &compressorPagedKvCompressOp);
    m.impl("compressor_prefill_reduction", &compressorPrefillReductionOp);
    m.impl("compressor_postprocess_scatter", &compressorPostProcessScatterOp);
}
