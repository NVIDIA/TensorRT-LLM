/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/quantization.h"
#include <NvInferRuntime.h>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// All parameters for the fused (optional residual-add +) RMSNorm + NVFP4
// input-quantize kernel: inputs, outputs, and layout config alike, so the
// launcher takes a single struct. Self-contained (does not borrow
// AllReduceParams) since this kernel performs no allreduce — it backs the
// standalone thop ops fused_add_rmsnorm_fp4_quantize /
// fused_rmsnorm_fp4_quantize on the attention-DP path.
struct RmsNormFp4QuantParams
{
    // --- inputs ---
    // Input values [m, hidden_size] (read with input_row_stride). Read-only: the
    // residual sum is written to residual_out_buffer, never back into this.
    void const* intermediate_buffer{nullptr};
    // residual_in [m, hidden_size]; nullptr disables the residual add.
    void const* residual_buffer{nullptr};
    // optional bias add [hidden_size]; nullptr disables it.
    void const* bias_buffer{nullptr};
    // RMSNorm gamma [hidden_size]; nullptr selects the non-affine path.
    void const* weight_buffer{nullptr};
    // Device pointer to the global per-tensor scale (= 448*6 / amax for
    // static-quant Linear); nullptr means 1.0.
    float const* scale_factor_ptr{nullptr};

    // --- outputs ---
    // Packed FP4 (E2M1) values, 2 per byte.
    void* quant_out{nullptr};
    // E4M3 scaling factors (one per SF_VEC_SIZE=16 block), laid out per sf_layout.
    void* scale_out{nullptr};
    // Optional BF16/FP16 post-RMSNorm value (packed rows); nullptr to skip.
    void* norm_out{nullptr};
    // residual_out [m, hidden_size] (packed): receives input + residual when
    // residual_buffer != nullptr. A distinct buffer from intermediate_buffer so
    // the input is never mutated (keeps the thop op functionalizable under
    // torch.compile) and no pre-kernel copy is needed. nullptr when no residual.
    void* residual_out_buffer{nullptr};

    // --- config ---
    int hidden_size{0};
    float eps{0.f};
    // Total element count (= m * hidden_size); used to derive the row count.
    int64_t elts_total{0};
    // Scaling-factor layout (typically SWIZZLED).
    ::tensorrt_llm::QuantizationSFLayout sf_layout{::tensorrt_llm::QuantizationSFLayout::SWIZZLED};
    // Element stride between input rows in intermediate_buffer. 0 means
    // "== hidden_size" (packed rows). Set >0 to read a strided slice (e.g. a
    // column-slice of a wider projection) without a preceding contiguous copy.
    // Outputs are always written packed.
    int input_row_stride{0};
};

// Fused (optional residual-add +) RMSNorm + NVFP4 input-quantize. Folds RMSNorm
// and the next op's NVFP4 input-quant so the (flashinfer RMSNorm + standalone
// fp4_quantize) pair becomes one launch on the attention-DP path. All inputs,
// outputs, and layout configuration are carried in params (see the struct
// field docs above); dataType selects the fp16/bf16 instantiation.
void residualRmsNormFp4Quant(RmsNormFp4QuantParams const& params, nvinfer1::DataType dataType, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
