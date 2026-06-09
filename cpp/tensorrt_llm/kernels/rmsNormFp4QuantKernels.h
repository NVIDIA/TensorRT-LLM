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

// Parameters for the fused (optional residual-add +) RMSNorm + NVFP4
// input-quantize kernel. Self-contained (does not borrow AllReduceParams) since
// this kernel performs no allreduce — it backs the standalone thop ops
// fused_add_rmsnorm_fp4_quantize / fused_rmsnorm_fp4_quantize on the
// attention-DP path.
struct RmsNormFp4QuantParams
{
    // Input values [m, hidden_size] (read with input_row_stride). When
    // residual_buffer != nullptr this is also overwritten with residual_out
    // (= input + residual).
    void* intermediate_buffer{nullptr};
    // residual_in [m, hidden_size]; nullptr disables the residual add.
    void const* residual_buffer{nullptr};
    // optional bias add [hidden_size]; nullptr disables it.
    void const* bias_buffer{nullptr};
    // RMSNorm gamma [hidden_size]; nullptr selects the non-affine path.
    void const* weight_buffer{nullptr};
    int hidden_size{0};
    float eps{0.f};
    // Total element count (= m * hidden_size); used to derive the row count.
    int64_t elts_total{0};
};

// Fused (optional residual-add +) RMSNorm + NVFP4 input-quantize. Folds RMSNorm
// and the next op's NVFP4 input-quant so the (flashinfer RMSNorm + standalone
// fp4_quantize) pair becomes one launch on the attention-DP path.
//
// Inputs (via params):
//   - intermediate_buffer: the input values (also residual_out on exit
//     when residual_buffer != nullptr)
//   - residual_buffer:     residual_in (set nullptr to disable)
//   - bias_buffer:         optional bias add (set nullptr to disable)
//   - weight_buffer:       RMSNorm gamma (set nullptr for no affine)
//   - hidden_size, eps:    standard
//
// Outputs:
//   - quant_out:        packed FP4 (E2M1) values, kSfVecSize=16 per byte
//   - scale_out:        E4M3 scaling factors (one per 16-elem block)
//   - norm_out_ptr:     optional BF16 normed value (matches the
//                       OUT_QUANT_NVFP4 fusion shape); pass nullptr to skip
//   - scale_factor_ptr: device pointer to the global per-tensor scale
//                       (= 448*6 / amax for static-quant Linear)
//   - sf_layout:        scaling-factor layout (typically Swizzled)
//   - input_row_stride: element stride between input rows in intermediate_buffer.
//                       Defaults to 0 meaning "== hidden_size" (packed rows), so
//                       every existing caller is byte-identical. Set >0 to read a
//                       strided slice (e.g. a column-slice of a wider projection)
//                       without a preceding contiguous copy. Outputs stay packed.
void residualRmsNormFp4Quant(RmsNormFp4QuantParams& params, void* quant_out, void* scale_out, void* norm_out_ptr,
    float const* scale_factor_ptr, ::tensorrt_llm::QuantizationSFLayout sf_layout, nvinfer1::DataType dataType,
    cudaStream_t stream, int input_row_stride = 0);

} // namespace kernels

TRTLLM_NAMESPACE_END
