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

#include <cstdint>
#include <cuda_runtime.h>
#include <NvInferRuntime.h>

TRTLLM_NAMESPACE_BEGIN
namespace kernels::cutlass_kernels
{

// Phase 6b.E follow-up, workaround (a): pre-aggregate the per-permuted-row
// FC2 LoRA delta into the per-original-token `final_output` buffer using
// the same unpermute + topk-weighted scale that the FINALIZE epilogue
// performs for the FC2 main GEMM result.
//
// Mathematically:
//
//   final_output[t, h] = sum over k in [0, experts_per_token):
//                            scales[t, k]
//                            * fc2_lora[unpermuted_row_to_permuted_row[t + k * num_rows], h]
//
// (with (t, k) pairs whose `token_selected_experts[t * k + k] - start_expert`
// falls outside `[0, num_experts_per_node)` skipped, mirroring
// `finalizeMoeRoutingKernel`'s alltoall/EP filter.)
//
// `final_output` is OVERWRITTEN, not accumulated. The intended call
// sequence is:
//
//   1. launchMoeLoraPreSum(...)        -> final_output := sum_k s_k * lora_delta_k
//   2. FC2 main GEMM with FINALIZE     -> final_output += sum_k s_k * gemm_result_k
//                                         (atomically, when use_reduction=true)
//
// Net result: `final_output[t, h] = sum_k s_k * (gemm_result_k + lora_delta_k)`,
// matching the original sequence
//
//   1. fc2_result    := GEMM2(input)
//   2. fc2_result    += lora_delta            (loraBiasApplyFunc)
//   3. final_output  := sum_k s_k * fc2_result
//
// but with the FC2 GEMM running on the stable `EpilogueFusion::FINALIZE`
// kernel template instead of the buggy `EpilogueFusion::NONE` one.
//
// `lora_delta_row_stride_elems` is the stride (in DeltaType elements)
// between consecutive per-permuted rows of `lora_delta`. For the in-tree
// `lora_fc2_result_` buffer this is `hidden_size`.
// `final_output_row_stride_elems` is the stride (in OutputType elements)
// between consecutive output rows. The FINALIZE epilogue uses
// `unpadded_hidden_size` for this stride, so we mirror that here.
//
// `dtype` selects the DeltaType / OutputType pair. Only `kBF16` and
// `kHALF` are wired up: those are the only activation dtypes the MoE
// LoRA path supports today.
void launchMoeLoraPreSum(void const* lora_delta, void* final_output, float const* unpermuted_final_scales,
    int const* unpermuted_row_to_permuted_row, int const* token_selected_experts, int64_t num_rows,
    int64_t lora_delta_row_stride_elems, int64_t final_output_row_stride_elems, int experts_per_token,
    int num_experts_per_node, int start_expert, nvinfer1::DataType dtype, cudaStream_t stream);

} // namespace kernels::cutlass_kernels
TRTLLM_NAMESPACE_END
