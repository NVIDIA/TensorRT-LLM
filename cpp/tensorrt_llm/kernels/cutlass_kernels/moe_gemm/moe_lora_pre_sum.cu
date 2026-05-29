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

#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_lora_pre_sum.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN
namespace kernels::cutlass_kernels
{

namespace
{

// One CTA per original token. Threads in the CTA stripe across the
// `final_output_row_stride_elems` (== unpadded_hidden_size) columns. The
// per-(token, k) lookup in `unpermuted_row_to_permuted_row` and
// `token_selected_experts` is cheap relative to the gather of `lora_delta`,
// so we don't bother caching it in shared memory.
constexpr int kThreadsPerBlock = 256;

template <typename DeltaType, typename OutputType>
__global__ void moeLoraPreSumKernel(DeltaType const* __restrict__ lora_delta,
    OutputType* __restrict__ final_output, float const* __restrict__ unpermuted_final_scales,
    int const* __restrict__ unpermuted_row_to_permuted_row, int const* __restrict__ token_selected_experts,
    int64_t num_rows, int64_t lora_delta_row_stride_elems, int64_t final_output_row_stride_elems, int experts_per_token,
    int num_experts_per_node, int start_expert)
{
    int64_t const t = blockIdx.x;
    if (t >= num_rows)
    {
        return;
    }

    OutputType* out_row = final_output + t * final_output_row_stride_elems;

    for (int64_t h = threadIdx.x; h < final_output_row_stride_elems; h += blockDim.x)
    {
        float acc = 0.0f;
        for (int k = 0; k < experts_per_token; ++k)
        {
            int64_t const k_offset = t * experts_per_token + k;
            int const expert_id = token_selected_experts[k_offset] - start_expert;
            // Mirrors finalizeMoeRoutingKernel: skip (t, k) pairs whose
            // expert is filtered out (alltoall, EP). MoE LoRA rejects
            // alltoall at the op level, but we still respect start_expert
            // so an EP slice doesn't aggregate foreign experts' deltas.
            if (expert_id < 0 || expert_id >= num_experts_per_node)
            {
                continue;
            }

            int64_t const expanded_orig = t + static_cast<int64_t>(k) * num_rows;
            int64_t const expanded_permuted = unpermuted_row_to_permuted_row[expanded_orig];

            float const row_scale = unpermuted_final_scales[k_offset];
            float const val
                = static_cast<float>(lora_delta[expanded_permuted * lora_delta_row_stride_elems + h]);
            acc += row_scale * val;
        }
        out_row[h] = static_cast<OutputType>(acc);
    }
}

template <typename DeltaType, typename OutputType>
void launchMoeLoraPreSumImpl(void const* lora_delta, void* final_output, float const* unpermuted_final_scales,
    int const* unpermuted_row_to_permuted_row, int const* token_selected_experts, int64_t num_rows,
    int64_t lora_delta_row_stride_elems, int64_t final_output_row_stride_elems, int experts_per_token,
    int num_experts_per_node, int start_expert, cudaStream_t stream)
{
    if (num_rows <= 0)
    {
        return;
    }
    int const blocks = static_cast<int>(num_rows);
    int const threads = kThreadsPerBlock;
    moeLoraPreSumKernel<DeltaType, OutputType><<<blocks, threads, 0, stream>>>(
        static_cast<DeltaType const*>(lora_delta), static_cast<OutputType*>(final_output), unpermuted_final_scales,
        unpermuted_row_to_permuted_row, token_selected_experts, num_rows, lora_delta_row_stride_elems,
        final_output_row_stride_elems, experts_per_token, num_experts_per_node, start_expert);
}

} // namespace

void launchMoeLoraPreSum(void const* lora_delta, void* final_output, float const* unpermuted_final_scales,
    int const* unpermuted_row_to_permuted_row, int const* token_selected_experts, int64_t num_rows,
    int64_t lora_delta_row_stride_elems, int64_t final_output_row_stride_elems, int experts_per_token,
    int num_experts_per_node, int start_expert, nvinfer1::DataType dtype, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(lora_delta != nullptr, "lora_delta must be non-null");
    TLLM_CHECK_WITH_INFO(final_output != nullptr, "final_output must be non-null");
    TLLM_CHECK_WITH_INFO(unpermuted_final_scales != nullptr, "unpermuted_final_scales must be non-null");
    TLLM_CHECK_WITH_INFO(unpermuted_row_to_permuted_row != nullptr, "unpermuted_row_to_permuted_row must be non-null");
    TLLM_CHECK_WITH_INFO(token_selected_experts != nullptr, "token_selected_experts must be non-null");
    TLLM_CHECK_WITH_INFO(experts_per_token > 0, "experts_per_token must be positive");
    TLLM_CHECK_WITH_INFO(num_experts_per_node > 0, "num_experts_per_node must be positive");
    TLLM_CHECK_WITH_INFO(lora_delta_row_stride_elems >= final_output_row_stride_elems,
        "lora_delta row stride must be at least the final_output row stride");

    switch (dtype)
    {
    case nvinfer1::DataType::kBF16:
        launchMoeLoraPreSumImpl<__nv_bfloat16, __nv_bfloat16>(lora_delta, final_output, unpermuted_final_scales,
            unpermuted_row_to_permuted_row, token_selected_experts, num_rows, lora_delta_row_stride_elems,
            final_output_row_stride_elems, experts_per_token, num_experts_per_node, start_expert, stream);
        break;
    case nvinfer1::DataType::kHALF:
        launchMoeLoraPreSumImpl<half, half>(lora_delta, final_output, unpermuted_final_scales,
            unpermuted_row_to_permuted_row, token_selected_experts, num_rows, lora_delta_row_stride_elems,
            final_output_row_stride_elems, experts_per_token, num_experts_per_node, start_expert, stream);
        break;
    default:
        TLLM_CHECK_WITH_INFO(
            false, "launchMoeLoraPreSum supports only bf16 and fp16 (matching the MoE LoRA path's dtype scope).");
    }
    sync_check_cuda_error(stream);
}

} // namespace kernels::cutlass_kernels
TRTLLM_NAMESPACE_END
