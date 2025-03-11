/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/mathUtils.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaPackedMask.h"
#include "tensorrt_llm/thop/thUtils.h"

namespace torch_ext
{
using torch::Tensor;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Build packed mask based on the attention mask types.
Tensor pack_fmha_mask_by_type(Tensor actual_q_seqlens, Tensor actual_kv_seqlens, int64_t const attention_mask_type,
    int64_t const batch_size, int64_t const max_q_seqlen, int64_t const max_kv_seqlen)
{
    // Create the cu_mask_rows tensor.
    Tensor cu_mask_rows
        = torch::empty({batch_size + 1}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    // Copy actual_q_seqlens from device to host.
    auto actual_q_seqlens_host = actual_q_seqlens.to(torch::kCPU);
    // Get the aligned cols.
    int aligned_cols = roundUp(int(max_kv_seqlen), int(FLASH_ATTEN_PACKED_MASK_N_ALIGNMENT));
    // Remove paddings in the row dimension.
    int total_aligned_rows = 0;
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++)
    {
        total_aligned_rows
            += roundUp(actual_q_seqlens_host.index({batch_idx}).item<int>(), int(FLASH_ATTEN_PACKED_MASK_M_ALIGNMENT));
    }
    Tensor packed_mask = torch::empty(
        {total_aligned_rows, aligned_cols / 32}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    // Set the parameters for creating packed mask.
    PackedMaskParams<float> maskParams{};
    maskParams.packedMask = get_ptr<uint32_t>(packed_mask);
    maskParams.cuMaskRows = get_ptr<int>(cu_mask_rows);
    maskParams.actualQSeqLens = get_ptr<int>(actual_q_seqlens);
    maskParams.actualKvSeqLens = get_ptr<int>(actual_kv_seqlens);
    maskParams.batchSize = int(batch_size);
    maskParams.maxQSeqLen = int(max_q_seqlen);
    maskParams.maxKvSeqLen = int(max_kv_seqlen);
    maskParams.attentionMaskType = static_cast<ContextAttentionMaskType>(int(attention_mask_type));

    // Launch the pack mask kernel.
    auto stream = at::cuda::getCurrentCUDAStream(actual_q_seqlens.get_device());
    invokeBuildPackedMask(maskParams, stream);

    return packed_mask;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Dispatch different mask data types.
template <typename T>
Tensor pack_fmha_mask_by_input_helper(
    Tensor mask_input, Tensor actual_q_seqlens, Tensor actual_kv_seqlens, double const valid_pos_value)
{
    CHECK_CONTIGUOUS(mask_input);
    CHECK_TH_CUDA(mask_input);
    CHECK_CONTIGUOUS(actual_q_seqlens);
    CHECK_TH_CUDA(actual_q_seqlens);
    TORCH_CHECK(mask_input.numel() != 0 && actual_q_seqlens.numel() != 0, "input should not be empty tensor");
    TORCH_CHECK(mask_input.dim() == 2 || mask_input.dim() == 3,
        "Invalid dim. The dim of mask_input should either be [num_tokens, max_kv_seqlen] or [batch_size, max_q_seqlen, "
        "max_kv_seqlen]");

    auto maskDataType = mask_input.scalar_type();
    TORCH_CHECK(maskDataType == torch::kFloat32 || maskDataType == torch::kFloat16 || maskDataType == torch::kBFloat16
            || maskDataType == torch::kBool || maskDataType == torch::kInt32,
        "Invalid datatype. input must be FP16, BF16, FP32, Bool or INT32");

    // Get the shape info.
    bool packed_mask_input = mask_input.dim() == 2;
    int batch_size = actual_q_seqlens.size(0);
    int max_kv_seqlen = packed_mask_input ? mask_input.size(1) : mask_input.size(2);

    // Create the cu_q_seqlens tensor.
    Tensor cu_q_seqlens
        = torch::empty({batch_size + 1}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    // Create the cu_mask_rows tensor.
    Tensor cu_mask_rows
        = torch::empty({batch_size + 1}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    // Copy actual_q_seqlens from device to host.
    auto actual_q_seqlens_host = actual_q_seqlens.to(torch::kCPU);
    // Get the aligned cols.
    int aligned_cols = roundUp(max_kv_seqlen, int(FLASH_ATTEN_PACKED_MASK_N_ALIGNMENT));
    // Remove paddings in the row dimension.
    int total_aligned_rows = 0;
    int max_q_seqlen = 0;
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++)
    {
        int actual_q_seqlen = actual_q_seqlens_host.index({batch_idx}).item<int>();
        total_aligned_rows += roundUp(actual_q_seqlen, int(FLASH_ATTEN_PACKED_MASK_M_ALIGNMENT));
        max_q_seqlen = std::max(max_q_seqlen, packed_mask_input ? actual_q_seqlen : int(mask_input.size(1)));
    }
    // Create the packed_mask tensor.
    Tensor packed_mask = torch::empty({total_aligned_rows, aligned_cols / NUM_POSITIONS_IN_UINT32},
        torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    // Set the parameters for creating packed mask.
    PackedMaskParams<T> maskParams{};
    maskParams.maskInput = get_ptr<T const>(mask_input);
    // It assumes the mask input shape is [batchSize, maxQSeqLen, maxKvSeqLen] when nullptr is given, otherwise
    // [batchSize, maxQSeqLen] is packed.
    maskParams.cuQSeqLens = packed_mask_input ? get_ptr<int>(cu_q_seqlens) : nullptr;
    maskParams.packedMask = get_ptr<uint32_t>(packed_mask);
    maskParams.cuMaskRows = get_ptr<int>(cu_mask_rows);
    maskParams.actualQSeqLens = get_ptr<int>(actual_q_seqlens);
    maskParams.actualKvSeqLens = get_ptr<int>(actual_kv_seqlens);
    maskParams.batchSize = batch_size;
    maskParams.maxQSeqLen = max_q_seqlen;
    maskParams.maxKvSeqLen = max_kv_seqlen;
    maskParams.attentionMaskType = ContextAttentionMaskType::CUSTOM_MASK;
    maskParams.validPosVal = T(valid_pos_value);

    // Launch the pack mask kernel.
    auto stream = at::cuda::getCurrentCUDAStream(mask_input.get_device());
    invokeBuildPackedMask(maskParams, stream);
    sync_check_cuda_error(stream);

    return packed_mask;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Tensor pack_fmha_mask_by_input(
    Tensor mask_input, Tensor actual_q_seqlens, Tensor actual_kv_seqlens, double const valid_pos_value)
{
    if (mask_input.scalar_type() == at::ScalarType::Float)
    {
        return pack_fmha_mask_by_input_helper<float>(mask_input, actual_q_seqlens, actual_kv_seqlens, valid_pos_value);
    }
    else if (mask_input.scalar_type() == at::ScalarType::Half)
    {
        return pack_fmha_mask_by_input_helper<half>(mask_input, actual_q_seqlens, actual_kv_seqlens, valid_pos_value);
    }
#ifdef ENABLE_BF16
    else if (mask_input.scalar_type() == at::ScalarType::BFloat16)
    {
        return pack_fmha_mask_by_input_helper<__nv_bfloat16>(
            mask_input, actual_q_seqlens, actual_kv_seqlens, valid_pos_value);
    }
#endif
    else if (mask_input.scalar_type() == at::ScalarType::Bool)
    {
        return pack_fmha_mask_by_input_helper<bool>(mask_input, actual_q_seqlens, actual_kv_seqlens, valid_pos_value);
    }
    else if (mask_input.scalar_type() == at::ScalarType::Int)
    {
        return pack_fmha_mask_by_input_helper<int>(mask_input, actual_q_seqlens, actual_kv_seqlens, valid_pos_value);
    }
    else
    {
        TORCH_CHECK(false, "Invalid datatype. mask input must be BF16/FP16/FP32/Bool/INT32");
        return Tensor{};
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace torch_ext

////////////////////////////////////////////////////////////////////////////////////////////////////

// Utility methods.
static auto pack_fmha_mask_by_type
    = torch::RegisterOperators("tensorrt_llm::pack_fmha_mask_by_type", &torch_ext::pack_fmha_mask_by_type);

// Utility methods.
static auto pack_fmha_mask_by_input
    = torch::RegisterOperators("tensorrt_llm::pack_fmha_mask_by_input", &torch_ext::pack_fmha_mask_by_input);
