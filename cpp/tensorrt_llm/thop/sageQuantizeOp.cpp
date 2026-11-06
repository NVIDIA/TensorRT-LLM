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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/sageQuant.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstdint>
#include <optional>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

using tensorrt_llm::common::SageQuantParams;
using tensorrt_llm::common::SageScalePartition;

void checkTokenMajor(torch::Tensor const& tensor, char const* name)
{
    TORCH_CHECK(tensor.is_cuda() && tensor.dim() == 3 && tensor.is_contiguous(), name,
        " must be a contiguous CUDA [tokens, heads, head_dim] tensor");
}

void checkCuSeqLens(torch::Tensor const& tensor, char const* name)
{
    TORCH_CHECK(tensor.is_cuda() && tensor.dim() == 1 && tensor.is_contiguous() && tensor.scalar_type() == torch::kInt32
            && tensor.size(0) > 1,
        name, " must be a contiguous CUDA int32 [batch_size + 1] tensor");
}

// The kernel reads the V scales and the means with 16-byte vector loads.
void checkScale(torch::Tensor const& tensor, int64_t numel, char const* name)
{
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() && tensor.scalar_type() == torch::kFloat32
            && tensor.numel() == numel && reinterpret_cast<std::uintptr_t>(tensor.data_ptr()) % 16 == 0,
        name, " must be a contiguous, 16-byte aligned CUDA float32 tensor with ", numel, " elements");
}

void checkKeyValue(torch::Tensor const& key, torch::Tensor const& value, torch::Tensor const& k_quant,
    torch::Tensor const& v_quant, torch::Tensor const& cu_seqlens_kv)
{
    checkTokenMajor(key, "key");
    checkTokenMajor(value, "value");
    TORCH_CHECK(
        key.scalar_type() == torch::kBFloat16 || key.scalar_type() == torch::kFloat16, "key must be BF16 or FP16");
    TORCH_CHECK(value.scalar_type() == key.scalar_type(), "key and value must share a dtype");
    TORCH_CHECK(key.sizes() == value.sizes(), "key and value must have the same shape");
    checkCuSeqLens(cu_seqlens_kv, "cu_seqlens_kv");
    checkTokenMajor(k_quant, "k_quant");
    checkTokenMajor(v_quant, "v_quant");
    TORCH_CHECK(k_quant.sizes() == key.sizes() && v_quant.sizes() == value.sizes(),
        "quantized outputs must match the input shapes");
    TORCH_CHECK(k_quant.scalar_type() == torch::kInt8 || k_quant.scalar_type() == torch::kFloat8_e4m3fn,
        "k_quant must be int8 or float8_e4m3fn");
    TORCH_CHECK(v_quant.scalar_type() == torch::kFloat8_e4m3fn, "v_quant must be float8_e4m3fn");
}

// Scales per head of a Q/K tensor quantized in token_block_size-token blocks.
int checkBlockScales(
    torch::Tensor const& input, torch::Tensor const& scale, int64_t token_block_size, int batchSize, char const* name)
{
    int const stride = tensorrt_llm::common::getSageScaleHeadStride(
        SageScalePartition::Contiguous, static_cast<int>(token_block_size), static_cast<int>(input.size(0)), batchSize);
    TORCH_CHECK(stride > 0, "q_block_size and k_block_size must be 1, 4, 16, or 32");
    checkScale(scale, input.size(1) * stride, name);
    return stride;
}

// The V part and the hardware fields of the kernel parameters.
SageQuantParams makeParams(torch::Tensor const& value, torch::Tensor& v_quant, torch::Tensor const& v_scale,
    torch::Tensor const& qk_quant, int batchSize, int const* ptrCuSeqLensV)
{
    SageQuantParams params{};
    params.headDim = static_cast<int>(value.size(2));
    params.inputType = value.scalar_type() == torch::kBFloat16 ? tensorrt_llm::kernels::DATA_TYPE_BF16
                                                               : tensorrt_llm::kernels::DATA_TYPE_FP16;
    params.quantType = qk_quant.scalar_type() == torch::kInt8 ? tensorrt_llm::kernels::DATA_TYPE_INT8
                                                              : tensorrt_llm::kernels::DATA_TYPE_E4M3;
    params.partition = SageScalePartition::Contiguous;
    params.batchSize = batchSize;
    params.sumSeqLensV = static_cast<int>(value.size(0));
    params.numHeadsV = static_cast<int>(value.size(1));
    params.ptrV = value.data_ptr();
    params.ptrVQuant = v_quant.data_ptr();
    params.ptrVScale = v_scale.data_ptr<float>();
    params.ptrCuSeqLensV = ptrCuSeqLensV;
    params.smCount = tensorrt_llm::common::getMultiProcessorCount();
    params.stream = at::cuda::getCurrentCUDAStream(value.get_device());
    return params;
}

// Quantizes Q or K in token_block_size-token blocks; vStage says what the launch does to V.
void launch(SageQuantParams params, torch::Tensor const& input, torch::Tensor const& cu_seqlens, torch::Tensor& quant,
    torch::Tensor& scale, int64_t token_block_size, int vStage)
{
    params.sumSeqLensQk = static_cast<int>(input.size(0));
    params.ptrCuSeqLensQk = cu_seqlens.data_ptr<int>();
    params.numHeads = static_cast<int>(input.size(1));
    params.tokenBlockSize = static_cast<int>(token_block_size);
    params.ptrQk = input.data_ptr();
    params.ptrQkQuant = quant.data_ptr();
    params.ptrQkScale = scale.data_ptr<float>();
    params.vStage = vStage;
    tensorrt_llm::common::invokeSageQuant(params);
}

} // namespace

// SageAttention Q/K/V quantization with contiguous token-block scales.
//
// Q and K are quantized to INT8 or FP8 E4M3 with one scale per q_block_size / k_block_size tokens
// of a sequence, laid out per head as tensorrt_llm::common::getSageScaleHeadStride describes. V is
// quantized to FP8 E4M3 with one scale per channel of each sequence. The V scales and the means
// are [batch, heads_kv, head_dim]; k_mean and v_mean are the K- and V-smoothing means, subtracted
// before quantization and filled exactly when given; a smoothed V's scales cover V minus its mean.
// Repeated calls on the same GPU write bitwise identical values, the means included.
//
// With reuse_stats, only K and V are quantized, with the V scales and the means of an earlier call
// as inputs, for tensors that must dequantize like that call's K and V, such as the K/V block means
// of PrimTS proxy routes; K still takes its own scales. The query arguments must then be absent.
void sage_quantize(std::optional<torch::Tensor> const& query, torch::Tensor const& key, torch::Tensor const& value,
    std::optional<torch::Tensor> const& cu_seqlens_q, torch::Tensor const& cu_seqlens_kv,
    std::optional<torch::Tensor> q_quant, torch::Tensor& k_quant, torch::Tensor& v_quant,
    std::optional<torch::Tensor> q_scale, torch::Tensor& k_scale, torch::Tensor& v_scale,
    std::optional<torch::Tensor> k_mean, std::optional<torch::Tensor> v_mean, int64_t q_block_size,
    int64_t k_block_size, bool reuse_stats)
{
    checkKeyValue(key, value, k_quant, v_quant, cu_seqlens_kv);
    int const batchSize = static_cast<int>(cu_seqlens_kv.size(0) - 1);
    checkBlockScales(key, k_scale, k_block_size, batchSize, "k_scale");
    int64_t const statRows = batchSize * value.size(1);
    checkScale(v_scale, statRows * value.size(2), "v_scale");
    if (k_mean.has_value())
    {
        checkScale(*k_mean, statRows * key.size(2), "k_mean");
    }
    if (v_mean.has_value())
    {
        checkScale(*v_mean, statRows * value.size(2), "v_mean");
    }

    SageQuantParams params = makeParams(value, v_quant, v_scale, k_quant, batchSize, cu_seqlens_kv.data_ptr<int>());
    params.kSmooth = k_mean.has_value();
    params.ptrKMean = params.kSmooth ? k_mean->data_ptr<float>() : nullptr;
    params.ptrVMean = v_mean.has_value() ? v_mean->data_ptr<float>() : nullptr;
    if (reuse_stats)
    {
        TORCH_CHECK(!query.has_value() && !cu_seqlens_q.has_value() && !q_quant.has_value() && !q_scale.has_value(),
            "reuse_stats quantizes K and V only: query, cu_seqlens_q, q_quant, and q_scale must be absent");
        launch(params, key, cu_seqlens_kv, k_quant, k_scale, k_block_size, 2);
        return;
    }

    TORCH_CHECK(query.has_value() && cu_seqlens_q.has_value() && q_quant.has_value() && q_scale.has_value(),
        "query, cu_seqlens_q, q_quant, and q_scale are required unless reuse_stats is set");
    checkTokenMajor(*query, "query");
    TORCH_CHECK(query->scalar_type() == key.scalar_type(), "query, key, and value must share a dtype");
    TORCH_CHECK(query->size(2) == key.size(2), "query and key must have the same head dimension");
    checkCuSeqLens(*cu_seqlens_q, "cu_seqlens_q");
    TORCH_CHECK(cu_seqlens_q->size(0) == cu_seqlens_kv.size(0), "Q and KV must have the same batch size");
    checkTokenMajor(*q_quant, "q_quant");
    TORCH_CHECK(q_quant->sizes() == query->sizes(), "quantized outputs must match the input shapes");
    TORCH_CHECK(q_quant->scalar_type() == k_quant.scalar_type(), "q_quant and k_quant must share a dtype");
    checkBlockScales(*query, *q_scale, q_block_size, batchSize, "q_scale");

    // The launch initializes the buffers it reduces into: the V scales, and the sums and minima of
    // the smoothing means below.
    std::optional<torch::Tensor> k_mean_sum;
    if (params.kSmooth)
    {
        k_mean_sum = torch::empty_like(*k_mean, torch::kInt64);
        params.ptrKForMean = key.data_ptr();
        params.ptrKMeanSum = k_mean_sum->data_ptr<int64_t>();
    }
    std::optional<torch::Tensor> v_mean_sum;
    std::optional<torch::Tensor> v_min;
    if (v_mean.has_value())
    {
        v_mean_sum = torch::empty_like(*v_mean, torch::kInt64);
        v_min = torch::empty_like(v_scale);
        params.ptrVMeanSum = v_mean_sum->data_ptr<int64_t>();
        params.ptrVMin = v_min->data_ptr<float>();
    }
    // Q, while collecting the V statistics and the K mean; then K (smoothed by the K mean) and V.
    launch(params, *query, *cu_seqlens_q, *q_quant, *q_scale, q_block_size, 1);
    launch(params, key, cu_seqlens_kv, k_quant, k_scale, k_block_size, 2);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "sage_quantize(Tensor? query, Tensor key, Tensor value, Tensor? cu_seqlens_q, Tensor cu_seqlens_kv, "
        "Tensor(a!)? q_quant, Tensor(b!) k_quant, Tensor(c!) v_quant, Tensor(d!)? q_scale, Tensor(e!) k_scale, "
        "Tensor(f!) v_scale, Tensor(g!)? k_mean, Tensor(h!)? v_mean, int q_block_size, int k_block_size, "
        "bool reuse_stats=False) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("sage_quantize", &sage_quantize);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
