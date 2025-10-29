
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

#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/lora/lora.h"
#include "tensorrt_llm/kernels/selectiveScan/selectiveScan.h"
#include "tensorrt_llm/thop/thUtils.h"

namespace th = torch;
namespace tk = tensorrt_llm::kernels;
using tensorrt_llm::common::fmtstr;

namespace torch_ext
{

enum class RequestType : int32_t
{
    kCONTEXT = 0,
    kGENERATION = 1
};

int64_t getNumTokens(th::Tensor const& input)
{
    int ndim = input.sizes().size();
    TLLM_CHECK_WITH_INFO(
        3 == ndim || 2 == ndim, "hidden_state dimension should be either 2 [numTokens, hidden], or 3 [b, s, hidden]");
    int64_t num_tokens = input.sizes()[0];
    if (ndim == 3)
    {
        num_tokens *= input.sizes()[1];
    }
    return num_tokens;
}

std::vector<th::Tensor> lora_grouped_gemm(th::Tensor const& input, th::Tensor const& host_request_types,
    std::vector<th::Tensor> const& lora_ranks, // numModules tensors, each tensors has single value
    std::vector<th::Tensor> const& lora_weights_pointers, th::Tensor const& host_context_lengths,
    std::vector<int64_t> const& output_hidden_sizes, bool transA, bool transB, int64_t const max_low_rank,
    int64_t const& weight_index, bool isRemoveInputPadding)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto const numReqs = lora_ranks[0].sizes()[0];
    auto const out_shape = input.sizes();
    int const numLoraModules = lora_ranks.size();
    TLLM_CHECK_WITH_INFO(lora_ranks.size() == lora_weights_pointers.size(), "both should be numLoraModules");
    std::vector<th::Tensor> output_torch;
    for (int i = 0; i < numLoraModules; i++)
    {
        std::vector<int64_t> output_shape = {out_shape[0], output_hidden_sizes[i]};
        if (!isRemoveInputPadding)
        {
            output_shape = {out_shape[0], out_shape[1], output_hidden_sizes[i]};
        }
        output_torch.push_back(torch::empty(output_shape, input.options()));
    }
    std::vector<void*> output;
    for (auto tensor_it = output_torch.begin(); tensor_it != output_torch.end(); tensor_it++)
    {
        output.push_back(tensor_it->data_ptr());
    }
    int const seqLen = isRemoveInputPadding ? 0 : input.sizes()[1];
    int32_t const* reqTypes = static_cast<int32_t const*>(host_request_types.data_ptr());
    int32_t const* hostContextLengths
        = isRemoveInputPadding ? static_cast<int32_t const*>(host_context_lengths.data_ptr()) : nullptr;

    int64_t numTokens = getNumTokens(input);

    std::vector<void const*> expandLoraWeightPtrs{};
    std::vector<int32_t> expandLoraRanks{};

    expandLoraWeightPtrs.reserve(numLoraModules * numTokens * 2);
    expandLoraRanks.reserve(numLoraModules * numTokens);

    for (int loraModuleIdx = 0; loraModuleIdx < numLoraModules; loraModuleIdx++)
    {
        auto const loraRankModule = static_cast<int32_t const*>(lora_ranks[loraModuleIdx].data_ptr());
        auto const loraWeightModulePtrs = static_cast<int64_t const*>(lora_weights_pointers[loraModuleIdx].data_ptr());

        int idx = 0;
        for (int reqId = 0; reqId < numReqs; reqId++)
        {
            // loraWeightModulePtrs has 3 pointers for each module: A,B, and an optional DoRA magnitude
            // the current LoRA plugin does not apply DoRA scaling, so the magnitude is ignored
            RequestType const reqType = static_cast<RequestType const>(reqTypes[reqId]);
            if (reqType == RequestType::kGENERATION)
            {
                expandLoraWeightPtrs.push_back(reinterpret_cast<void const*>(loraWeightModulePtrs[reqId * 3]));
                expandLoraWeightPtrs.push_back(reinterpret_cast<void const*>(loraWeightModulePtrs[reqId * 3 + 1]));
                expandLoraRanks.push_back(loraRankModule[reqId]);
                idx += 1;
            }
            else
            {
                int contextLen = (isRemoveInputPadding ? hostContextLengths[reqId] : seqLen);

                for (int contextId = 0; contextId < contextLen; contextId++)
                {
                    expandLoraWeightPtrs.push_back(reinterpret_cast<void const*>(loraWeightModulePtrs[reqId * 3]));
                    expandLoraWeightPtrs.push_back(reinterpret_cast<void const*>(loraWeightModulePtrs[reqId * 3 + 1]));
                    expandLoraRanks.push_back(loraRankModule[reqId]);
                    idx += 1;
                }
            }
        }

        // In 1st generation phase cross attention qkv lora, cross qkv is skipped by passing an empty encoder_output
        // (passing 0 to dim) getNumTokens() will get in cross qkv_lora. Skipping the check for this case.
        if (numTokens > 0)
        {
            TLLM_CHECK_WITH_INFO(idx == numTokens,
                fmtstr("LoraParams and input dims don't match, lora tokens %d input tokens %ld", idx, numTokens));
        }
    }

    thread_local std::shared_ptr<tensorrt_llm::common::CublasMMWrapper> cublasWrapper;
    if (cublasWrapper == nullptr)
    {
        auto cublasHandle = getCublasHandle();
        auto cublasLtHandle = getCublasLtHandle();
        cublasWrapper
            = std::make_shared<tensorrt_llm::common::CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);
    }

    int const inHiddenSize = input.sizes()[input.sizes().size() - 1];

    std::vector<int> outHiddenSizes(output_hidden_sizes.size());
    for (int i = 0; i < numLoraModules; i++)
    {
        outHiddenSizes[i] = output_hidden_sizes[i];
    }
    nvinfer1::DataType loraRuntimeDataType;
    switch (input.scalar_type())
    {
    case torch::kFloat16: loraRuntimeDataType = nvinfer1::DataType::kHALF; break;
    case torch::kBFloat16: loraRuntimeDataType = nvinfer1::DataType::kBF16; break;
    default: throw std::invalid_argument("Invalid dtype, only supports float16, bfloat16");
    }

    auto mLoraImpl = std::make_shared<tensorrt_llm::kernels::LoraImpl>(
        inHiddenSize, outHiddenSizes, transA, transB, numLoraModules, loraRuntimeDataType, max_low_rank, cublasWrapper);

    // TODO (dafrimi): use Profiler to find the best tactic as used in lora_plugin
    mLoraImpl->setBestTactic(std::nullopt);

    auto const workspace_size = mLoraImpl->getWorkspaceSize(numTokens, numReqs, loraRuntimeDataType);

    auto workspace = torch::empty(std::vector<int64_t>{static_cast<int64_t>(workspace_size)}, input.options());

    mLoraImpl->run(numTokens, numReqs, input.data_ptr(), expandLoraRanks.data(), expandLoraWeightPtrs.data(),
        weight_index, output.data(), workspace.data_ptr(), stream);
    sync_check_cuda_error(stream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return output_torch;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "lora_grouped_gemm(Tensor input, "
        "Tensor host_request_types, "
        "Tensor [] lora_ranks, "
        "Tensor [] lora_weights_pointers, "
        "Tensor host_context_lengths, "
        "int [] output_hidden_sizes, "
        "bool transA, "
        "bool transB, "
        "int max_low_rank, "
        "int weight_index, "
        "bool isRemoveInputPadding) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("lora_grouped_gemm", &torch_ext::lora_grouped_gemm);
}
