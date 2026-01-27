
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
#include "tensorrt_llm/kernels/cuda_graph_grouped_gemm.h"
#include "tensorrt_llm/kernels/lora/lora.h"
#include "tensorrt_llm/kernels/lora/loraGroupGEMMParamFillRowReorderFusion.h"
#include "tensorrt_llm/kernels/selectiveScan/selectiveScan.h"
#include "tensorrt_llm/thop/thUtils.h"

namespace th = torch;
namespace tk = tensorrt_llm::kernels;
using tensorrt_llm::common::fmtstr;
namespace tc = tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

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

void lora_grouped_gemm_cuda_graph(th::Tensor const& lora_in_sizes, // [layer_module_num, max_lora_size, 3]
    th::Tensor const& lora_out_sizes,                              // [layer_module_num, max_lora_size, 3]
    th::Tensor const& a_offsets,                                   // [layer_module_num, max_lora_size]
    th::Tensor const& b_ptrs,                                      // [layer_module_num, max_lora_size]
    th::Tensor const& d_offsets,                                   // [layer_module_num, max_lora_size]
    th::Tensor const& b_prime_ptrs,                                // [layer_module_num, max_lora_size]
    th::Tensor const& d_prime_offsets,                             // [layer_module_num, max_lora_size]
    int64_t problem_count,
    th::Tensor const& lda, // Leading dimensions for A matrices [layer_module_num, max_lora_size]
    th::Tensor const& ldb, // Leading dimensions for B matrices [layer_module_num, max_lora_size]
    th::Tensor const& ldd, // Leading dimensions for C matrices [layer_module_num, max_lora_size] (unused)
    th::Tensor const& ldb_prime, th::Tensor const& ldd_prime, th::Tensor const& host_max_in_sizes,
    th::Tensor const& host_max_out_sizes, th::Tensor const& splitk_offsets, c10::ScalarType dtype, int64_t minKN,
    int64_t splitKSlices = 16)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    sync_check_cuda_error(stream);

    auto* a_ptrs_gpu = reinterpret_cast<void**>(const_cast<void*>(a_offsets.data_ptr()));
    auto* d_ptrs_gpu = reinterpret_cast<void**>(const_cast<void*>(d_offsets.data_ptr()));
    auto* a_prime_ptrs_gpu = reinterpret_cast<void**>(const_cast<void*>(d_offsets.data_ptr()));
    auto* d_prime_ptrs_gpu = reinterpret_cast<void**>(const_cast<void*>(d_prime_offsets.data_ptr()));

    auto* problem_sizes_1_ptr = reinterpret_cast<cutlass::gemm::GemmCoord*>(lora_in_sizes.data_ptr());
    auto* problem_sizes_2_ptr = reinterpret_cast<cutlass::gemm::GemmCoord*>(lora_out_sizes.data_ptr());

    auto* host_max_in_sizes_ptr = reinterpret_cast<cutlass::gemm::GemmCoord*>(host_max_in_sizes.data_ptr());
    auto* host_max_out_sizes_ptr = reinterpret_cast<cutlass::gemm::GemmCoord*>(host_max_out_sizes.data_ptr());

    auto* b_ptrs_gpu = reinterpret_cast<void**>(const_cast<void*>(b_ptrs.data_ptr()));
    auto* b_prime_ptrs_gpu = reinterpret_cast<void**>(const_cast<void*>(b_prime_ptrs.data_ptr()));

    auto* lda_gpu = reinterpret_cast<int64_t*>(const_cast<void*>(lda.data_ptr()));
    auto* ldb_gpu = reinterpret_cast<int64_t*>(const_cast<void*>(ldb.data_ptr()));
    auto* ldd_gpu = reinterpret_cast<int64_t*>(const_cast<void*>(ldd.data_ptr()));
    auto* ldb_prime_gpu = reinterpret_cast<int64_t*>(const_cast<void*>(ldb_prime.data_ptr()));
    auto* ldd_prime_gpu = reinterpret_cast<int64_t*>(const_cast<void*>(ldd_prime.data_ptr()));

    auto* splitk_offsets_gpu = reinterpret_cast<int64_t*>(const_cast<void*>(splitk_offsets.data_ptr()));

    // Get data type
    nvinfer1::DataType loraRuntimeDataType;
    switch (dtype)
    {
    case torch::kFloat16: loraRuntimeDataType = nvinfer1::DataType::kHALF; break;
    case torch::kBFloat16: loraRuntimeDataType = nvinfer1::DataType::kBF16; break;
    default: TORCH_CHECK(false, "Invalid dtype, only supports float16, bfloat16, got %s", c10::toString(dtype));
    }

    int const minKnInt = std::max(1, static_cast<int>(minKN));

    // Call CUDA Graph compatible grouped GEMM for lora_in (split-K)
    if (problem_count > 0)
    {
        TLLM_LOG_TRACE("Start Grouped GEMM for LoRA in.");

        tk::cudaGraphSplitKGroupedGemm(problem_sizes_1_ptr, problem_count, a_ptrs_gpu, b_ptrs_gpu,
            d_ptrs_gpu,                                     // ptrC (no bias)
            d_ptrs_gpu, lda_gpu, ldb_gpu, ldd_gpu, ldd_gpu, // Precomputed leading dimensions
            true,                                           // isLoraIn
            loraRuntimeDataType,
            static_cast<int>(splitKSlices),                 // splitKSlices
            minKnInt,                                       // minKN
            host_max_in_sizes_ptr, splitk_offsets_gpu, stream);
        sync_check_cuda_error(stream);

        // Call CUDA Graph compatible grouped GEMM for lora_out
        TLLM_LOG_TRACE("Start Grouped GEMM for LoRA out.");
        tk::cudaGraphGroupedGemm(problem_sizes_2_ptr, problem_count, a_prime_ptrs_gpu, b_prime_ptrs_gpu,
            d_prime_ptrs_gpu,                                                       // ptrC (no bias)
            d_prime_ptrs_gpu, ldd_gpu, ldb_prime_gpu, ldd_prime_gpu, ldd_prime_gpu, // Precomputed leading dimensions
            false,                                                                  // isLoraIn
            loraRuntimeDataType,
            minKnInt,                                                               // minKN
            host_max_out_sizes_ptr, stream);
        sync_check_cuda_error(stream);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void lora_group_gemm_param_fill_row_reorder_fusion(th::Tensor const& in_sizes, // [module_count, max_lora_count, 3]
    th::Tensor const& out_sizes,                                               // [module_count, max_lora_count, 3]
    th::Tensor const& a_ptrs,                                                  // [module_count, max_lora_count]
    th::Tensor const& d_ptrs,                                                  // [module_count, max_lora_count]
    th::Tensor const& d_prime_ptrs,                                            // [module_count, max_lora_count]
    th::Tensor const& lda,                                                     // [module_count, max_lora_count]
    th::Tensor const& ldd,                                                     // [module_count, max_lora_count]
    th::Tensor const& ldb_prime,                                               // [module_count, max_lora_count]
    th::Tensor const& ldd_prime,                                               // [module_count, max_lora_count]
    th::Tensor const& splitk_offsets,                                          // [module_count, max_lora_count]
    th::Tensor const& reordered_input,                                         // [batch_size, input_hidden_size]
    int64_t max_lora_count, int64_t max_lora_rank, int64_t sum_output_hidden_size, int64_t input_hidden_size,
    int64_t batch_size,
    th::Tensor const& slot_counts,         // [max_lora_count]
    th::Tensor const& slot_ranks,          // [max_lora_count]
    th::Tensor const& slot_offsets,        // [max_lora_count + 1]
    th::Tensor const& module_out_sizes,    // [module_count]
    th::Tensor const& module_out_prefix,   // [module_count]
    th::Tensor const& b_ptrs,              // [module_count, max_lora_count]
    th::Tensor const& b_prime_ptrs,        // [module_count, max_lora_count]
    th::Tensor const& input,               // [batch_size, input_hidden_size]
    th::Tensor const& sorted_ids,          // [batch_size]
    th::Tensor const& intermediate_buffer, // [batch_size, max_lora_rank]
    th::Tensor const& output_buffer,       // [batch_size, sum_output_hidden_size]
    c10::ScalarType dtype)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Validate inputs
    TORCH_CHECK(in_sizes.device().is_cuda(), "in_sizes must be a CUDA tensor");
    TORCH_CHECK(out_sizes.device().is_cuda(), "out_sizes must be a CUDA tensor");
    TORCH_CHECK(reordered_input.device().is_cuda(), "reordered_input must be a CUDA tensor");
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");

    // Get module count from tensor shapes
    int32_t const module_count = static_cast<int32_t>(in_sizes.size(0));

    // Get data type info
    nvinfer1::DataType loraRuntimeDataType;
    switch (dtype)
    {
    case torch::kFloat16: loraRuntimeDataType = nvinfer1::DataType::kHALF; break;
    case torch::kBFloat16: loraRuntimeDataType = nvinfer1::DataType::kBF16; break;
    default: TORCH_CHECK(false, "Invalid dtype, only supports float16, bfloat16, got %s", c10::toString(dtype));
    }

    int64_t const dtype_element_size = input.element_size();

    int64_t const a_base_ptr = reinterpret_cast<int64_t>(reordered_input.data_ptr());
    int64_t const d_base_ptr = reinterpret_cast<int64_t>(intermediate_buffer.data_ptr());
    int64_t const d_prime_base_ptr = reinterpret_cast<int64_t>(output_buffer.data_ptr());

    tk::launchLoraGroupGEMMParamFillRowReorderFusion(reinterpret_cast<int32_t*>(const_cast<void*>(in_sizes.data_ptr())),
        reinterpret_cast<int32_t*>(const_cast<void*>(out_sizes.data_ptr())),
        reinterpret_cast<int64_t*>(const_cast<void*>(a_ptrs.data_ptr())),
        reinterpret_cast<int64_t*>(const_cast<void*>(d_ptrs.data_ptr())),
        reinterpret_cast<int64_t*>(const_cast<void*>(d_prime_ptrs.data_ptr())),
        reinterpret_cast<int64_t*>(const_cast<void*>(lda.data_ptr())),
        reinterpret_cast<int64_t*>(const_cast<void*>(ldd.data_ptr())),
        reinterpret_cast<int64_t*>(const_cast<void*>(ldb_prime.data_ptr())),
        reinterpret_cast<int64_t*>(const_cast<void*>(ldd_prime.data_ptr())),
        reinterpret_cast<int64_t*>(const_cast<void*>(splitk_offsets.data_ptr())),
        const_cast<void*>(reordered_input.data_ptr()), static_cast<int32_t>(max_lora_count),
        static_cast<int32_t>(max_lora_rank), static_cast<int32_t>(sum_output_hidden_size),
        static_cast<int32_t>(input_hidden_size), dtype_element_size, batch_size, a_base_ptr, d_base_ptr,
        d_prime_base_ptr, reinterpret_cast<int32_t const*>(slot_counts.data_ptr()),
        reinterpret_cast<int32_t const*>(slot_ranks.data_ptr()),
        reinterpret_cast<int64_t const*>(slot_offsets.data_ptr()),
        reinterpret_cast<int32_t const*>(module_out_sizes.data_ptr()),
        reinterpret_cast<int64_t const*>(module_out_prefix.data_ptr()),
        reinterpret_cast<int64_t const*>(b_ptrs.data_ptr()), reinterpret_cast<int64_t const*>(b_prime_ptrs.data_ptr()),
        input.data_ptr(), reinterpret_cast<int64_t const*>(sorted_ids.data_ptr()), module_count, loraRuntimeDataType,
        stream);

    sync_check_cuda_error(stream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

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

    m.def(
        "lora_grouped_gemm_cuda_graph("
        "Tensor lora_in_sizes, "
        "Tensor lora_out_sizes, "
        "Tensor a_offsets, "
        "Tensor b_ptrs, "
        "Tensor d_offsets, "
        "Tensor b_prime_ptrs, "
        "Tensor d_prime_offsets, "
        "int problem_count, "
        "Tensor lda, "
        "Tensor ldb, "
        "Tensor ldd, "
        "Tensor ldb_prime, "
        "Tensor ldd_prime, "
        "Tensor host_max_in_sizes, "
        "Tensor host_max_out_sizes, "
        "Tensor splitk_offsets, "
        "ScalarType dtype, "
        "int minKN, "
        "int splitKSlices=16) -> ()");

    m.def(
        "lora_group_gemm_param_fill_row_reorder_fusion("
        "Tensor in_sizes, "
        "Tensor out_sizes, "
        "Tensor a_ptrs, "
        "Tensor d_ptrs, "
        "Tensor d_prime_ptrs, "
        "Tensor lda, "
        "Tensor ldd, "
        "Tensor ldb_prime, "
        "Tensor ldd_prime, "
        "Tensor splitk_offsets, "
        "Tensor reordered_input, "
        "int max_lora_count, "
        "int max_lora_rank, "
        "int sum_output_hidden_size, "
        "int input_hidden_size, "
        "int batch_size, "
        "Tensor slot_counts, "
        "Tensor slot_ranks, "
        "Tensor slot_offsets, "
        "Tensor module_out_sizes, "
        "Tensor module_out_prefix, "
        "Tensor b_ptrs, "
        "Tensor b_prime_ptrs, "
        "Tensor input, "
        "Tensor sorted_ids, "
        "Tensor intermediate_buffer, "
        "Tensor output_buffer, "
        "ScalarType dtype) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("lora_grouped_gemm", &tensorrt_llm::torch_ext::lora_grouped_gemm);
    m.impl("lora_grouped_gemm_cuda_graph", &tensorrt_llm::torch_ext::lora_grouped_gemm_cuda_graph);
    m.impl("lora_group_gemm_param_fill_row_reorder_fusion",
        &tensorrt_llm::torch_ext::lora_group_gemm_param_fill_row_reorder_fusion);
}
