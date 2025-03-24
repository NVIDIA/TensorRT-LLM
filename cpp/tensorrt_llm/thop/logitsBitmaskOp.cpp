/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/logitsBitmask.h"
#include "tensorrt_llm/thop/thUtils.h"

namespace torch_ext
{

int32_t constexpr kBitsPerMaskElement = 32;

void logitsBitmask(std::vector<torch::Tensor> const& logits, std::vector<torch::Tensor> const& bitmask)
{
    TORCH_CHECK(bitmask.size() == logits.size(), "The lengths of logits and bitmask do not match.");
    int32_t batchSize = logits.size();
    if (batchSize == 0)
    {
        return;
    }
    int32_t vocabSizePadded = logits[0].size(0);
    int32_t bitmaskSize = tensorrt_llm::common::ceilDiv(vocabSizePadded, kBitsPerMaskElement);

    auto options = torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCPU).pinned_memory(true);
    auto logitsPtrsHost = torch::empty({batchSize}, options);
    auto bitmaskPtrsHost = torch::empty({batchSize}, options);
    for (int i = 0; i < batchSize; i++)
    {
        TORCH_CHECK(logits[i].is_cuda(), "logits must be a CUDA tensor.");
        TORCH_CHECK(logits[i].is_contiguous(), "logits must be contiguous.");
        TORCH_CHECK(logits[i].dim() == 1, "logits must be a 1D tensor.");
        TORCH_CHECK(logits[i].size(0) == vocabSizePadded, "logits must have the same vocab size.");
        TORCH_CHECK(bitmask[i].is_cuda(), "bitmask must be a CUDA tensor.");
        TORCH_CHECK(bitmask[i].is_contiguous(), "bitmask must be contiguous.");
        TORCH_CHECK(bitmask[i].dim() == 1, "bitmask must be a 1D tensor.");
        TORCH_CHECK(bitmask[i].size(0) == bitmaskSize, "bitmask must have size equal to ceilDiv(vocab_size, 32).");
        TORCH_CHECK(bitmask[i].scalar_type() == torch::kUInt32 || bitmask[i].scalar_type() == torch::kInt32,
            "bitmask must have element type uint32 or int32.");

        logitsPtrsHost[i] = reinterpret_cast<uint64_t>(logits[i].data_ptr());
        bitmaskPtrsHost[i] = reinterpret_cast<uint64_t>(bitmask[i].data_ptr());
    }

    auto logitsPtrs = logitsPtrsHost.to(torch::kCUDA);
    auto bitmaskPtrs = bitmaskPtrsHost.to(torch::kCUDA);

    auto stream = at::cuda::getCurrentCUDAStream(logits[0].get_device()).stream();

    switch (logits[0].scalar_type())
    {
    case torch::kFloat32:
    {
        tensorrt_llm::kernels::invokeLogitsBitmask<float>(reinterpret_cast<float**>(logitsPtrs.data_ptr()),
            reinterpret_cast<uint32_t const**>(bitmaskPtrs.data_ptr()), batchSize, vocabSizePadded, stream);
        break;
    }
    case torch::kFloat16:
    {
        tensorrt_llm::kernels::invokeLogitsBitmask<__half>(reinterpret_cast<__half**>(logitsPtrs.data_ptr()),
            reinterpret_cast<uint32_t const**>(bitmaskPtrs.data_ptr()), batchSize, vocabSizePadded, stream);
        break;
    }
    case torch::kBFloat16:
    {
        tensorrt_llm::kernels::invokeLogitsBitmask<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16**>(logitsPtrs.data_ptr()),
            reinterpret_cast<uint32_t const**>(bitmaskPtrs.data_ptr()), batchSize, vocabSizePadded, stream);
        break;
    }
    default: TORCH_CHECK(false, "logits dtype must be float, half or bfloat16."); break;
    }
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("logits_bitmask(Tensor[] logits, Tensor[] bitmask) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("logits_bitmask", &torch_ext::logitsBitmask);
}
