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

void logitsBitmask(torch::Tensor const& logits, torch::Tensor const& bitmask,
    at::optional<torch::Tensor> const& tokenMask = at::nullopt, at::optional<torch::Tensor> const& d2t = at::nullopt)
{
    int32_t const batchSize = logits.size(0);
    if (batchSize == 0)
    {
        return;
    }
    TORCH_CHECK(bitmask.size(0) == batchSize, "bitmask must have the same batch size as logits.");

    int32_t vocabSizePadded = logits.size(1);
    int32_t bitmaskSize = bitmask.size(1);
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor.");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous.");
    TORCH_CHECK(logits.dim() == 2, "logits must be a 2D tensor.");
    TORCH_CHECK(bitmask.is_cuda(), "bitmask must be a CUDA tensor.");
    TORCH_CHECK(bitmask.is_contiguous(), "bitmask must be contiguous.");
    TORCH_CHECK(bitmask.dim() == 2, "bitmask must be a 2D tensor.");
    TORCH_CHECK(bitmask.scalar_type() == torch::kUInt32 || bitmask.scalar_type() == torch::kInt32,
        "bitmask must have element type uint32 or int32.");

    int32_t const* tokenMaskPtr = nullptr;
    if (tokenMask.has_value())
    {
        TORCH_CHECK(tokenMask->is_cuda(), "tokenMask must be a CUDA tensor.");
        TORCH_CHECK(tokenMask->is_contiguous(), "tokenMask must be contiguous.");
        TORCH_CHECK(tokenMask->dim() == 1, "tokenMask must be a 1D tensor.");
        TORCH_CHECK(tokenMask->size(0) == batchSize, "tokenMask must have the same batch size as logits.");
        TORCH_CHECK(tokenMask->scalar_type() == torch::kInt32, "tokenMask must have element type int32.");
        tokenMaskPtr = reinterpret_cast<int32_t const*>(tokenMask->data_ptr());
    }

    int32_t const* d2tPtr = nullptr;
    if (d2t.has_value())
    {
        TORCH_CHECK(d2t->is_cuda(), "d2t must be a CUDA tensor.");
        TORCH_CHECK(d2t->is_contiguous(), "d2t must be contiguous.");
        TORCH_CHECK(d2t->dim() == 1, "d2t must be a 1D tensor.");
        TORCH_CHECK(d2t->size(0) == vocabSizePadded, "d2t must have the same vocab size as logits.");
        TORCH_CHECK(d2t->scalar_type() == torch::kInt32, "d2t must have element type int32.");
        d2tPtr = reinterpret_cast<int32_t const*>(d2t->data_ptr());
    }

    auto stream = at::cuda::getCurrentCUDAStream(logits.get_device()).stream();

    switch (logits.scalar_type())
    {
    case torch::kFloat32:
    {
        tensorrt_llm::kernels::invokeContiguousLogitsBitmask<float>(reinterpret_cast<float*>(logits.data_ptr()),
            reinterpret_cast<uint32_t const*>(bitmask.data_ptr()), tokenMaskPtr, d2tPtr, batchSize, vocabSizePadded,
            bitmaskSize, stream);
        break;
    }
    case torch::kFloat16:
    {
        tensorrt_llm::kernels::invokeContiguousLogitsBitmask<__half>(reinterpret_cast<__half*>(logits.data_ptr()),
            reinterpret_cast<uint32_t const*>(bitmask.data_ptr()), tokenMaskPtr, d2tPtr, batchSize, vocabSizePadded,
            bitmaskSize, stream);
        break;
    }
    case torch::kBFloat16:
    {
        tensorrt_llm::kernels::invokeContiguousLogitsBitmask<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16*>(logits.data_ptr()), reinterpret_cast<uint32_t const*>(bitmask.data_ptr()),
            tokenMaskPtr, d2tPtr, batchSize, vocabSizePadded, bitmaskSize, stream);
        break;
    }
    default: TORCH_CHECK(false, "logits dtype must be float, half or bfloat16."); break;
    }
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("logits_bitmask(Tensor(a!) logits, Tensor bitmask, Tensor? token_mask=None, Tensor? d2t=None) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("logits_bitmask", &torch_ext::logitsBitmask);
}
