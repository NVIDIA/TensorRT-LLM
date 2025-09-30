/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <ATen/ATen.h>

#include <stdexcept>

namespace tensorrt_llm::runtime
{

class Torch
{
public:
    static at::Tensor tensor(ITensor::SharedPtr tensor)
    {
        auto const tensorOptions = at::device(TorchUtils::device((*tensor).data()))
                                       .pinned_memory((*tensor).getMemoryType() == MemoryType::kPINNEDPOOL)
                                       .dtype(TorchUtils::dataType((*tensor).getDataType()))
                                       .layout(at::kStrided);
        return at::for_blob(tensor->data(), TorchUtils::shape(tensor->getShape())) // NOLINT(*-use-after-move)
            .options(tensorOptions)
            .deleter(
                [ptr = std::move(tensor)](void* data) mutable
                {
                    if (data != ptr->data())
                    {
                        TLLM_LOG_WARNING("Torch tensor refers to deallocated memory.");
                    }
                    ptr.reset();
                })
            .make_tensor();
    }

    static at::Tensor buffer(IBuffer::SharedPtr buffer)
    {
        auto const shape = ITensor::makeShape({static_cast<runtime::SizeType32>(buffer->getSize())});
        return tensor(ITensor::view(std::move(buffer), shape));
    }

    static void setCurrentStream(runtime::CudaStream& cudaStream)
    {
        at::cuda::setCurrentCUDAStream(TorchUtils::stream(cudaStream));
    }

private:
    Torch() = default;
};

} // namespace tensorrt_llm::runtime
