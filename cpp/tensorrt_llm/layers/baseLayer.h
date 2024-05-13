/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/allocator.h"
#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/runtime/common.h"

namespace tensorrt_llm
{
namespace layers
{

class BaseLayer
{
public:
    using SizeType = runtime::SizeType;
    using TokenIdType = runtime::TokenIdType;
    using SizeType32 = runtime::SizeType32;

    BaseLayer(DecoderDomain const& decoderDomain, cudaStream_t stream,
        std::shared_ptr<tensorrt_llm::common::IAllocator> allocator)
        : mStream(stream)
        , mAllocator(std::move(allocator))
        , mDecoderDomain(decoderDomain)
    {
    }

    virtual ~BaseLayer() = default;

    //! @returns cuda stream associated with layer
    [[nodiscard]] cudaStream_t getStream() const noexcept
    {
        return mStream;
    }

    //! @brief set stream to the layer
    void setStream(cudaStream_t stream) noexcept
    {
        mStream = stream;
    }

    //! @returns workspace needed for this layer in bytes
    [[nodiscard]] size_t getWorkspaceSize() const noexcept
    {
        return mWorkspaceSize;
    }

    //! @returns size of memory allocated by layer in bytes
    [[nodiscard]] size_t getAllocatedSize() const noexcept
    {
        return mAllocatedSize;
    }

    // clang-format off
    //! \brief Virtual function to setup internal states of the layer with sampling params
    //! specified in setupParams for the entries specified by batchSlots.
    //! It updates data for new requests in internal tensors inplace.
    //! Thus, it must be called only once for new requests.
    //!
    //! \param batchSize current batch size configured in the system
    //! \param beamWidth current beam width configured in the system
    //! \param batchSlots input tensor [maxBatchSize], address map of the new requests, in pinned memory
    //! \param setupParams shared pointer to params inherited from BaseSetupParams
    // clang-format on
    virtual void setup(runtime::SizeType batchSize, runtime::SizeType beamWidth, runtime::SizeType const* batchSlots,
        std::shared_ptr<BaseSetupParams> setupParams)
        = 0;

    // clang-format off
    //! \brief Virtual function to execute layer.
    //!
    //! \param outputs shared pointer to params inherited from BaseOutputParams
    //! \param inputs shared pointer to params inherited from BaseForwardParams
    // clang-format on
    virtual void forward(std::shared_ptr<BaseOutputParams> outputs, std::shared_ptr<BaseInputParams> inputs) = 0;

protected:
    // Cuda stream
    cudaStream_t mStream;
    // Memory allocator
    std::shared_ptr<tensorrt_llm::common::IAllocator> mAllocator;

    // Required workspace size in bytes
    size_t mWorkspaceSize{0};
    // Allocated memory size in bytes
    size_t mAllocatedSize{0};

    // Domain in which token decoding is computed
    DecoderDomain mDecoderDomain;

    // TODO to be deprecated
    bool mIsAllocateBuffer{false};
};

} // namespace layers
} // namespace tensorrt_llm
