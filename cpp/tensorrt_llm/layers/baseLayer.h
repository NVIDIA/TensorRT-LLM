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

#include <utility>

#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/decodingLayerWorkspace.h"

namespace tensorrt_llm::layers
{

class BaseLayer
{
public:
    using SizeType32 = runtime::SizeType32;
    using TokenIdType = runtime::TokenIdType;
    using BufferConstPtr = runtime::IBuffer::SharedConstPtr;
    using BufferPtr = runtime::IBuffer::SharedPtr;
    using TensorConstPtr = runtime::ITensor::SharedConstPtr;
    using TensorPtr = runtime::ITensor::SharedPtr;

    BaseLayer(DecoderDomain decoderDomain, std::shared_ptr<runtime::BufferManager> bufferManager)
        : mBufferManager(std::move(bufferManager))
        , mDecoderDomain(std::move(decoderDomain))
    {
    }

    virtual ~BaseLayer() = default;

    //! @returns cuda stream associated with layer
    [[nodiscard]] cudaStream_t getStream() const noexcept
    {
        return mBufferManager->getStream().get();
    }

    //! @returns workspace needed for this layer in bytes
    [[nodiscard]] virtual size_t getWorkspaceSize() const noexcept
    {
        return 0;
    };

    // clang-format off
    //! \brief Virtual function to setup internal states of the layer with sampling params
    //! specified in setupParams for the entries specified by batchSlots.
    //! It updates data for new requests in internal tensors inplace.
    //! Thus, it must be called only once for new requests.
    //!
    //! \param batchSize current batch size configured in the system
    //! \param beamWidth current beam width configured in the system
    //! \param batchSlots input buffer [maxBatchSize], address map of the new requests, in pinned memory
    //! \param setupParams shared pointer to params inherited from BaseSetupParams
    // clang-format on
    virtual void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
        = 0;

    // clang-format off
    //! \brief Virtual function to execute layer async on GPU.
    //! There must be no stream synchronization inside this function.
    //!
    //! \param outputs shared pointer to params inherited from BaseDecodingOutputs
    //! \param inputs shared pointer to params inherited from BaseForwardParams
    // clang-format on
    virtual void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
        = 0;

    // clang-format off
    //! \brief Virtual function to execute layer synchronously on CPU / GPU.
    //! It is allowed (but not necassary) to synchronize on stream inside this function.
    //! It is targeted mainly for prototyping.
    //!
    //! \param outputs shared pointer to params inherited from BaseDecodingOutputs
    //! \param inputs shared pointer to params inherited from BaseForwardParams
    // clang-format on
    virtual void forwardSync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
    {
    }

protected:
    // Buffer Manager
    std::shared_ptr<runtime::BufferManager> mBufferManager;

    // Domain in which token decoding is computed
    DecoderDomain mDecoderDomain;
};

} // namespace tensorrt_llm::layers
