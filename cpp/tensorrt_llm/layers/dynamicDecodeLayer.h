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

#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/penaltyLayer.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm::layers
{

template <typename T>
class DynamicDecodeLayer : public BaseLayer
{
    using Base = BaseLayer;

public:
    DynamicDecodeLayer(executor::DecodingMode const& mode, DecoderDomain const& decodingDomain, cudaStream_t stream,
        std::shared_ptr<tc::IAllocator> allocator);

    ~DynamicDecodeLayer() override;

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, runtime::SizeType32 const* batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams) override;

    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs) override;

    void forwardSync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs) override;

    // Function is only used by test.
    // It is guaranteed by LayersFactory that the first layer is the Penalty layer.
    T* getRuntimeLogitsDevice()
    {
        return dynamic_cast<PenaltyLayer<T>*>(mLayers[0].get())->getRuntimeLogitsDevice();
    }

private:
    void allocateBuffer();
    void freeBuffer();

    void initialize();
    void initializeLayers();

    void prepareIdsPtrs(std::shared_ptr<BaseDecodingOutputs> const& outputs, runtime::SizeType32 const* batchSlots,
        runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, runtime::SizeType32 maxSeqLen);
    static void prepareOutputData(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<DecodingInputs> const& params, runtime::ITensor::SharedPtr const& idsPtrsHost,
        runtime::SizeType32 const* batchSlots, runtime::SizeType32 batchSize, runtime::SizeType32 maxBatchSize,
        runtime::SizeType32 beamWidth, runtime::SizeType32 maxSeqLen, runtime::SizeType32 maxTokensPerStep,
        runtime::SizeType32 cyclicStep, bool outputLogProbs, cudaStream_t stream);

private:
    using Base::mAllocator;
    using Base::mStream;
    using Base::mDecoderDomain;

    std::vector<std::unique_ptr<BaseLayer>> mLayers;

    executor::DecodingMode mDecodingMode;

    runtime::TokenIdType* mZeroParentIdsDevice{nullptr};
    runtime::ITensor::SharedPtr mIdsPtrHost;

    bool mHasDiffRuntimeArgs{false};

    bool mOutputLogProbs{false};

    runtime::SizeType32 mCyclicStep{0};
    runtime::SizeType32 mRuntimeMaxSeqLen{0};
    runtime::SizeType32 mConfiguredBeamWidth{-1};
};

} // namespace tensorrt_llm::layers
