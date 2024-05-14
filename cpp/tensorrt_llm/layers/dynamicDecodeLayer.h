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

#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/layers/banWordsLayer.h"
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/beamSearchLayer.h"
#include "tensorrt_llm/layers/decodingLayer.h"
#include "tensorrt_llm/layers/medusaDecodingLayer.h"
#include "tensorrt_llm/layers/penaltyLayer.h"
#include "tensorrt_llm/layers/samplingLayer.h"
#include "tensorrt_llm/layers/stopCriteriaLayer.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/decodingMode.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{
struct BeamHypotheses;
}

namespace layers
{

template <typename T>
class DynamicDecodeLayer : public BaseLayer
{
    using Base = BaseLayer;

public:
    DynamicDecodeLayer(runtime::DecodingMode const& mode, DecoderDomain const& decodingDomain, cudaStream_t stream,
        std::shared_ptr<tc::IAllocator> allocator);

    ~DynamicDecodeLayer() override;

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, runtime::SizeType32 const* batchSlots,
        std::shared_ptr<BaseSetupParams> setupParams) override;

    void forward(std::shared_ptr<BaseOutputParams> outputs, std::shared_ptr<BaseInputParams> inputs) override;

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

    void prepareIdsPtrs(std::shared_ptr<DynamicDecodeOutputParams> const& outputs,
        runtime::SizeType32 const* batchSlots, runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth,
        runtime::SizeType32 maxSeqLen);
    static void prepareOutputData(std::shared_ptr<DynamicDecodeOutputParams> const& outputs,
        std::shared_ptr<DynamicDecodeInputParams> const& params, runtime::ITensor::SharedPtr const& idsPtrsHost,
        runtime::SizeType32 const* batchSlots, runtime::SizeType32 batchSize, runtime::SizeType32 maxBatchSize,
        runtime::SizeType32 beamWidth, runtime::SizeType32 maxSeqLen, runtime::SizeType32 maxTokensPerStep,
        runtime::SizeType32 cyclicStep, cudaStream_t stream);

private:
    using Base::mAllocator;
    using Base::mStream;
    using Base::mDecoderDomain;

    std::vector<std::unique_ptr<BaseLayer>> mLayers;

    runtime::DecodingMode mDecodingMode;

    runtime::TokenIdType* mZeroParentIdsDevice{nullptr};
    runtime::ITensor::SharedPtr mIdsPtrHost;

    bool mHasDiffRuntimeArgs{false};

    runtime::SizeType32 mCyclicStep{0};
    runtime::SizeType32 mRuntimeMaxSeqLen{0};
    runtime::SizeType32 mConfiguredBeamWidth{-1};
};

} // namespace layers
} // namespace tensorrt_llm
