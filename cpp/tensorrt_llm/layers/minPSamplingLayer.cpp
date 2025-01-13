/*
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

#include "minPSamplingLayer.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/samplingMinPKernels.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/layerUtils.h"

#include <algorithm>
#include <cfloat>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::layers
{

template <typename T>
MinPSamplingLayer<T>::MinPSamplingLayer(DecoderDomain const& decoderDomain,
    std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    allocateBuffer(mDecoderDomain.getBatchSize());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MinPSamplingLayer<T>::allocateBuffer(SizeType32 batchSize)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mWorkspaceSize = getMinPWorkspaceSize<T>(batchSize, mDecoderDomain.getVocabSizePadded());

    auto const batchSizeShape = ITensor::makeShape({batchSize});
    mRuntimeMinPDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);
    mTemperatureDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);

    mSetupWorkspaceSize = std::max({
        mRuntimeMinPDevice->getSizeInBytes(),
        mTemperatureDevice->getSizeInBytes()
    });

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MinPSamplingLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<SamplingSetupParams>(baseSetupParams);

    auto defaultMinP = DefaultDecodingParams::getMinP();
    auto defaultTemperature = DefaultDecodingParams::getTemperature();

    auto runtimeMinP = setupParams->runtimeMinP.value_or(std::vector{defaultMinP});
    auto temperature = setupParams->penaltyParams->temperature.value_or(std::vector{defaultTemperature});

    auto const paramsSize = expandMatchElements(batchSize, runtimeMinP, temperature);
    TLLM_CHECK_WITH_INFO(paramsSize != 0,
        fmtstr("MinPSamplingLayer got parameter with unexpected size, want 1 or batchSize(%d), got"
               "runtimeMinP.size() = %zu, "
               "temperature.size() = %zu",
            batchSize, runtimeMinP.size(), temperature.size()));

    for (size_t i = 0; i < paramsSize; ++i)
    {
        auto& currMinP = runtimeMinP[i];
        auto& currTemperature = temperature[i];

        if (currMinP <= 0.f)
        {
            TLLM_LOG_WARNING(
                "Min (%f) is out of range ((0.0, inf]). Change to default (%f).", currMinP, defaultMinP);

            currMinP = defaultMinP;
        }

        if (currTemperature <= 0.f)
        {
            TLLM_LOG_WARNING(
                "Temperature (%f) is out of range ((0.0, inf]). Change to default (%f).", currTemperature, defaultTemperature);

            currTemperature = defaultTemperature;
        }
    }

    float* MinPsPtr = nullptr;
    float* TemperaturesPtr = nullptr;

    if (paramsSize > 1)
    {
        auto initWorkspaceSizes = getMinPInitWorkspaceSizes<T>(batchSize);
        std::vector<void*> alignedPointers;
        calcAlignedPointers(workspace->getRawWorkspaceDevicePtr(), initWorkspaceSizes)(MinPsPtr, TemperaturesPtr);

        DecodingLayerWorkspace::copyToWorkspace(
            *mBufferManager, runtimeMinP, IBuffer::wrap(MinPsPtr, initWorkspaceSizes[0] / sizeof(*MinPsPtr)));
        
        DecodingLayerWorkspace::copyToWorkspace(
            *mBufferManager, temperature, IBuffer::wrap(TemperaturesPtr, initWorkspaceSizes[1] / sizeof(*TemperaturesPtr)));
    }

    auto const* batchSlotsDevicePtr = workspace->getDeviceBatchSlotsPtr();

    invokeSetMinPRuntimeArgs(batchSize,
        {MinPsPtr, runtimeMinP.front(), bufferCast<float>(*mRuntimeMinPDevice)},
        {TemperaturesPtr, temperature.front(), bufferCast<float>(*mTemperatureDevice)},
        batchSlotsDevicePtr, getStream());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void MinPSamplingLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(MinPSamplingLayer_forwardAsync);

    auto inputs = std::dynamic_pointer_cast<SamplingInputs>(baseInputs);
    auto const batchSize = inputs->logits.value()->getDimension<0>();

    // Probabilities must be already computed instead of logits
    auto probs = bufferCastOrNull<T>(inputs->logits);
    auto const* endIds = bufferCastOrNull<TokenIdType>(inputs->endIds);

    auto const* finishedInput = (inputs->finished)
        ? reinterpret_cast<FinishedState const*>(
            bufferCastOrNull<FinishedState::UnderlyingType>(inputs->finished.value()))
        : nullptr;

    auto* finishedOutput = (outputs->finished)
        ? reinterpret_cast<FinishedState*>(
            bufferCastOrNull<FinishedState::UnderlyingType>(outputs->finished.value()))
        : nullptr;

    MinPSamplingKernelParams<T> params{};
    params.probs = probs;
    params.outputIdsPtrs = bufferCastOrNull<TokenIdType*>(outputs->outputIdsPtr);
    params.workspace = workspace->getRawWorkspaceDevicePtr();
    params.minPs = bufferCastOrNull<float>(mRuntimeMinPDevice);
    params.temperatures = bufferCastOrNull<float>(mTemperatureDevice);
    params.sequenceLength = bufferCastOrNull<SizeType32>(outputs->sequenceLength);
    params.endIds = endIds;
    params.batchSlots = workspace->getDeviceBatchSlotsPtr();
    params.finishedInput = finishedInput;
    params.finishedOutput = finishedOutput;
    params.cumLogProbs = bufferCastOrNull<float>(outputs->cumLogProbs);
    params.outputLogProbs = bufferCastOrNull<float>(outputs->outputLogProbsTiled);
    params.curandState = inputs->curandStates;
    params.batchSize = batchSize;
    params.maxBatchSize = mDecoderDomain.getBatchSize();
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
    invokeBatchMinPSampling<T>(params, getStream());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
size_t MinPSamplingLayer<T>::getWorkspaceSize() const noexcept
{
    return std::max(mSetupWorkspaceSize, mWorkspaceSize);
}

template class MinPSamplingLayer<float>;
template class MinPSamplingLayer<half>;

} // namespace tensorrt_llm::layers
