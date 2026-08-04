/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "doraPlugin.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/runtime/iBuffer.h"

#include <numeric>

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::DoraPlugin;
using tensorrt_llm::plugins::DoraPluginCreator;

static char const* DORA_PLUGIN_VERSION{"1"};
static char const* DORA_PLUGIN_NAME{"Dora"};
PluginFieldCollection DoraPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> DoraPluginCreator::mPluginAttributes;

DoraPlugin::DoraPlugin(std::vector<int32_t> const& outHiddenSizes, nvinfer1::DataType type, bool removeInputPadding)
    : mType(type)
    , mRemoveInputPadding(removeInputPadding)
    , mDoraImpl(outHiddenSizes, type)
{
    mOutHiddenSizes.resize(outHiddenSizes.size());
    mOutHiddenSizes.assign(outHiddenSizes.cbegin(), outHiddenSizes.cend());
    init();
}

void DoraPlugin::init()
{
    // initialize data to serialize
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(
        "out_hidden_sizes", mOutHiddenSizes.data(), PluginFieldType::kINT32, mOutHiddenSizes.size());
    mDataToSerialize.emplace_back("type", &mType, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("remove_input_padding", &mRemoveInputPadding, PluginFieldType::kINT8, 1);
    mFieldsToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFieldsToSerialize.fields = mDataToSerialize.data();
}

// IPluginV3 methods
nvinfer1::IPluginCapability* DoraPlugin::getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept
{
    switch (type)
    {
    case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
    case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
    case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
    }
    return nullptr;
}

nvinfer1::IPluginV3* DoraPlugin::clone() noexcept
{
    std::unique_ptr<DoraPlugin> plugin{std::make_unique<DoraPlugin>(mOutHiddenSizes, mType, mRemoveInputPadding)};
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin.release();
}

// IPluginV3OneCore methods
char const* DoraPlugin::getPluginName() const noexcept
{
    return DORA_PLUGIN_NAME;
}

char const* DoraPlugin::getPluginVersion() const noexcept
{
    return DORA_PLUGIN_VERSION;
}

// IPluginV3OneBuild methods
int32_t DoraPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return 0;
}

int32_t DoraPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        TLLM_CHECK(nbOutputs == 1);
        TLLM_CHECK(nbInputs == 2 + static_cast<int32_t>(mOutHiddenSizes.size()) + (mRemoveInputPadding ? 1 : 0));
        TLLM_CHECK(inputTypes[IdxEntry::kINPUT_TENSOR] == mType);
        // output has the same dtype as the input, the plugin just applies scaling
        outputTypes[0] = inputTypes[IdxEntry::kINPUT_TENSOR];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return 0;
}

int32_t DoraPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        TLLM_CHECK(nbOutputs == 1);
        TLLM_CHECK(nbShapeInputs == 0);
        TLLM_CHECK(nbInputs == 2 + static_cast<int32_t>(mOutHiddenSizes.size()) + (mRemoveInputPadding ? 1 : 0));

        auto const inputTensorDims = inputs[IdxEntry::kINPUT_TENSOR];
        TLLM_CHECK(inputTensorDims.nbDims == (mRemoveInputPadding ? 2 : 3));

        auto const lastDim = inputTensorDims.d[inputTensorDims.nbDims - 1];
        TLLM_CHECK(lastDim->isConstant());
        TLLM_CHECK(lastDim->getConstantValue() == std::accumulate(mOutHiddenSizes.cbegin(), mOutHiddenSizes.cend(), 0));

        outputs[0].nbDims = inputTensorDims.nbDims;
        for (auto dim = 0; dim < inputTensorDims.nbDims; ++dim)
        {
            outputs[0].d[dim] = inputTensorDims.d[dim];
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return 0;
}

bool DoraPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    auto const numModules = static_cast<int32_t>(mOutHiddenSizes.size());
    if (nbInputs != 2 + numModules + (mRemoveInputPadding ? 1 : 0))
    {
        return false;
    }

    bool const isInput = pos < nbInputs;
    if (pos == IdxEntry::kHOST_REQUEST_TYPES)
    {
        return (inOut[pos].desc.type == nvinfer1::DataType::kINT32);
    }
    // optional host_context_lens after lora pointers
    else if (pos == IdxEntry::kLORA_WEIGHTS_PTRS_START + numModules and isInput)
    {
        return (inOut[pos].desc.type == nvinfer1::DataType::kINT32 and mRemoveInputPadding);
    }
    // lora weight pointers
    else if (pos >= IdxEntry::kLORA_WEIGHTS_PTRS_START and pos < IdxEntry::kLORA_WEIGHTS_PTRS_START + numModules)
    {
        return (inOut[pos].desc.type == nvinfer1::DataType::kINT64);
    }
    else if (pos != 0 and isInput)
    {
        TLLM_LOG_WARNING("%s: got an unexpected input at position %d", __PRETTY_FUNCTION__, pos);
        return false;
    }

    return (inOut[pos].desc.type == mType) and (inOut[pos].desc.format == TensorFormat::kLINEAR);
}

int32_t DoraPlugin::getNbOutputs() const noexcept
{
    return 1;
}

size_t DoraPlugin::getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    auto const inputTensorMax = inputs[IdxEntry::kINPUT_TENSOR].max;
    auto const maxNumTokens = mRemoveInputPadding ? inputTensorMax.d[0] : inputTensorMax.d[0] * inputTensorMax.d[1];
    auto const size = mDoraImpl.getWorkspaceSize(maxNumTokens);
    return size;
}

int32_t DoraPlugin::getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept
{
    return 0;
}

int32_t DoraPlugin::getNbTactics() noexcept
{
    return 0;
}

char const* DoraPlugin::getTimingCacheID() noexcept
{
    return nullptr;
}

int32_t DoraPlugin::getFormatCombinationLimit() noexcept
{
    return 1;
}

char const* DoraPlugin::getMetadataString() noexcept
{
    return nullptr;
}

// IPluginV3OneRuntime methods
int32_t DoraPlugin::setTactic(int32_t tactic) noexcept
{
    return 0;
}

int32_t DoraPlugin::onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return 0;
}

int32_t DoraPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }

    auto const numModules = static_cast<int32_t>(mOutHiddenSizes.size());
    auto const numReqs = inputDesc[IdxEntry::kHOST_REQUEST_TYPES].dims.d[0];

    auto const inputTensorDesc = inputDesc[IdxEntry::kINPUT_TENSOR];
    auto const numTokens
        = mRemoveInputPadding ? inputTensorDesc.dims.d[0] : inputTensorDesc.dims.d[0] * inputTensorDesc.dims.d[1];
    auto const seqLen = mRemoveInputPadding ? 0 : inputTensorDesc.dims.d[1];

    void const* inputTensor = inputs[IdxEntry::kINPUT_TENSOR];
    auto const* hostRequestTypes = static_cast<int32_t const*>(inputs[IdxEntry::kHOST_REQUEST_TYPES]);
    void const* const* loraWeightsPtrs = &inputs[IdxEntry::kLORA_WEIGHTS_PTRS_START];

    int32_t const* hostContextLengths = mRemoveInputPadding
        ? static_cast<int32_t const*>(inputs[IdxEntry::kLORA_WEIGHTS_PTRS_START + numModules])
        : nullptr;

    mExpandDoraWeightPtrs.clear();
    mExpandDoraWeightPtrs.reserve(numModules * numTokens);

    bool hasAnyDora = false;

    for (auto moduleIdx = 0; moduleIdx < numModules; moduleIdx++)
    {
        auto const loraWeightModulePtrs = static_cast<int64_t const*>(loraWeightsPtrs[moduleIdx]);

        int idx = 0;
        for (int reqId = 0; reqId < numReqs; reqId++)
        {
            // loraWeightModulePtrs has 3 pointers for each module: A,B, and an optional DoRA magnitude
            // the current DoRA plugin does not apply LoRA, so A and B are ignored.
            RequestType const reqType = static_cast<RequestType const>(hostRequestTypes[reqId]);
            auto const* modulePtr = reinterpret_cast<void const*>(loraWeightModulePtrs[reqId * 3 + 2]);
            hasAnyDora = hasAnyDora or modulePtr != nullptr;

            if (reqType == RequestType::kGENERATION)
            {
                mExpandDoraWeightPtrs.push_back(modulePtr);
                idx += 1;
            }
            else
            {
                int contextLen = (mRemoveInputPadding ? hostContextLengths[reqId] : seqLen);

                for (int contextId = 0; contextId < contextLen; contextId++)
                {
                    mExpandDoraWeightPtrs.push_back(modulePtr);
                    idx += 1;
                }
            }
        }
        if (idx != numTokens)
        {
            TLLM_LOG_ERROR("LoraParams and input dims don't match, lora tokens %d input tokens %d", idx, numTokens);
            return -1;
        }
    }

    if (hasAnyDora)
    {
        mDoraImpl.run(numTokens, inputTensor, mExpandDoraWeightPtrs.data(), outputs, workspace, stream);
    }
    else
    {
        // skip dora scaling if all requests are pure-lora
        auto const inputRank = inputTensorDesc.dims.nbDims;
        auto const numel
            = std::accumulate(inputTensorDesc.dims.d, inputTensorDesc.dims.d + inputRank, 1, std::multiplies());
        auto const elemSize = tensorrt_llm::common::getDTypeSize(mType);
        tensorrt_llm::common::cudaAutoCpy((int8_t*) outputs[0], (int8_t*) inputTensor, numel * elemSize, stream);
    }

    sync_check_cuda_error(stream);
    return 0;
}

nvinfer1::IPluginV3* DoraPlugin::attachToContext(nvinfer1::IPluginResourceContext* context) noexcept
{
    return clone();
}

nvinfer1::PluginFieldCollection const* DoraPlugin::getFieldsToSerialize() noexcept
{
    return &mFieldsToSerialize;
}

DoraPluginCreator::DoraPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back("num_modules", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("type", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("remove_input_padding", nullptr, PluginFieldType::kINT8, 1);
    mFC.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

char const* DoraPluginCreator::getPluginName() const noexcept
{
    return DORA_PLUGIN_NAME;
}

char const* DoraPluginCreator::getPluginVersion() const noexcept
{
    return DORA_PLUGIN_VERSION;
}

nvinfer1::PluginFieldCollection const* DoraPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV3* DoraPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept
{
    PluginField const* fields = fc->fields;
    nvinfer1::DataType type{};
    bool removeInputPadding{};
    std::vector<int32_t> outHiddenSizes;

    // Read configurations from each field
    for (int i = 0; i < fc->nbFields; ++i)
    {
        auto const field = fields[i];
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "type"))
        {
            TLLM_CHECK(field.type == PluginFieldType::kINT32 and field.length == 1);
            type = *static_cast<nvinfer1::DataType const*>(field.data);
        }
        else if (!strcmp(attrName, "remove_input_padding"))
        {
            TLLM_CHECK(field.type == PluginFieldType::kINT8 and field.length == 1);
            removeInputPadding = *static_cast<bool const*>(field.data);
        }
        else if (!strcmp(attrName, "out_hidden_sizes"))
        {
            TLLM_CHECK(field.type == PluginFieldType::kINT32);
            auto const* outHiddenSizesPtr = static_cast<int32_t const*>(field.data);
            outHiddenSizes.resize(field.length);
            outHiddenSizes.assign(outHiddenSizesPtr, outHiddenSizesPtr + field.length);
        }
        else
        {
            TLLM_LOG_WARNING("%s: got an unexpected attribute: %s", __PRETTY_FUNCTION__, attrName);
        }
    }

    try
    {
        auto* obj = new DoraPlugin(outHiddenSizes, type, removeInputPadding);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
