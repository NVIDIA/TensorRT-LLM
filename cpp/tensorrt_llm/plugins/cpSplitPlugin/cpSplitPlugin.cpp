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

#include <cstdio>

#include "cpSplitPlugin.h"

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::CpSplitPluginCreator;
using tensorrt_llm::plugins::CpSplitPlugin;

static char const* CPSPLIT_PLUGIN_VERSION{"1"};
static char const* CPSPLIT_PLUGIN_NAME{"CpSplit"};
PluginFieldCollection CpSplitPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> CpSplitPluginCreator::mPluginAttributes;

CpSplitPlugin::CpSplitPlugin()
{
    initFieldsToSerialize();
}

CpSplitPlugin::CpSplitPlugin(int cpSize, int cpRank)
    : mCpSize(cpSize)
    , mCpRank(cpRank)
{
    initFieldsToSerialize();
}

void CpSplitPlugin::initFieldsToSerialize()
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(PluginField("cp_size", &mCpSize, PluginFieldType::kINT32, 1));
    mDataToSerialize.emplace_back(PluginField("cp_rank", &mCpRank, PluginFieldType::kINT32, 1));
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
}

// IPluginV3 methods
nvinfer1::IPluginCapability* CpSplitPlugin::getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept
{
    switch (type)
    {
    case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
    case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
    case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
    }
    return nullptr;
}

nvinfer1::IPluginV3* CpSplitPlugin::clone() noexcept
{
    std::unique_ptr<CpSplitPlugin> plugin{std::make_unique<CpSplitPlugin>(*this)};
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initFieldsToSerialize();
    return plugin.release();
}

// IPluginV3OneCore methods
char const* CpSplitPlugin::getPluginName() const noexcept
{
    return CPSPLIT_PLUGIN_NAME;
}

char const* CpSplitPlugin::getPluginVersion() const noexcept
{
    return CPSPLIT_PLUGIN_VERSION;
}

// IPluginV3OneBuild methods
int32_t CpSplitPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return 0;
}

int32_t CpSplitPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    outputTypes[0] = inputTypes[0];
    outputTypes[1] = DataType::kINT32;
    outputTypes[2] = DataType::kINT32;
    return 0;
}

int32_t CpSplitPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    outputs[0].nbDims = 1;

    auto cpSize = exprBuilder.constant(mCpSize);
    auto upper = inputs[0].d[0];
    auto opt = exprBuilder.operation(DimensionOperation::kCEIL_DIV, *upper, *cpSize);
    outputs[0].d[0] = exprBuilder.declareSizeTensor(1, *opt, *upper);

    // We must have such an output size tensor (with dim == 0) to notify the shape of output tensor above
    outputs[1].nbDims = 0;
    outputs[2].nbDims = 1;
    outputs[2].d[0] = upper;
    return 0;
}

bool CpSplitPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (pos == IdxEntry::INPUT_IDS)
    {
        return ((inOut[pos].desc.type == DataType::kINT32) && (inOut[pos].desc.format == TensorFormat::kLINEAR));
    }
    else if (pos == IdxEntry::REQUEST_TYPES || pos == IdxEntry::HOST_CONTEXT_LENGTH)
    {
        return inOut[pos].desc.type == DataType::kINT32;
    }
    else
    {
        return ((inOut[pos].desc.type == DataType::kINT32) && (inOut[pos].desc.format == TensorFormat::kLINEAR));
    }
    return false;
}

int32_t CpSplitPlugin::getNbOutputs() const noexcept
{
    return 3;
}

size_t CpSplitPlugin::getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t CpSplitPlugin::getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept
{
    return 0;
}

int32_t CpSplitPlugin::getNbTactics() noexcept
{
    return 0;
}

char const* CpSplitPlugin::getTimingCacheID() noexcept
{
    return nullptr;
}

int32_t CpSplitPlugin::getFormatCombinationLimit() noexcept
{
    return 1;
}

char const* CpSplitPlugin::getMetadataString() noexcept
{
    return nullptr;
}

// IPluginV3OneRuntime methods
int32_t CpSplitPlugin::setTactic(int32_t tactic) noexcept
{
    return 0;
}

int32_t CpSplitPlugin::onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return 0;
}

int32_t CpSplitPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // inputs
    //     @param inputIds  [tokenNum]
    //     @param host_request_types [batchSize]: Tensor = None (On CPU)
    //          The tensor on the host that indicates if a request is in context or
    //          generation phase. Its shape is [batch_size]. See Inflight Batching
    //          in docs/gpt_attention.md,
    //     @param host_context_lengths [batchSize]: Tensor = None (On CPU)
    //          A host tensor that contains the lengths of the different inputs
    // outputs
    //     @param outputIds [tokenNum spiltted by cp]
    //     @param outputLength scalar
    //     @param joinIdx [tokenNum]

    int64_t tokenNum = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        tokenNum *= inputDesc[0].dims.d[i];
    }

    RequestType const* reqTypes = static_cast<RequestType const*>(inputs[IdxEntry::REQUEST_TYPES]);
    int32_t const* hContextLengths = static_cast<int32_t const*>(inputs[IdxEntry::HOST_CONTEXT_LENGTH]);
    int const* inputIds = reinterpret_cast<int const*>(inputs[IdxEntry::INPUT_IDS]);
    int* outputIds = reinterpret_cast<int*>(outputs[0]);
    int32_t* outputLength = reinterpret_cast<int32_t*>(outputs[1]);
    int32_t* outputJoinIdx = reinterpret_cast<int32_t*>(outputs[2]);

    int32_t const nbSeq = inputDesc[IdxEntry::HOST_CONTEXT_LENGTH].dims.d[0];

    int32_t* hInputs = new int[inputDesc[IdxEntry::INPUT_IDS].dims.d[0]];
    int32_t* hOutputs = new int[inputDesc[IdxEntry::INPUT_IDS].dims.d[0]];
    int32_t* hOutputJoinIdx = new int[inputDesc[IdxEntry::INPUT_IDS].dims.d[0]];
    cudaMemcpyAsync(
        hInputs, inputIds, sizeof(int32_t) * inputDesc[IdxEntry::INPUT_IDS].dims.d[0], cudaMemcpyDeviceToHost, stream);
    sync_check_cuda_error(stream);

    int32_t inputIdx = 0;
    int32_t outputIdx = 0;
    for (int32_t seqIdx = 0; seqIdx < nbSeq; ++seqIdx)
    {
        if (reqTypes[seqIdx] == RequestType::kCONTEXT)
        {
            auto const& ctxLength = hContextLengths[seqIdx];
            int32_t partialAverageLength = (ctxLength + mCpSize - 1) / mCpSize;
            int32_t partialLength
                = mCpRank == mCpSize - 1 ? ctxLength - partialAverageLength * (mCpSize - 1) : partialAverageLength;
            for (int i = 0; i < partialLength; i++)
            {
                hOutputs[outputIdx + i] = hInputs[inputIdx + partialAverageLength * mCpRank + i];
            }
            inputIdx += ctxLength;
            outputIdx += partialAverageLength;
        }
        else if (reqTypes[seqIdx] == RequestType::kGENERATION)
        {
            auto const& genLength = nbSeq - seqIdx;
            int32_t partialAverageLength = (genLength + mCpSize - 1) / mCpSize;
            int32_t partialLength
                = mCpRank == mCpSize - 1 ? genLength - partialAverageLength * (mCpSize - 1) : partialAverageLength;
            for (int i = 0; i < partialLength; i++)
            {
                hOutputs[outputIdx + i] = hInputs[inputIdx + partialAverageLength * mCpRank + i];
            }
            outputIdx += partialAverageLength;
            break;
        }
    }
    int32_t hOutputLength = outputIdx;
    inputIdx = 0;
    outputIdx = 0;
    for (int32_t seqIdx = 0; seqIdx < nbSeq; ++seqIdx)
    {
        if (reqTypes[seqIdx] == RequestType::kCONTEXT)
        {
            auto const& ctxLength = hContextLengths[seqIdx];
            int32_t partialAverageLength = (ctxLength + mCpSize - 1) / mCpSize;
            for (int32_t idx = 0; idx < ctxLength; ++idx)
            {
                hOutputJoinIdx[inputIdx + idx]
                    = idx % partialAverageLength + idx / partialAverageLength * hOutputLength + outputIdx;
            }
            inputIdx += ctxLength;
            outputIdx += partialAverageLength;
        }
        else if (reqTypes[seqIdx] == RequestType::kGENERATION)
        {
            auto const& genLength = nbSeq - seqIdx;
            int32_t partialAverageLength = (genLength + mCpSize - 1) / mCpSize;
            for (int32_t idx = 0; idx < genLength; ++idx)
            {
                hOutputJoinIdx[inputIdx + idx]
                    = idx % partialAverageLength + idx / partialAverageLength * hOutputLength + outputIdx;
            }
            break;
        }
    }
    cudaMemcpyAsync(outputIds, hOutputs, sizeof(int32_t) * hOutputLength, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(outputLength, &hOutputLength, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(outputJoinIdx, hOutputJoinIdx, sizeof(int32_t) * tokenNum, cudaMemcpyHostToDevice, stream);
    sync_check_cuda_error(stream);
    return 0;
}

nvinfer1::IPluginV3* CpSplitPlugin::attachToContext(nvinfer1::IPluginResourceContext* context) noexcept
{
    return clone();
}

nvinfer1::PluginFieldCollection const* CpSplitPlugin::getFieldsToSerialize() noexcept
{
    return &mFCToSerialize;
}

CpSplitPluginCreator::CpSplitPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("cp_size", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("cp_rank", nullptr, PluginFieldType::kINT32));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* CpSplitPluginCreator::getPluginName() const noexcept
{
    return CPSPLIT_PLUGIN_NAME;
}

char const* CpSplitPluginCreator::getPluginVersion() const noexcept
{
    return CPSPLIT_PLUGIN_VERSION;
}

nvinfer1::PluginFieldCollection const* CpSplitPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV3* CpSplitPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept
{
    PluginField const* fields = fc->fields;
    int cp_size{};
    int cp_rank{};
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "cp_size"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            cp_size = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "cp_rank"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            cp_rank = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new CpSplitPlugin(cp_size, cp_rank);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
