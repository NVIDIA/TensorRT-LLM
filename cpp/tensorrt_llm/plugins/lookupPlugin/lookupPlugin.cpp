/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
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

#include "lookupPlugin.h"
#include "tensorrt_llm/kernels/lookupKernels.h"
#include "tensorrt_llm/plugins/common/plugin.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::LookupPluginCreator;
using tensorrt_llm::plugins::LookupPlugin;

static const char* LOOKUP_PLUGIN_VERSION{"1"};
static const char* LOOKUP_PLUGIN_NAME{"Lookup"};
PluginFieldCollection LookupPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> LookupPluginCreator::mPluginAttributes;

LookupPlugin::LookupPlugin(nvinfer1::DataType type, int rank)
    : mType(type)
    , mRank(rank)
{
}

// Parameterized constructor
LookupPlugin::LookupPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mType);
    read(d, mRank);
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* LookupPlugin::clone() const noexcept
{
    auto* plugin = new LookupPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
}

nvinfer1::DimsExprs LookupPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        TLLM_CHECK(nbInputs == 2);
        TLLM_CHECK(outputIndex == 0);
        DimsExprs ret;
        const int nbDimsInput = inputs[0].nbDims;
        const int nbDimsWeight = inputs[1].nbDims;
        ret.nbDims = nbDimsInput + 1;

        for (int i = 0; i < nbDimsInput; ++i)
        {
            ret.d[i] = inputs[0].d[i];
        }
        ret.d[nbDimsInput] = inputs[1].d[nbDimsWeight - 1];

        return ret;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool LookupPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    bool res = false;
    switch (pos)
    {
    case 0: res = ((inOut[0].type == DataType::kINT32) && (inOut[0].format == TensorFormat::kLINEAR)); break;
    case 1: res = ((inOut[1].type == mType) && (inOut[1].format == TensorFormat::kLINEAR)); break;
    case 2: res = ((inOut[2].type == mType) && (inOut[2].format == TensorFormat::kLINEAR)); break;
    default: // should NOT be here!
        res = false;
    }

    return res;
}

void LookupPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t LookupPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int LookupPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // inputs
    //     input  [batchSize]
    //     weight [localVocabSize, hidden]
    // outputs
    //     embedding [batchSize, hidden]

    int batchSize = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        batchSize *= inputDesc[0].dims.d[i];
    }

    const int localVocabSize = inputDesc[1].dims.d[0];
    const int hidden = inputDesc[1].dims.d[inputDesc[1].dims.nbDims - 1];
    const int* input = reinterpret_cast<const int*>(inputs[0]);

    int offset = mRank * localVocabSize;

    if (mType == DataType::kHALF)
    {
        const half* weight = reinterpret_cast<const half*>(inputs[1]);
        half* output = reinterpret_cast<half*>(outputs[0]);
        invokeLookUp<half, int>(output, input, weight, batchSize, offset, localVocabSize, hidden, stream);
    }
    else if (mType == DataType::kFLOAT)
    {
        const float* weight = reinterpret_cast<const float*>(inputs[1]);
        float* output = reinterpret_cast<float*>(outputs[0]);
        invokeLookUp<float, int>(output, input, weight, batchSize, offset, localVocabSize, hidden, stream);
    }
    else if (mType == DataType::kBF16)
    {
        const __nv_bfloat16* weight = reinterpret_cast<const __nv_bfloat16*>(inputs[1]);
        __nv_bfloat16* output = reinterpret_cast<__nv_bfloat16*>(outputs[0]);
        invokeLookUp<__nv_bfloat16, int>(output, input, weight, batchSize, offset, localVocabSize, hidden, stream);
    }

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType LookupPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == 0);
    return inputTypes[1];
}

// IPluginV2 Methods

const char* LookupPlugin::getPluginType() const noexcept
{
    return LOOKUP_PLUGIN_NAME;
}

const char* LookupPlugin::getPluginVersion() const noexcept
{
    return LOOKUP_PLUGIN_VERSION;
}

int LookupPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int LookupPlugin::initialize() noexcept
{
    return 0;
}

void LookupPlugin::destroy() noexcept
{
    delete this;
}

size_t LookupPlugin::getSerializationSize() const noexcept
{
    return sizeof(mType) + sizeof(mRank);
}

void LookupPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    write(d, mRank);

    assert(d == a + getSerializationSize());
}

void LookupPlugin::terminate() noexcept {}

///////////////

LookupPluginCreator::LookupPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("rank", nullptr, PluginFieldType::kINT32, 0));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* LookupPluginCreator::getPluginName() const noexcept
{
    return LOOKUP_PLUGIN_NAME;
}

const char* LookupPluginCreator::getPluginVersion() const noexcept
{
    return LOOKUP_PLUGIN_VERSION;
}

const PluginFieldCollection* LookupPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* LookupPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    nvinfer1::DataType type;
    int rank;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "rank"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            rank = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new LookupPlugin(type, rank);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* LookupPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call LookupPlugin::destroy()
    try
    {
        auto* obj = new LookupPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
