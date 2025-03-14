/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "cumsumLastDimPlugin.h"
#include "tensorrt_llm/common/assert.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::CumsumLastDimPluginCreator;
using tensorrt_llm::plugins::CumsumLastDimPlugin;

static char const* CUMSUM_LAST_DIM_PLUGIN_VERSION{"1"};
static char const* CUMSUM_LAST_DIM_PLUGIN_NAME{"CumsumLastDim"};
PluginFieldCollection CumsumLastDimPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> CumsumLastDimPluginCreator::mPluginAttributes;

static constexpr SizeType32 LENGTH_LIMIT_FOR_BLOCKSCAN = 4096;

CumsumLastDimPlugin::CumsumLastDimPlugin(SizeType32 inputLength, nvinfer1::DataType type, size_t temp_storage_bytes)
    : mInputLength(inputLength)
    , mTempStorageBytes(temp_storage_bytes)
    , mType(type)
{
    TLLM_CHECK_WITH_INFO((mType == DataType::kBF16) || (mType == DataType::kFLOAT) || (mType == DataType::kHALF)
            || (mType == DataType::kINT32),
        "Only support int, float, half, and bfloat16.");
    if (mTempStorageBytes == 0)
    {
        mTempStorageBytes = getWorkspaceSizeNeeded(inputLength, type);
    }
}

// Parameterized constructor
CumsumLastDimPlugin::CumsumLastDimPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mInputLength);
    read(d, mTempStorageBytes);
    read(d, mType);
    TLLM_CHECK(d == a + length);
    TLLM_CHECK_WITH_INFO((mType == DataType::kBF16) || (mType == DataType::kFLOAT) || (mType == DataType::kHALF)
            || (mType == DataType::kINT32),
        "Only support int, float, half, and bfloat16.");
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* CumsumLastDimPlugin::clone() const noexcept
{
    auto* plugin = new CumsumLastDimPlugin(mInputLength, mType, mTempStorageBytes);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

// Outputs
// output_tensor: [batch_size, inputLength]
nvinfer1::DimsExprs CumsumLastDimPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    TLLM_CHECK_WITH_INFO(outputIndex == 0, "Only one output.");
    return inputs[getInputTensorIdx()];
}

bool CumsumLastDimPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void CumsumLastDimPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t CumsumLastDimPlugin::getWorkspaceSizeNeeded(SizeType32 inputLength, nvinfer1::DataType type)
{
    size_t tempStorageBytes{0};
    if (inputLength < LENGTH_LIMIT_FOR_BLOCKSCAN) // last dim unknown or small, use BlockScan
    {
        tempStorageBytes = 0;
    }
    else if (type == DataType::kINT32)
    {
        tempStorageBytes = invokeComputeCumsumLastDimWorkspaceSize<int>(inputLength);
    }
    else if (type == DataType::kHALF)
    {
        tempStorageBytes = invokeComputeCumsumLastDimWorkspaceSize<half>(inputLength);
    }
    else if (type == DataType::kFLOAT)
    {
        tempStorageBytes = invokeComputeCumsumLastDimWorkspaceSize<float>(inputLength);
    }
#ifdef ENABLE_BF16
    else if (type == DataType::kBF16)
    {
        tempStorageBytes = invokeComputeCumsumLastDimWorkspaceSize<__nv_bfloat16>(inputLength);
    }
#endif
    return tempStorageBytes;
}

size_t CumsumLastDimPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return mTempStorageBytes;
}

template <typename T>
int CumsumLastDimPlugin::enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    // inputs
    //     0.  input_tensor [batch_size, inputLength]
    // outputs
    //     0.  output_tensor [batch_size, inputLength]
    auto const batchSize = inputDesc[getInputTensorIdx()].dims.d[0];
    auto const inputLength = inputDesc[getInputTensorIdx()].dims.d[1];
    /*
        Two cases where we should use BlockScan:
            1. inputLength is small
            2. batchSize is large (since DeviceScan causes kernel launch per row)
    */
    void* wp = inputLength < LENGTH_LIMIT_FOR_BLOCKSCAN || batchSize > 2 ? nullptr : workspace;
    invokeCumsumLastDim<T>(
        batchSize, inputLength, inputs[getInputTensorIdx()], outputs[0], wp, mTempStorageBytes, stream);

    sync_check_cuda_error(stream);
    return 0;
}

int CumsumLastDimPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    if (mType == DataType::kINT32)
    {
        return enqueueImpl<int>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else if (mType == DataType::kHALF)
    {
        return enqueueImpl<half>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else if (mType == DataType::kFLOAT)
    {
        return enqueueImpl<float>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16)
    {
        return enqueueImpl<__nv_bfloat16>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
#endif
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType CumsumLastDimPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK_WITH_INFO(index == 0, "Only one output.");
    return inputTypes[getInputTensorIdx()];
}

// IPluginV2 Methods

char const* CumsumLastDimPlugin::getPluginType() const noexcept
{
    return CUMSUM_LAST_DIM_PLUGIN_NAME;
}

char const* CumsumLastDimPlugin::getPluginVersion() const noexcept
{
    return CUMSUM_LAST_DIM_PLUGIN_VERSION;
}

int CumsumLastDimPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int CumsumLastDimPlugin::initialize() noexcept
{
    return 0;
}

void CumsumLastDimPlugin::terminate() noexcept {}

size_t CumsumLastDimPlugin::getSerializationSize() const noexcept
{
    return sizeof(mInputLength) + sizeof(mTempStorageBytes) + sizeof(mType);
}

void CumsumLastDimPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mInputLength);
    write(d, mTempStorageBytes);
    write(d, mType);
    TLLM_CHECK(d == a + getSerializationSize());
}

void CumsumLastDimPlugin::destroy() noexcept
{
    delete this;
}

///////////////

CumsumLastDimPluginCreator::CumsumLastDimPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("input_length", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* CumsumLastDimPluginCreator::getPluginName() const noexcept
{
    return CUMSUM_LAST_DIM_PLUGIN_NAME;
}

char const* CumsumLastDimPluginCreator::getPluginVersion() const noexcept
{
    return CUMSUM_LAST_DIM_PLUGIN_VERSION;
}

PluginFieldCollection const* CumsumLastDimPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* CumsumLastDimPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int inputLength{};
    nvinfer1::DataType type{};
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "input_length"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            inputLength = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new CumsumLastDimPlugin(inputLength, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* CumsumLastDimPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call CumsumLastDimPlugin::destroy()
    try
    {
        auto* obj = new CumsumLastDimPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
