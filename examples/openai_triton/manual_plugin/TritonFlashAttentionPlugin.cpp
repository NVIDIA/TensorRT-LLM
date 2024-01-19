/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "TritonFlashAttentionPlugin.h"

#include "tensorrt_llm/plugins/common/checkMacrosPlugin.h"
#include "tensorrt_llm/plugins/common/plugin.h"

// Import a generated header to use generated triton kernels.
extern "C"
{
#include "aot/fmha_kernel_fp16.h"
#include "aot/fmha_kernel_fp32.h"
}

#include <string>

using namespace nvinfer1;
using openai_triton::plugin::TritonFlashAttentionPluginCreator;
using openai_triton::plugin::TritonFlashAttentionPlugin;
using tensorrt_llm::plugins::read;
using tensorrt_llm::plugins::write;
using tensorrt_llm::plugins::caughtError;

static const char* TRITON_FLASH_ATTENTION_PLUGIN_VERSION{"1"};
static const char* TRITON_FLASH_ATTENTION_PLUGIN_NAME{"TritonFlashAttention"};
PluginFieldCollection TritonFlashAttentionPluginCreator::mFC{};
std::vector<PluginField> TritonFlashAttentionPluginCreator::mPluginAttributes;

TritonFlashAttentionPlugin::TritonFlashAttentionPlugin(
    int numHeads, int headSize, float softmaxScale, nvinfer1::DataType type)
    : mNumHeads(numHeads)
    , mHeadSize(headSize)
    , mSoftmaxScale(softmaxScale)
    , mType(type)
{
}

// Parameterized constructor
TritonFlashAttentionPlugin::TritonFlashAttentionPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mNumHeads);
    read(d, mHeadSize);
    read(d, mSoftmaxScale);
    read(d, mType);
    TLLM_CHECK(d == a + length);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* TritonFlashAttentionPlugin::clone() const noexcept
{
    auto* plugin = new TritonFlashAttentionPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs TritonFlashAttentionPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Output shape.
    //   output tensor [batchSize, seqLen, mNumHeads, head_size]
    TLLM_CHECK(outputIndex == 0);
    return inputs[outputIndex];
}

bool TritonFlashAttentionPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // In this example, inputs: Q, K, V, outputs: Out
    TLLM_CHECK(nbInputs + nbOutputs == 4);
    TLLM_CHECK(0 <= pos && pos < nbInputs + nbOutputs);

    bool is_valid = false;
    if (0 <= pos && pos < 3) // Q, K, V
    {
        is_valid = inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if (pos == nbInputs) // Out
    {
        is_valid = inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
    }
    return is_valid;
}

void TritonFlashAttentionPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t TritonFlashAttentionPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    // Set workspace size if needed. In this example, we need for L and m buffers.
    const auto Q = inputs[0];
    const int batchSize = Q.dims.d[0];
    const int seqLen = Q.dims.d[2];
    const int numBuffers = 2;
    size_t workspaces[numBuffers];
    workspaces[0] = sizeof(float) * batchSize * mNumHeads * seqLen;
    workspaces[1] = sizeof(float) * batchSize * mNumHeads * seqLen;
    return tensorrt_llm::common::calculateTotalWorkspaceSize(workspaces, numBuffers);
}

template <typename T>
int TritonFlashAttentionPlugin::enqueueImpl(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    TLLM_CHECK(inputDesc[0].dims.d[1] == mNumHeads && inputDesc[0].dims.d[3] == mHeadSize);
    TLLM_CHECK(inputDesc[1].dims.d[1] == mNumHeads && inputDesc[1].dims.d[3] == mHeadSize);
    TLLM_CHECK(inputDesc[2].dims.d[1] == mNumHeads && inputDesc[2].dims.d[3] == mHeadSize);

    int batchSize = inputDesc[0].dims.d[0];
    int seqLen = inputDesc[0].dims.d[2];

    T* Out = reinterpret_cast<T*>(outputs[0]);

    const size_t bufSize = sizeof(float) * batchSize * mNumHeads * seqLen;
    float* L = reinterpret_cast<float*>(workspace);
    float* M = reinterpret_cast<float*>(tensorrt_llm::common::nextWorkspacePtr(reinterpret_cast<int8_t*>(L), bufSize));

    const T* Q = reinterpret_cast<const T*>(inputs[0]);
    const T* K = reinterpret_cast<const T*>(inputs[1]);
    const T* V = reinterpret_cast<const T*>(inputs[2]);

    // Launch a cuda kernel generated by Triton AoT.
    int res = 0;
    if (std::is_same<T, float>::value)
    {
        res = fmha_d64_fp32_default(stream, reinterpret_cast<CUdeviceptr>(Out), reinterpret_cast<CUdeviceptr>(L),
            reinterpret_cast<CUdeviceptr>(M), reinterpret_cast<CUdeviceptr>(Q), reinterpret_cast<CUdeviceptr>(K),
            reinterpret_cast<CUdeviceptr>(V), mSoftmaxScale, batchSize, mNumHeads, seqLen);
    }
    else
    {
        res = fmha_d64_fp16_default(stream, reinterpret_cast<CUdeviceptr>(Out), reinterpret_cast<CUdeviceptr>(L),
            reinterpret_cast<CUdeviceptr>(M), reinterpret_cast<CUdeviceptr>(Q), reinterpret_cast<CUdeviceptr>(K),
            reinterpret_cast<CUdeviceptr>(V), mSoftmaxScale, batchSize, mNumHeads, seqLen);
    }
    return res;
}

int TritonFlashAttentionPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    if (mType == DataType::kHALF)
    {
        return enqueueImpl<half>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else if (mType == DataType::kFLOAT)
    {
        return enqueueImpl<float>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    return 1;
}

// IPluginV2Ext Methods
nvinfer1::DataType TritonFlashAttentionPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

const char* TritonFlashAttentionPlugin::getPluginType() const noexcept
{
    return TRITON_FLASH_ATTENTION_PLUGIN_NAME;
}

const char* TritonFlashAttentionPlugin::getPluginVersion() const noexcept
{
    return TRITON_FLASH_ATTENTION_PLUGIN_VERSION;
}

int TritonFlashAttentionPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int TritonFlashAttentionPlugin::initialize() noexcept
{
    // Load kernels generated by Triton AoT.
    load_fmha_d64_fp32();
    load_fmha_d64_fp16();
    return 0;
}

void TritonFlashAttentionPlugin::terminate() noexcept
{
    // Unload kernels generated by Triton AoT.
    unload_fmha_d64_fp32();
    unload_fmha_d64_fp16();
}

size_t TritonFlashAttentionPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(mSoftmaxScale) + sizeof(mType);
}

void TritonFlashAttentionPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mNumHeads);
    write(d, mHeadSize);
    write(d, mSoftmaxScale);
    write(d, mType);
    assert(d == a + getSerializationSize());
}

void TritonFlashAttentionPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void TritonFlashAttentionPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* TritonFlashAttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

///////////////

TritonFlashAttentionPluginCreator::TritonFlashAttentionPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, -1));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32, -1));
    mPluginAttributes.emplace_back(PluginField("softmax_scale", nullptr, PluginFieldType::kFLOAT32, 1.0f));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* TritonFlashAttentionPluginCreator::getPluginName() const noexcept
{
    return TRITON_FLASH_ATTENTION_PLUGIN_NAME;
}

const char* TritonFlashAttentionPluginCreator::getPluginVersion() const noexcept
{
    return TRITON_FLASH_ATTENTION_PLUGIN_VERSION;
}

const PluginFieldCollection* TritonFlashAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* TritonFlashAttentionPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int numHeads = 0;
    int headSize = 0;
    float softmaxScale = 1.0f;
    nvinfer1::DataType type;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "num_heads"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            numHeads = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "head_size"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            headSize = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "softmax_scale"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            softmaxScale = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new TritonFlashAttentionPlugin(numHeads, headSize, softmaxScale, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* TritonFlashAttentionPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call TritonFlashAttentionPlugin::destroy()
    try
    {
        auto* obj = new TritonFlashAttentionPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void TritonFlashAttentionPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* TritonFlashAttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
