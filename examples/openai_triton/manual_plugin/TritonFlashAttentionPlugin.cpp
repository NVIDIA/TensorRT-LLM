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

// Import a generated header to use generated triton kernels.
extern "C"
{
#include "aot/fmha_kernel_fp16.h"
#include "aot/fmha_kernel_fp32.h"
}

#include <cstring>
#include <cuda_fp16.h>
#include <iostream>
#include <string>

using namespace nvinfer1;
using openai_triton::plugin::TritonFlashAttentionPluginCreator;
using openai_triton::plugin::TritonFlashAttentionPlugin;

static char const* TRITON_FLASH_ATTENTION_PLUGIN_VERSION{"1"};
static char const* TRITON_FLASH_ATTENTION_PLUGIN_NAME{"TritonFlashAttention"};
PluginFieldCollection TritonFlashAttentionPluginCreator::mFC{};
std::vector<PluginField> TritonFlashAttentionPluginCreator::mPluginAttributes;

namespace openai_triton::plugin
{

// Write values into buffer
template <typename T>
void writeArg(char*& buffer, T const& val)
{
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
void readArg(char const*& buffer, T& val)
{
    std::memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
}

std::uintptr_t constexpr kCudaMemAlign = 128;

int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize)
{
    uintptr_t addr = (uintptr_t) ptr;
    addr += previousWorkspaceSize;
    if (addr % kCudaMemAlign)
    {
        addr += kCudaMemAlign - addr % kCudaMemAlign;
    }
    return (int8_t*) addr;
}

TritonFlashAttentionPlugin::TritonFlashAttentionPlugin(
    int numHeads, int headSize, float softmaxScale, nvinfer1::DataType type)
    : mNumHeads(numHeads)
    , mHeadSize(headSize)
    , mSoftmaxScale(softmaxScale)
    , mType(type)
{
}

// Parameterized constructor
TritonFlashAttentionPlugin::TritonFlashAttentionPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    readArg(d, mNumHeads);
    readArg(d, mHeadSize);
    readArg(d, mSoftmaxScale);
    readArg(d, mType);
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
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Output shape.
    //   output tensor [batchSize, seqLen, mNumHeads, head_size]
    assert(outputIndex == 0);
    return inputs[outputIndex];
}

bool TritonFlashAttentionPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    // In this example, inputs: Q, K, V, outputs: Out
    assert(nbInputs + nbOutputs == 4);
    assert(0 <= pos && pos < nbInputs + nbOutputs);

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

void TritonFlashAttentionPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t TritonFlashAttentionPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    // Set workspace size if needed. In this example, we need for L and m buffers.
    auto const Q = inputs[0];
    int const batchSize = Q.dims.d[0];
    int const seqLen = Q.dims.d[2];
    int const numBuffers = 2;
    size_t workspaces[numBuffers];
    workspaces[0] = sizeof(float) * batchSize * mNumHeads * seqLen;
    workspaces[1] = sizeof(float) * batchSize * mNumHeads * seqLen;

    size_t total = 0;
    for (int i = 0; i < numBuffers; i++)
    {
        total += workspaces[i];
        if (workspaces[i] % kCudaMemAlign)
        {
            total += kCudaMemAlign - (workspaces[i] % kCudaMemAlign);
        }
    }
    return total;
}

template <typename T>
int TritonFlashAttentionPlugin::enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    assert(inputDesc[0].dims.d[1] == mNumHeads && inputDesc[0].dims.d[3] == mHeadSize);
    assert(inputDesc[1].dims.d[1] == mNumHeads && inputDesc[1].dims.d[3] == mHeadSize);
    assert(inputDesc[2].dims.d[1] == mNumHeads && inputDesc[2].dims.d[3] == mHeadSize);

    int batchSize = inputDesc[0].dims.d[0];
    int seqLen = inputDesc[0].dims.d[2];

    T* Out = reinterpret_cast<T*>(outputs[0]);

    const size_t bufSize = sizeof(float) * batchSize * mNumHeads * seqLen;
    float* L = reinterpret_cast<float*>(workspace);
    float* M = reinterpret_cast<float*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(L), bufSize));

    T const* Q = reinterpret_cast<T const*>(inputs[0]);
    T const* K = reinterpret_cast<T const*>(inputs[1]);
    T const* V = reinterpret_cast<T const*>(inputs[2]);

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

int TritonFlashAttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    int res = 1;
    if (mType == DataType::kHALF)
    {
        res = enqueueImpl<half>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else if (mType == DataType::kFLOAT)
    {
        res = enqueueImpl<float>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    sync_check_cuda_error();
    return res;
}

// IPluginV2Ext Methods
nvinfer1::DataType TritonFlashAttentionPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

char const* TritonFlashAttentionPlugin::getPluginType() const noexcept
{
    return TRITON_FLASH_ATTENTION_PLUGIN_NAME;
}

char const* TritonFlashAttentionPlugin::getPluginVersion() const noexcept
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
    writeArg(d, mNumHeads);
    writeArg(d, mHeadSize);
    writeArg(d, mSoftmaxScale);
    writeArg(d, mType);
    TLLM_CHECK(d == a + getSerializationSize());
}

void TritonFlashAttentionPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void TritonFlashAttentionPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* TritonFlashAttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

///////////////

TritonFlashAttentionPluginCreator::TritonFlashAttentionPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("softmax_scale", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* TritonFlashAttentionPluginCreator::getPluginName() const noexcept
{
    return TRITON_FLASH_ATTENTION_PLUGIN_NAME;
}

char const* TritonFlashAttentionPluginCreator::getPluginVersion() const noexcept
{
    return TRITON_FLASH_ATTENTION_PLUGIN_VERSION;
}

PluginFieldCollection const* TritonFlashAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* TritonFlashAttentionPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int numHeads = 0;
    int headSize = 0;
    float softmaxScale = 1.0f;
    nvinfer1::DataType type;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "num_heads"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            numHeads = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "head_size"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            headSize = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "softmax_scale"))
        {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            softmaxScale = static_cast<float>(*(static_cast<float const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new TritonFlashAttentionPlugin(numHeads, headSize, softmaxScale, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }
    return nullptr;
}

IPluginV2* TritonFlashAttentionPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call TritonFlashAttentionPlugin::destroy()
    try
    {
        auto* obj = new TritonFlashAttentionPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }
    return nullptr;
}

void TritonFlashAttentionPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* TritonFlashAttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

} // namespace openai_triton::plugin
