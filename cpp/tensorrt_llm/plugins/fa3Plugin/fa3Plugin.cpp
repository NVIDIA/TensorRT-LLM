/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION &
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
#include "fa3Plugin.h"
#include "flash_api.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/recoverFromRingAtten.h"
#include "tensorrt_llm/kernels/sageAttentionKernels.h"
#include "tensorrt_llm/kernels/selectiveScan/CudaType.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include <cassert>
#include <cstddef>

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;

using tensorrt_llm::plugins::Fa3PluginCreator;
using tensorrt_llm::plugins::Fa3Plugin;

static char const* FA3_PLUGIN_VERSION{"1"};
static char const* FA3_PLUGIN_NAME{"Fa3"};
PluginFieldCollection Fa3PluginCreator::mFC{};
std::vector<nvinfer1::PluginField> Fa3PluginCreator::mPluginAttributes;

Fa3Plugin::Fa3Plugin(int num_heads, int head_size, nvinfer1::DataType type)
    : mNumHeads(num_heads)
    , mHeadSize(head_size)
    , mType(type)
{
}

// Parameterized constructor
Fa3Plugin::Fa3Plugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mNumHeads);
    read(d, mHeadSize);
    read(d, mType);

    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* Fa3Plugin::clone() const noexcept
{
    auto* plugin = new Fa3Plugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
}

nvinfer1::DimsExprs Fa3Plugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    TLLM_CHECK(outputIndex == 0);
    auto ret = inputs[0];
    return ret;
}

bool Fa3Plugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    // inputs: [0] query, [1] key, [2] value
    // outputs: [X] hidden_states
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void Fa3Plugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t Fa3Plugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    // query:   [batch_size, seqlen)q,  num_head, head_dim]
    // key:     [batch_size, seqlen_k, num_head, head_dim]
    // value:   [batch_size, seqlen_k, num_head, head_dim]

    int const batch_size = inputs[0].dims.d[0];
    int const seqlen_q = inputs[0].dims.d[1];
    // const int seqlen_k = inputs[1].dims.d[1];
    int const num_heads = inputs[0].dims.d[2];

    bool fail = false;

    if (inputs[0].dims.d[3] != inputs[1].dims.d[3])
    {
        printf("inner dim of q,k and v should match!");
        fail = true;
    }
    if (inputs[0].dims.d[3] != inputs[2].dims.d[3])
    {
        printf("inner dim of q,k and v should match!");
        fail = true;
    }

    if (fail)
    {
        return -1;
    }
    int const NUM_BUFFERS = 1;
    size_t workspaces[NUM_BUFFERS];
    workspaces[0] = static_cast<size_t>(batch_size * num_heads * seqlen_q) * sizeof(float);

    return tc::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);
}

template <typename T>
int Fa3Plugin::enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, bool is_bf16, cudaStream_t stream)
{

    // inputs
    //     query: [batch_size, seqlen_q, num_head, head_dim]
    //     key: [batch_size,  seqlen_k, num_head, head_dim]
    //     value: [batch_size,  seqlen_k, num_head, head_dim]
    // outputs
    //     output_tensor [batch_size, seqlen_q, num_head, head_dim]

    // if remove padding, inputs[0] dim is [num_tokens] which doesn't have workspace info
    // should get max_batch_size from inputs[1] and max_input_length from plugin attribute
    int const batch_size = inputDesc[0].dims.d[0];
    int const seqlen_q = inputDesc[0].dims.d[1];
    int const seqlen_k = inputDesc[1].dims.d[1];
    int const num_heads = inputDesc[0].dims.d[2];
    int const dim = inputDesc[0].dims.d[3];

    assert(inputs[0].dims.d[3] == inputs[1].dims.d[3] && "inner dim of q,k and v should match!");
    assert(inputs[0].dims.d[3] == inputs[2].dims.d[3] && "inner dim of q,k and v should match!");

    T const* query = reinterpret_cast<T const*>(inputs[0]);
    T const* key = reinterpret_cast<T const*>(inputs[1]);
    T const* value = reinterpret_cast<T const*>(inputs[2]);

    T* context_buf_ = reinterpret_cast<T*>(outputs[0]);

    float* softmax_lse_ptr = reinterpret_cast<float*>(workspace);

    mha_fwd(query, key, value, context_buf_, softmax_lse_ptr, batch_size, seqlen_q, seqlen_k, num_heads, dim, is_bf16,
        stream);

    sync_check_cuda_error(stream);
    return 0;
}

template int Fa3Plugin::enqueueImpl<half>(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    bool is_bf16, cudaStream_t stream);

template int Fa3Plugin::enqueueImpl<float>(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    bool is_bf16, cudaStream_t stream);

#ifdef ENABLE_BF16
template int Fa3Plugin::enqueueImpl<__nv_bfloat16>(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    bool is_bf16, cudaStream_t stream);
#endif

int Fa3Plugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (mType == DataType::kHALF)
    {
        return enqueueImpl<half>(inputDesc, outputDesc, inputs, outputs, workspace, false, stream);
    }
    if (mType == DataType::kBF16)
    {
        return enqueueImpl<__nv_bfloat16>(inputDesc, outputDesc, inputs, outputs, workspace, true, stream);
    }
    printf("\ndtype %d is not supported in this attention!\n", mType);
    return -1;
}

// IPluginV2Ext Methods
nvinfer1::DataType Fa3Plugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

char const* Fa3Plugin::getPluginType() const noexcept
{
    return FA3_PLUGIN_NAME;
}

char const* Fa3Plugin::getPluginVersion() const noexcept
{
    return FA3_PLUGIN_VERSION;
}

int Fa3Plugin::getNbOutputs() const noexcept
{
    return 1;
}

int Fa3Plugin::initialize() noexcept {}

void Fa3Plugin::destroy() noexcept
{
    delete this;
}

size_t Fa3Plugin::getSerializationSize() const noexcept
{
    return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(mType);
}

void Fa3Plugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mNumHeads);
    write(d, mHeadSize);
    write(d, mType);
    TLLM_CHECK(d == a + getSerializationSize());
}

void Fa3Plugin::terminate() noexcept {}

///////////////

Fa3PluginCreator::Fa3PluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();

    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* Fa3PluginCreator::getPluginName() const noexcept
{
    return FA3_PLUGIN_NAME;
}

char const* Fa3PluginCreator::getPluginVersion() const noexcept
{
    return FA3_PLUGIN_VERSION;
}

PluginFieldCollection const* Fa3PluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* Fa3PluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int num_heads{};
    int head_size{};
    ContextFMHAType context_fmha_type{};
    float q_scaling{};
    nvinfer1::DataType type{};
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "num_heads"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            num_heads = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "head_size"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            head_size = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "q_scaling"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            q_scaling = static_cast<float>(*(static_cast<float const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "context_fmha_type"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            context_fmha_type = static_cast<ContextFMHAType>(*(static_cast<int8_t const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new Fa3Plugin(num_heads, head_size, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* Fa3PluginCreator::deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call BertAttentionPlugin::destroy()
    try
    {
        auto* obj = new Fa3Plugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
