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

#include "lruPlugin.h"
#include "tensorrt_llm/common/assert.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::lruPluginCreator;
using tensorrt_llm::plugins::lruPlugin;

static char const* LRU_PLUGIN_VERSION{"1"};
static char const* LRU_PLUGIN_NAME{"LRU"};
PluginFieldCollection lruPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> lruPluginCreator::mPluginAttributes;

lruPlugin::lruPlugin(int dim, int block_size, nvinfer1::DataType type, bool removePadding, bool pagedState,
    bool yEnabled, bool yBiasEnabled, bool fuseGateEnabled, bool gateBiasEnabled)
    : mDim(dim)
    , mBlockSize(block_size)
    , mType(type)
    , mRemovePadding(removePadding)
    , mPagedState(pagedState)
    , mYEnabled(yEnabled)
    , mYBiasEnabled(yBiasEnabled)
    , mFuseGateEnabled(fuseGateEnabled)
    , mGateBiasEnabled(gateBiasEnabled)
{
    TLLM_CHECK_WITH_INFO((mType == DataType::kBF16) || (mType == DataType::kFLOAT) || (mType == DataType::kHALF),
        "Only support float, half, and bfloat16.");
}

// Parameterized constructor
lruPlugin::lruPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mDim);
    read(d, mBlockSize);
    read(d, mType);
    read(d, mRemovePadding);
    read(d, mPagedState);
    read(d, mYEnabled);
    read(d, mYBiasEnabled);
    read(d, mFuseGateEnabled);
    read(d, mGateBiasEnabled);
    TLLM_CHECK(d == a + length);
    TLLM_CHECK_WITH_INFO((mType == DataType::kBF16) || (mType == DataType::kFLOAT) || (mType == DataType::kHALF),
        "Only support float, half, and bfloat16.");
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* lruPlugin::clone() const noexcept
{
    auto* plugin = new lruPlugin(mDim, mBlockSize, mType, mRemovePadding, mPagedState, mYEnabled, mYBiasEnabled,
        mFuseGateEnabled, mGateBiasEnabled);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

// Outputs
//     output_tensor: [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
//     state: [batch_size, dim]
nvinfer1::DimsExprs lruPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    if (outputIndex == 0)
    {
        return inputs[getXIdx()];
    }
    return inputs[getStateIdx()];
}

bool lruPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (pos == getHostRequestTypesIdx() || pos == getLastTokenIdsIdx() || (mPagedState && pos == getSlotMappingIdx()))
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else if (mPagedState && pos == getStateIdx())
    {
        return inOut[pos].type == nvinfer1::DataType::kINT64;
    }
    else if (pos == getStateIdx() || pos == (nbInputs + 1))
    {
        // Use float for both input and output state
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else
    {
        return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
}

void lruPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t lruPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

void lruPlugin::setLruParams(lruParams& params, const size_t batch, const size_t dim, const size_t block_size,
    const size_t maxSeqLen, void* statePtr, void const* x, void const* gate, void const* gate_bias, void const* gate_x,
    void const* gate_x_bias, void const* gate_a, void const* gate_a_bias, void const* y, void const* y_bias,
    void const* A, int const* lastTokenIds, int const* slotMapping, void* out, bool removePadding)
{
    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.width = dim;
    params.block_size = block_size;
    params.max_seqlen = maxSeqLen;
    params.remove_padding = removePadding;

    // Set the pointers and strides.
    params.A_ptr = const_cast<void*>(A);
    params.x_ptr = const_cast<void*>(x);
    params.y_ptr = const_cast<void*>(y);
    params.y_bias_ptr = const_cast<void*>(y_bias);
    params.gate_ptr = const_cast<void*>(gate);
    params.gate_bias_ptr = const_cast<void*>(gate_bias);
    params.gate_x_ptr = const_cast<void*>(gate_x);
    params.gate_x_bias_ptr = const_cast<void*>(gate_x_bias);
    params.gate_a_ptr = const_cast<void*>(gate_a);
    params.gate_a_bias_ptr = const_cast<void*>(gate_a_bias);
    params.state_ptr = statePtr;
    params.out_ptr = out;
    params.last_token_ids_ptr = lastTokenIds;
    params.slot_mapping_ptr = slotMapping;
}

template <typename T>
int lruPlugin::enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    // inputs
    //     0.  x [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
    //     1.  A [dim]
    //     2.  state [batch_size, dim] or host [1] containing only pointer for paged_state
    //     3.  host_request_types [batch_size] int32. 0: context; 1: generation; 2: none.
    //     4.  last_token_ids [batch_size] int32
    //     5.  state_slot_mapping [batch_size] int32, optional for paged state
    //     6.  y [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
    //     7.  y_bias [dim]
    //     8.  gate [batch_size, seq_len, 2 * dim] or [num_tokens, 2 * dim] for remove_input_padding
    //     9.  gate_bias [2 * dim]
    //    10.  gate_x [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
    //    11.  gate_a [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
    //    12.  gate_x_bias [2 * dim]
    //    13.  gate_a_bias [2 * dim]
    // outputs
    //     0. output_tensor [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
    //     1. state [batch_size, dim]
    auto const batch_size = inputDesc[getHostRequestTypesIdx()].dims.d[0];
    int max_seq_len;
    if (mRemovePadding)
    {
        max_seq_len = -1;
    }
    else
    {
        max_seq_len = inputDesc[getXIdx()].dims.d[1];
    }

    // only support context or generation, not for both of them
    RequestType const* reqTypes = static_cast<RequestType const*>(inputs[getHostRequestTypesIdx()]);

    lruParams lru_params;

    int const* slotMapping = mPagedState ? static_cast<int const*>(inputs[getSlotMappingIdx()]) : nullptr;
    void const* y = mYEnabled ? inputs[getYIdx()] : nullptr;
    void const* y_bias = mYBiasEnabled ? inputs[getYBiasIdx()] : nullptr;
    void const* gate = mFuseGateEnabled ? inputs[getGateIdx()] : nullptr;
    void const* gate_bias = (mFuseGateEnabled && mGateBiasEnabled) ? inputs[getGateBiasIdx()] : nullptr;
    void const* gate_x = mFuseGateEnabled ? nullptr : inputs[getGateXIdx()];
    void const* gate_a = mFuseGateEnabled ? nullptr : inputs[getGateAIdx()];
    void const* gate_x_bias = (!mFuseGateEnabled && mGateBiasEnabled) ? inputs[getGateXBiasIdx()] : nullptr;
    void const* gate_a_bias = (!mFuseGateEnabled && mGateBiasEnabled) ? inputs[getGateABiasIdx()] : nullptr;

    void* statePtr = mPagedState ? *reinterpret_cast<void**>(const_cast<void*>(inputs[getStateIdx()])) : outputs[1];

    setLruParams(lru_params, batch_size, mDim, mBlockSize, max_seq_len, statePtr, inputs[getXIdx()], gate, gate_bias,
        gate_x, gate_x_bias, gate_a, gate_a_bias, y, y_bias, inputs[getAIdx()],
        static_cast<int const*>(inputs[getLastTokenIdsIdx()]), slotMapping, outputs[0], mRemovePadding);

    if (reqTypes[0] == RequestType::kCONTEXT)
    {
        invokeRGLRU<T>(lru_params, stream);
    }
    else if (reqTypes[0] == RequestType::kGENERATION)
    {
        invokeRGLRUUpdate<T>(lru_params, stream);
    }
    sync_check_cuda_error(stream);
    return 0;
}

int lruPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }
    if (mType == DataType::kHALF)
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
nvinfer1::DataType lruPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    if (index == 0)
    {
        return inputTypes[getXIdx()];
    }
    else
    {
        return inputTypes[getStateIdx()];
    }
}

// IPluginV2 Methods

char const* lruPlugin::getPluginType() const noexcept
{
    return LRU_PLUGIN_NAME;
}

char const* lruPlugin::getPluginVersion() const noexcept
{
    return LRU_PLUGIN_VERSION;
}

int lruPlugin::getNbOutputs() const noexcept
{
    return mPagedState ? 1 : 2;
}

int lruPlugin::initialize() noexcept
{
    return 0;
}

void lruPlugin::terminate() noexcept {}

size_t lruPlugin::getSerializationSize() const noexcept
{
    return sizeof(mDim) + sizeof(mBlockSize) + sizeof(mType) + sizeof(mRemovePadding) + sizeof(mPagedState)
        + sizeof(mYEnabled) + sizeof(mYBiasEnabled) + sizeof(mFuseGateEnabled) + sizeof(mGateBiasEnabled);
}

void lruPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mDim);
    write(d, mBlockSize);
    write(d, mType);
    write(d, mRemovePadding);
    write(d, mPagedState);
    write(d, mYEnabled);
    write(d, mYBiasEnabled);
    write(d, mFuseGateEnabled);
    write(d, mGateBiasEnabled);
    TLLM_CHECK(d == a + getSerializationSize());
}

void lruPlugin::destroy() noexcept
{
    delete this;
}

///////////////

lruPluginCreator::lruPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("dim", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("block_size", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("remove_input_padding", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("paged_state", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("y_enabled", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("y_bias_enabled", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("fuse_gate_enabled", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("gate_bias_enabled", nullptr, PluginFieldType::kINT8));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* lruPluginCreator::getPluginName() const noexcept
{
    return LRU_PLUGIN_NAME;
}

char const* lruPluginCreator::getPluginVersion() const noexcept
{
    return LRU_PLUGIN_VERSION;
}

PluginFieldCollection const* lruPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* lruPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int dim{};
    int block_size{};
    bool removePadding{};
    bool pagedState{};
    bool yEnabled{};
    bool yBiasEnabled{};
    bool fuseGateEnabled{};
    bool gateBiasEnabled{};
    nvinfer1::DataType type{};
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "dim"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            dim = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        if (!strcmp(attrName, "block_size"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            block_size = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "remove_input_padding"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            removePadding = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "paged_state"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            pagedState = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "y_enabled"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            yEnabled = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "y_bias_enabled"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            yBiasEnabled = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "fuse_gate_enabled"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            fuseGateEnabled = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "gate_bias_enabled"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            gateBiasEnabled = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new lruPlugin(
            dim, block_size, type, removePadding, pagedState, yEnabled, yBiasEnabled, fuseGateEnabled, gateBiasEnabled);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* lruPluginCreator::deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call lruPlugin::destroy()
    try
    {
        auto* obj = new lruPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
