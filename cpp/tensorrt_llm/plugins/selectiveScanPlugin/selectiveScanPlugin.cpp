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

#include "selectiveScanPlugin.h"
#include "tensorrt_llm/common/assert.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::SelectiveScanPluginCreator;
using tensorrt_llm::plugins::SelectiveScanPlugin;

static char const* SELECTIVE_SCAN_PLUGIN_VERSION{"1"};
static char const* SELECTIVE_SCAN_PLUGIN_NAME{"SelectiveScan"};
PluginFieldCollection SelectiveScanPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> SelectiveScanPluginCreator::mPluginAttributes;

SelectiveScanPlugin::SelectiveScanPlugin(int dim, int dstate, int dt_rank, bool isVariableB, bool isVariableC,
    bool deltaSoftplus, nvinfer1::DataType type, bool removePadding, bool pagedState)
    : mDim(dim)
    , mDState(dstate)
    , mDtRank(dt_rank)
    , mIsVariableB(isVariableB)
    , mIsVariableC(isVariableC)
    , mDeltaSoftplus(deltaSoftplus)
    , mType(type)
    , mRemovePadding(removePadding)
    , mPagedState(pagedState)
{
    TLLM_CHECK_WITH_INFO((getSMVersion() >= 80) || (mType != DataType::kBF16),
        "Unsupported data type, pre SM 80 GPUs do not support bfloat16");
    TLLM_CHECK_WITH_INFO((mType == DataType::kBF16) || (mType == DataType::kFLOAT) || (mType == DataType::kHALF),
        "Only support float, half, and bfloat16.");
}

// Parameterized constructor
SelectiveScanPlugin::SelectiveScanPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mDim);
    read(d, mDState);
    read(d, mDtRank);
    read(d, mIsVariableB);
    read(d, mIsVariableC);
    read(d, mDeltaSoftplus);
    read(d, mType);
    read(d, mRemovePadding);
    read(d, mPagedState);
    TLLM_CHECK(d == a + length);
    TLLM_CHECK_WITH_INFO((getSMVersion() >= 80) || (mType != DataType::kBF16), "Unsupported data type");
    TLLM_CHECK_WITH_INFO((mType == DataType::kBF16) || (mType == DataType::kFLOAT) || (mType == DataType::kHALF),
        "Only support float, half, and bfloat16.");
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* SelectiveScanPlugin::clone() const noexcept
{
    auto* plugin = new SelectiveScanPlugin(
        mDim, mDState, mDtRank, mIsVariableB, mIsVariableC, mDeltaSoftplus, mType, mRemovePadding, mPagedState);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

// Outputs
//     output_tensor: [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
//     state: [batch_size, dstate, dim]
nvinfer1::DimsExprs SelectiveScanPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    if (outputIndex == 0)
    {
        return inputs[getInputTensorIdx()];
    }
    return inputs[getStateIdx()];
}

bool SelectiveScanPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (pos == getHostRequestTypesIdx() || pos == getLastTokenIdsIdx() || (mPagedState && pos == getSlotMappingIdx()))
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else if (pos == getAIdx() || pos == getDeltaBiasIdx() || pos == getDIdx())
    {
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else if (mPagedState && pos == getStateIdx())
    {
        return inOut[pos].type == nvinfer1::DataType::kINT64;
    }
    else
    {
        return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
}

void SelectiveScanPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t SelectiveScanPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

void SelectiveScanPlugin::setSSMParams(SSMParamsBase& params, const size_t batch, const size_t dim,
    const size_t maxSeqLen, const size_t dstate, const size_t dtRank, bool const isVariableB, bool const isVariableC,
    void* statePtr, void const* x, void const* delta, void const* deltaBias, void const* A, void const* BC,
    void const* D, void const* z, int const* lastTokenIds, int const* slotMapping, void* out, bool deltaSoftplus,
    bool removePadding)
{
    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.max_seqlen = maxSeqLen;
    params.dstate = dstate;
    params.dt_rank = dtRank;

    params.delta_softplus = deltaSoftplus;
    params.remove_padding = removePadding;

    params.is_variable_B = isVariableB;
    params.is_variable_C = isVariableC;

    // Set the pointers and strides.
    params.u_ptr = const_cast<void*>(x);
    params.delta_ptr = const_cast<void*>(delta);
    params.A_ptr = const_cast<void*>(A);
    params.BC_ptr = const_cast<void*>(BC);
    params.D_ptr = const_cast<void*>(D);
    params.delta_bias_ptr = const_cast<void*>(deltaBias);
    params.out_ptr = out;
    params.x_ptr = statePtr;
    params.z_ptr = const_cast<void*>(z);
    params.last_token_ids_ptr = lastTokenIds;
    params.slot_mapping_ptr = slotMapping;
}

template <typename T>
int SelectiveScanPlugin::enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    // inputs
    //     0.  input_tensor [batch_size, max_seq_len, dim] or [num_tokens, dim]
    //     1.  state [batch_size, dstate, dim] or host [1] containing only pointer for paged_state
    //     2.  delta [batch_size, max_seq_len, dim] or [num_tokens, dim]
    //     3.  delta_bias [dim]
    //     4.  A [dstate, dim]
    //     5.  BC [batch_size, max_seq_len, dt_rank + dstate * 2] or [num_tokens, dt_rank + dstate * 2]
    //     6.  D [dim]
    //     7.  z [batch_size, max_seq_len, dim] or [num_tokens, dim]
    //     8.  host_request_types [batch_size] int32. 0: context; 1: generation.
    //     9.  last_token_ids [batch_size] int32
    //    10.  state_slot_mapping [batch_size] int32, optional for paged state
    // outputs
    //     0. output_tensor [batch_size, max_seq_len, dim] or [num_tokens, dim]
    //     1. state [batch_size, dstate, dim]
    auto const batch_size = inputDesc[getHostRequestTypesIdx()].dims.d[0];
    int max_seq_len;
    if (mRemovePadding)
    {
        max_seq_len = -1;
    }
    else
    {
        max_seq_len = inputDesc[getInputTensorIdx()].dims.d[1];
    }

    // only support context or generation, not for both of them
    RequestType const* reqTypes = static_cast<RequestType const*>(inputs[getHostRequestTypesIdx()]);

    SSMParamsBase ssm_params;

    int const* slotMapping = mPagedState ? static_cast<int const*>(inputs[getSlotMappingIdx()]) : nullptr;

    void* statePtr = mPagedState ? *reinterpret_cast<void**>(const_cast<void*>(inputs[getStateIdx()])) : outputs[1];

    setSSMParams(ssm_params, batch_size, mDim, max_seq_len, mDState, mDtRank, mIsVariableB, mIsVariableC, statePtr,
        inputs[getInputTensorIdx()], inputs[getDeltaIdx()], inputs[getDeltaBiasIdx()], inputs[getAIdx()],
        inputs[getBCIdx()], inputs[getDIdx()], inputs[getZIdx()], static_cast<int const*>(inputs[getLastTokenIdsIdx()]),
        slotMapping, outputs[0], mDeltaSoftplus, mRemovePadding);

    if (reqTypes[0] == RequestType::kCONTEXT)
    {
        invokeSelectiveScan<T, float>(ssm_params, stream);
    }
    else if (reqTypes[0] == RequestType::kGENERATION)
    {
        invokeSelectiveScanUpdate<T, float>(ssm_params, stream);
    }
    return 0;
}

int SelectiveScanPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
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
nvinfer1::DataType SelectiveScanPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    if (index == 0)
    {
        return inputTypes[getInputTensorIdx()];
    }
    else
    {
        return inputTypes[getStateIdx()];
    }
}

// IPluginV2 Methods

char const* SelectiveScanPlugin::getPluginType() const noexcept
{
    return SELECTIVE_SCAN_PLUGIN_NAME;
}

char const* SelectiveScanPlugin::getPluginVersion() const noexcept
{
    return SELECTIVE_SCAN_PLUGIN_VERSION;
}

int SelectiveScanPlugin::getNbOutputs() const noexcept
{
    return mPagedState ? 1 : 2;
}

int SelectiveScanPlugin::initialize() noexcept
{
    return 0;
}

void SelectiveScanPlugin::terminate() noexcept {}

size_t SelectiveScanPlugin::getSerializationSize() const noexcept
{
    return sizeof(mDim) + sizeof(mDState) + sizeof(mDtRank) + sizeof(mIsVariableB) + sizeof(mIsVariableC)
        + sizeof(mDeltaSoftplus) + sizeof(mType) + sizeof(mRemovePadding) + sizeof(mPagedState);
}

void SelectiveScanPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mDim);
    write(d, mDState);
    write(d, mDtRank);
    write(d, mIsVariableB);
    write(d, mIsVariableC);
    write(d, mDeltaSoftplus);
    write(d, mType);
    write(d, mRemovePadding);
    write(d, mPagedState);
    assert(d == a + getSerializationSize());
}

void SelectiveScanPlugin::destroy() noexcept
{
    delete this;
}

///////////////

SelectiveScanPluginCreator::SelectiveScanPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("dim", nullptr, PluginFieldType::kINT32, 16));
    mPluginAttributes.emplace_back(PluginField("dstate", nullptr, PluginFieldType::kINT32, 16));
    mPluginAttributes.emplace_back(PluginField("dt_rank", nullptr, PluginFieldType::kINT32, 16));
    mPluginAttributes.emplace_back(PluginField("is_variable_B", nullptr, PluginFieldType::kINT8, 1));
    mPluginAttributes.emplace_back(PluginField("is_variable_C", nullptr, PluginFieldType::kINT8, 1));
    mPluginAttributes.emplace_back(PluginField("delta_softplus", nullptr, PluginFieldType::kINT8, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("remove_input_padding", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("paged_state", nullptr, PluginFieldType::kINT8, 0));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* SelectiveScanPluginCreator::getPluginName() const noexcept
{
    return SELECTIVE_SCAN_PLUGIN_NAME;
}

char const* SelectiveScanPluginCreator::getPluginVersion() const noexcept
{
    return SELECTIVE_SCAN_PLUGIN_VERSION;
}

PluginFieldCollection const* SelectiveScanPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* SelectiveScanPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int dim, dstate, dtRank;
    bool isVariableB, isVariableC, deltaSoftplus, removePadding, pagedState;
    nvinfer1::DataType type;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "dim"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            dim = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "dstate"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            dstate = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "dt_rank"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            dtRank = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "is_variable_B"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            isVariableB = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "is_variable_C"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            isVariableC = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "delta_softplus"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            deltaSoftplus = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
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
    }
    try
    {
        auto* obj = new SelectiveScanPlugin(
            dim, dstate, dtRank, isVariableB, isVariableC, deltaSoftplus, type, removePadding, pagedState);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SelectiveScanPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call SelectiveScanPlugin::destroy()
    try
    {
        auto* obj = new SelectiveScanPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
