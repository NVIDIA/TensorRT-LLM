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

static const char* SELECTIVE_SCAN_PLUGIN_VERSION{"1"};
static const char* SELECTIVE_SCAN_PLUGIN_NAME{"SelectiveScan"};
PluginFieldCollection SelectiveScanPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> SelectiveScanPluginCreator::mPluginAttributes;

SelectiveScanPlugin::SelectiveScanPlugin(
    int dim, int dstate, bool isVariableB, bool isVariableC, bool deltaSoftplus, nvinfer1::DataType type)
    : mDim(dim)
    , mDState(dstate)
    , mIsVariableB(isVariableB)
    , mIsVariableC(isVariableC)
    , mDeltaSoftplus(deltaSoftplus)
    , mType(type)
{
    TLLM_CHECK_WITH_INFO((getSMVersion() >= 80) || (mType != DataType::kBF16),
        "Unsupported data type, pre SM 80 GPUs do not support bfloat16");
    TLLM_CHECK_WITH_INFO((mType == DataType::kBF16) || (mType == DataType::kFLOAT) || (mType == DataType::kHALF),
        "Only support float, half, and bfloat16.");
}

// Parameterized constructor
SelectiveScanPlugin::SelectiveScanPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mDim);
    read(d, mDState);
    read(d, mIsVariableB);
    read(d, mIsVariableC);
    read(d, mDeltaSoftplus);
    read(d, mType);
    TLLM_CHECK(d == a + length);
    TLLM_CHECK_WITH_INFO((getSMVersion() >= 80) || (mType != DataType::kBF16), "Unsupported data type");
    TLLM_CHECK_WITH_INFO((mType == DataType::kBF16) || (mType == DataType::kFLOAT) || (mType == DataType::kHALF),
        "Only support float, half, and bfloat16.");
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* SelectiveScanPlugin::clone() const noexcept
{
    auto* plugin = new SelectiveScanPlugin(mDim, mDState, mIsVariableB, mIsVariableC, mDeltaSoftplus, mType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

// Outputs
//     output_tensor: [batch_size, dim, seq_len]
//     state: [batch_size, dim, dstate]
nvinfer1::DimsExprs SelectiveScanPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    if (outputIndex == 0)
    {
        return inputs[getInputTensorIdx()];
    }
    return inputs[getStateIdx()];
}

bool SelectiveScanPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (pos == getHostRequestTypesIdx())
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else if (pos == getAIdx() || pos == getDeltaBiasIdx() || pos == getDIdx() || pos == nbInputs + 1)
    {
        return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else
    {
        return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
}

void SelectiveScanPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t SelectiveScanPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

void SelectiveScanPlugin::setSSMParams(SSMParamsBase& params, const size_t batch, const size_t dim, const size_t seqLen,
    const size_t dstate, const size_t nChunks, const bool isVariableB, const bool isVariableC, void* statePtr,
    const void* x, const void* delta, const void* deltaBias, const void* A, const void* B, const void* C, const void* D,
    const void* z, void* out, const size_t strideXBatch, const size_t strideDtBatch, const size_t strideADim,
    const size_t strideBBatch, const size_t strideCBatch, const size_t strideZBatch, const size_t strideOutBatch,
    const size_t strideStateBatch, const size_t strideStateDim, bool deltaSoftplus)
{
    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqLen;
    params.dstate = dstate;
    params.n_groups = 1;
    params.n_chunks = nChunks;
    params.dim_ngroups_ratio = dim;

    params.delta_softplus = deltaSoftplus;

    params.is_variable_B = isVariableB;
    params.is_variable_C = isVariableC;

    // Set the pointers and strides.
    params.u_ptr = const_cast<void*>(x);
    params.delta_ptr = const_cast<void*>(delta);
    params.A_ptr = const_cast<void*>(A);
    params.B_ptr = const_cast<void*>(B);
    params.C_ptr = const_cast<void*>(C);
    params.D_ptr = const_cast<void*>(D);
    params.delta_bias_ptr = const_cast<void*>(deltaBias);
    params.out_ptr = out;
    params.x_ptr = statePtr;
    params.z_ptr = const_cast<void*>(z);
    // All stride are in elements, not bytes.
    params.A_d_stride = strideADim;
    params.A_dstate_stride = 1;
    if (!isVariableB)
    {
        params.B_d_stride = dim * dstate;
    }
    else
    {
        params.B_batch_stride = strideBBatch;
        params.B_group_stride = strideBBatch;
    }
    params.B_dstate_stride = !isVariableB ? dstate : seqLen;
    if (!isVariableC)
    {
        params.C_d_stride = dim * dstate;
    }
    else
    {
        params.C_batch_stride = strideCBatch;
        params.C_group_stride = strideCBatch;
    }
    params.C_dstate_stride = !isVariableC ? dstate : seqLen;
    params.u_batch_stride = strideXBatch;
    params.u_d_stride = seqLen;
    params.delta_batch_stride = strideDtBatch;
    params.delta_d_stride = seqLen;
    params.z_batch_stride = strideZBatch;
    params.z_d_stride = seqLen;
    params.out_batch_stride = strideOutBatch;
    params.out_d_stride = seqLen;
    params.state_batch_stride = strideStateBatch;
    params.state_d_stride = strideStateDim;
}

template <typename T>
int SelectiveScanPlugin::enqueueImpl(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    // inputs
    //     0.  input_tensor [batch_size, dim, seq_len]
    //     1.  state [batch_size, dim, dstate]
    //     2.  delta [batch_size, dim, seq_len]
    //     3.  delta_bias [dim]
    //     4.  A [dim, dstate]
    //     5.  B [batch_size, dstate, seq_len]
    //     6.  C [batch_size, dstate, seq_len]
    //     7.  D [dim]
    //     8.  z [batch_size, dim, seq_len]
    //     9.  host_request_types [batch_size] int32. 0: context; 1: generation.
    // outputs
    //     0. output_tensor [batch_size, dim, seq_len]
    //     1. state [batch_size, dim, dstate]
    auto const batch_size = inputDesc[getInputTensorIdx()].dims.d[0];
    auto const seq_len = inputDesc[getInputTensorIdx()].dims.d[2];
    auto const stride_state_batch = mDim * mDState;
    auto const stride_state_dim = mDState;
    auto const stride_x_batch = mDim * seq_len;
    auto const stride_dt_batch = mDim * seq_len;
    auto const stride_A_dim = mDState;
    auto const stride_B_batch = mDState * seq_len;
    auto const stride_C_batch = mDState * seq_len;
    auto const stride_z_batch = mDim * seq_len;
    auto const stride_out_batch = mDim * seq_len;

    // only support context or generation, not for both of them
    RequestType const* reqTypes = static_cast<RequestType const*>(inputs[getHostRequestTypesIdx()]);

    auto const n_chunks = (seq_len + 2048 - 1) / 2048;
    SSMParamsBase ssm_params;
    setSSMParams(ssm_params, batch_size, mDim, seq_len, mDState, n_chunks, mIsVariableB, mIsVariableC, outputs[1],
        inputs[getInputTensorIdx()], inputs[getDeltaIdx()], inputs[getDeltaBiasIdx()], inputs[getAIdx()],
        inputs[getBIdx()], inputs[getCIdx()], inputs[getDIdx()], inputs[getZIdx()], outputs[0], stride_x_batch,
        stride_dt_batch, stride_A_dim, stride_B_batch, stride_C_batch, stride_z_batch, stride_out_batch,
        stride_state_batch, stride_state_dim, mDeltaSoftplus);

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

int SelectiveScanPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
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
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
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

const char* SelectiveScanPlugin::getPluginType() const noexcept
{
    return SELECTIVE_SCAN_PLUGIN_NAME;
}

const char* SelectiveScanPlugin::getPluginVersion() const noexcept
{
    return SELECTIVE_SCAN_PLUGIN_VERSION;
}

int SelectiveScanPlugin::getNbOutputs() const noexcept
{
    return 2;
}

int SelectiveScanPlugin::initialize() noexcept
{
    return 0;
}

void SelectiveScanPlugin::terminate() noexcept {}

size_t SelectiveScanPlugin::getSerializationSize() const noexcept
{
    return sizeof(mDim) + sizeof(mDState) + sizeof(mIsVariableB) + sizeof(mIsVariableC) + sizeof(mDeltaSoftplus)
        + sizeof(mType);
}

void SelectiveScanPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mDim);
    write(d, mDState);
    write(d, mIsVariableB);
    write(d, mIsVariableC);
    write(d, mDeltaSoftplus);
    write(d, mType);
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
    mPluginAttributes.emplace_back(PluginField("is_variable_B", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("is_variable_C", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("delta_softplus", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SelectiveScanPluginCreator::getPluginName() const noexcept
{
    return SELECTIVE_SCAN_PLUGIN_NAME;
}

const char* SelectiveScanPluginCreator::getPluginVersion() const noexcept
{
    return SELECTIVE_SCAN_PLUGIN_VERSION;
}

const PluginFieldCollection* SelectiveScanPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* SelectiveScanPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int dim, dstate;
    bool isVariableB, isVariableC, deltaSoftplus;
    nvinfer1::DataType type;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "dim"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            dim = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "dstate"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            dstate = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "is_variable_B"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            isVariableB = static_cast<bool>(*(static_cast<const bool*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "is_variable_C"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            isVariableC = static_cast<bool>(*(static_cast<const bool*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "delta_softplus"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            deltaSoftplus = static_cast<bool>(*(static_cast<const bool*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new SelectiveScanPlugin(dim, dstate, isVariableB, isVariableC, deltaSoftplus, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SelectiveScanPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call SelectiveScanPlugin::destroy()
    try
    {
        auto* obj = new SelectiveScanPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
