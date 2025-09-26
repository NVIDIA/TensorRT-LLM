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

SelectiveScanPlugin::SelectiveScanPlugin(int dim, int dstate, int dtRank, int nHeads, int nGroups, int chunkSize,
    bool deltaSoftplus, nvinfer1::DataType type, bool removePadding, bool pagedState, bool zEnabled, bool isMamba2)
    : mDim(dim)
    , mDState(dstate)
    , mDtRank(dtRank)
    , mNHeads(nHeads)
    , mNGroups(nGroups)
    , mChunkSize(chunkSize)
    , mDeltaSoftplus(deltaSoftplus)
    , mType(type)
    , mRemovePadding(removePadding)
    , mPagedState(pagedState)
    , mZEnabled(zEnabled)
    , mIsMamba2(isMamba2)
    , mDriver(tensorrt_llm::common::CUDADriverWrapper::getInstance())
{
    TLLM_CHECK_WITH_INFO(
        (mChunkSize == 256 || mChunkSize == 128) || (!mIsMamba2), "Only support CHUNK_SIZE 256 or 128");
    TLLM_CHECK_WITH_INFO((mType == DataType::kBF16) || (mType == DataType::kFLOAT) || (mType == DataType::kHALF),
        "Only support float, half, and bfloat16.");
}

// Parameterized constructor
SelectiveScanPlugin::SelectiveScanPlugin(void const* data, size_t length)
    : mDriver(tensorrt_llm::common::CUDADriverWrapper::getInstance())
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mDim);
    read(d, mDState);
    read(d, mDtRank);
    read(d, mNHeads);
    read(d, mNGroups);
    read(d, mChunkSize);
    read(d, mDeltaSoftplus);
    read(d, mType);
    read(d, mRemovePadding);
    read(d, mPagedState);
    read(d, mZEnabled);
    read(d, mIsMamba2);
    TLLM_CHECK(d == a + length);
    TLLM_CHECK_WITH_INFO(
        (mChunkSize == 256 || mChunkSize == 128) || (!mIsMamba2), "Only support CHUNK_SIZE 256 or 128");
    TLLM_CHECK_WITH_INFO((mType == DataType::kBF16) || (mType == DataType::kFLOAT) || (mType == DataType::kHALF),
        "Only support float, half, and bfloat16.");
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* SelectiveScanPlugin::clone() const noexcept
{
    auto* plugin = new SelectiveScanPlugin(mDim, mDState, mDtRank, mNHeads, mNGroups, mChunkSize, mDeltaSoftplus, mType,
        mRemovePadding, mPagedState, mZEnabled, mIsMamba2);
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
        if (mIsMamba2)
        {
            auto ret = inputs[getInputTensorIdx()];
            ret.d[mRemovePadding ? 1 : 2] = exprBuilder.constant(mDim);
            return ret;
        }
        else
        {
            return inputs[getInputTensorIdx()];
        }
    }
    return inputs[getStateIdx()];
}

bool SelectiveScanPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (pos == getHostRequestTypesIdx() || pos == getLastTokenIdsIdx()
        || (mRemovePadding && pos == getHostContextLengthIdx()) || (mPagedState && pos == getSlotMappingIdx()))
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
    if (!mIsMamba2)
        return 0;

    int const NUM_BUFFERS = 6;
    size_t workspaces[NUM_BUFFERS];

    if (mRemovePadding)
    {
        int B = inputs[getLastTokenIdsIdx()].dims.d[0];
        int BxL = inputs[getInputTensorIdx()].dims.d[0]; // num_tokens
        int H = mNHeads;
        int P = mDim / H;
        int G = mNGroups;
        int N = mDState;
        int Q = mChunkSize;
        int BxC = (BxL + Q - 1) / Q + B;

        workspaces[0] = long(BxC) * H * N * P * 2; // g_mxOs_
        workspaces[1] = long(BxC) * H * N * P * 4; // g_mxSt_ in float
        workspaces[2] = long(BxC) * H * Q * 4;     // g_mxdc_ in float
        workspaces[3] = long(BxC) * H * Q * 4;     // g_mxdA_ in float
        workspaces[4] = long(BxC) * G * Q * Q * 2; // g_mxCB_
        workspaces[5] = 1024;                      // TMA descs
    }
    else
    {
        int B = inputs[getInputTensorIdx()].dims.d[0];
        int L = inputs[getInputTensorIdx()].dims.d[1];
        int H = mNHeads;
        int P = mDim / H;
        int G = mNGroups;
        int N = mDState;
        int Q = mChunkSize;
        int C = (L + Q - 1) / Q;

        workspaces[0] = long(B * C) * H * N * P * 2; // g_mxOs_
        workspaces[1] = long(B * C) * H * N * P * 4; // g_mxSt_ in float
        workspaces[2] = long(B * C) * H * Q * 4;     // g_mxdc_ in float
        workspaces[3] = long(B * C) * H * Q * 4;     // g_mxdA_ in float
        workspaces[4] = long(B * C) * G * Q * Q * 2; // g_mxCB_
        workspaces[5] = 1024;                        // TMA descs
    }

    return calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);
}

void SelectiveScanPlugin::setSSMParams(SSMParamsBase& params, const size_t batch, const size_t dim,
    const size_t maxSeqLen, const size_t numTokens, const size_t dstate, const size_t dtRank, const size_t nHeads,
    const size_t nGroups, const size_t chunkSize, void* statePtr, void const* x, void const* delta,
    void const* deltaBias, void const* A, void const* BC, void const* D, void const* z, void* osPtr, void* stPtr,
    void* dcPtr, void* dAPtr, void* cbPtr, void* descPtr, int const* lastTokenIds, int const* slotMapping, void* out,
    bool deltaSoftplus, bool removePadding)
{
    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.max_seqlen = maxSeqLen;
    params.num_tokens = numTokens;
    params.dstate = dstate;
    params.dt_rank = dtRank;
    params.nheads = nHeads;
    params.ngroups = nGroups;
    params.chunk_size = chunkSize;

    params.delta_softplus = deltaSoftplus;
    params.remove_padding = removePadding;
    params.is_mamba2 = mIsMamba2;

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
    params.Os_ptr = osPtr;
    params.St_ptr = stPtr;
    params.dc_ptr = dcPtr;
    params.dA_ptr = dAPtr;
    params.CB_ptr = cbPtr;
    params.desc_ptr = descPtr;
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
    //     1.  state mamba: [batch_size, dstate, dim] or host [1] containing only pointer for paged_state
    //               mamba2: [batch_size, nheads, dstate, dim] or host [1] containing only pointer for paged_state
    //     2.  delta, mamba: [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
    //                mamba2: [batch_size, seq_len, nheads] or [num_tokens, nheads] for remove_input_padding
    //     3.  delta_bias, [dim] for mamba, [nheads] for mamba2
    //     4.  A, [dstate, dim] for mamba, [nheads] for mamba2
    //     5.  BC, mamba: [batch_size, seq_len, dstate * 2] or [num_tokens, dstate * 2] for remove_input_padding
    //             mamba2: [batch_size, seq_len, ngroups * dstate * 2] or [num_tokens, ngroups * dstate * 2] for
    //             remove_input_padding
    //     6.  D, [dim] for mamba, [nheads] for mamba2
    //     7.  host_request_types [batch_size] int32. 0: context; 1: generation.
    //     8.  last_token_ids [batch_size] int32
    //     9.  host_context_lengths [batch_size] int32, optional for remove_input_padding
    //    10.  state_slot_mapping [batch_size] int32, optional for paged state
    //    11.  z [batch_size, max_seq_len, dim] or [num_tokens, dim]
    // outputs
    //     0. output_tensor [batch_size, max_seq_len, dim] or [num_tokens, dim]
    //     1. state, [batch_size, dstate, dim] for mamba, [batch_size, nheads, dstate, dim] for mamba2
    auto const batch_size = inputDesc[getHostRequestTypesIdx()].dims.d[0];
    int max_seq_len;
    if (mRemovePadding)
    {
        int const* host_context_length = static_cast<int const*>(inputs[getHostContextLengthIdx()]);
        max_seq_len = *std::max_element(host_context_length, host_context_length + batch_size);
    }
    else
    {
        max_seq_len = inputDesc[getInputTensorIdx()].dims.d[1];
    }

    // only support context or generation, not for both of them
    RequestType const* reqTypes = static_cast<RequestType const*>(inputs[getHostRequestTypesIdx()]);

    SSMParamsBase ssm_params;

    int const* slotMapping = mPagedState ? static_cast<int const*>(inputs[getSlotMappingIdx()]) : nullptr;
    void const* z = mZEnabled ? inputs[getZIdx()] : nullptr;

    void* statePtr = mPagedState ? *reinterpret_cast<void**>(const_cast<void*>(inputs[getStateIdx()])) : outputs[1];

    // Workspace pointer shift
    int8_t* workspace_byte_ptr = reinterpret_cast<int8_t*>(workspace);
    size_t offset = 0;

    T* mxOs = nullptr;
    float* mxSt = nullptr;
    float* mxdc = nullptr;
    float* mxdA = nullptr;
    T* mxCB = nullptr;
    void* descs = nullptr;

    if (!mIsMamba2 || reqTypes[0] == RequestType::kGENERATION) /* no workspace needed */
        ;
    else if (mRemovePadding)
    {
        int B = inputDesc[getLastTokenIdsIdx()].dims.d[0];
        int BxL = inputDesc[getInputTensorIdx()].dims.d[0]; // num_tokens
        int H = mNHeads;
        int P = mDim / H;
        int G = mNGroups;
        int N = mDState;
        int Q = mChunkSize;
        int BxC = (BxL + Q - 1) / Q + B;

        mxOs = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, long(BxC) * H * N * P * 2));
        mxSt = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, long(BxC) * H * N * P * 4));
        mxdc = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, long(BxC) * H * Q * 4));
        mxdA = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, long(BxC) * H * Q * 4));
        mxCB = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, long(BxC) * G * Q * Q * 2));
        descs = nextWorkspacePtr(workspace_byte_ptr, offset, 1024);
    }
    else
    {
        int B = inputDesc[getInputTensorIdx()].dims.d[0];
        int L = inputDesc[getInputTensorIdx()].dims.d[1];
        int H = mNHeads;
        int P = mDim / H;
        int G = mNGroups;
        int N = mDState;
        int Q = mChunkSize;
        int C = (L + Q - 1) / Q;

        mxOs = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, long(B * C) * H * N * P * 2));
        mxSt = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, long(B * C) * H * N * P * 4));
        mxdc = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, long(B * C) * H * Q * 4));
        mxdA = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, long(B * C) * H * Q * 4));
        mxCB = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, long(B * C) * G * Q * Q * 2));
        descs = nextWorkspacePtr(workspace_byte_ptr, offset, 1024);
    }

    int numTokens = inputDesc[getInputTensorIdx()].dims.d[0];
    if (!mRemovePadding)
        numTokens *= inputDesc[getInputTensorIdx()].dims.d[1];

    setSSMParams(ssm_params, batch_size, mDim, max_seq_len, numTokens, mDState, mDtRank, mNHeads, mNGroups, mChunkSize,
        statePtr, inputs[getInputTensorIdx()], inputs[getDeltaIdx()], inputs[getDeltaBiasIdx()], inputs[getAIdx()],
        inputs[getBCIdx()], inputs[getDIdx()], z, mxOs, mxSt, mxdc, mxdA, mxCB, descs,
        static_cast<int const*>(inputs[getLastTokenIdsIdx()]), slotMapping, outputs[0], mDeltaSoftplus, mRemovePadding);

    if (reqTypes[0] == RequestType::kCONTEXT)
    {
        if (mIsMamba2)
        {
            invokeChunkScan<T, float>(ssm_params, stream, mDriver.get());
        }
        else
        {
            invokeSelectiveScan<T, float>(ssm_params, stream);
        }
    }
    else if (reqTypes[0] == RequestType::kGENERATION)
    {
        invokeSelectiveScanUpdate<T, float>(ssm_params, stream);
    }
    sync_check_cuda_error(stream);
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
    return sizeof(mDim) + sizeof(mDState) + sizeof(mDtRank) + sizeof(mNHeads) + sizeof(mNGroups) + sizeof(mChunkSize)
        + sizeof(mDeltaSoftplus) + sizeof(mType) + sizeof(mRemovePadding) + sizeof(mPagedState) + sizeof(mZEnabled)
        + sizeof(mIsMamba2);
}

void SelectiveScanPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mDim);
    write(d, mDState);
    write(d, mDtRank);
    write(d, mNHeads);
    write(d, mNGroups);
    write(d, mChunkSize);
    write(d, mDeltaSoftplus);
    write(d, mType);
    write(d, mRemovePadding);
    write(d, mPagedState);
    write(d, mZEnabled);
    write(d, mIsMamba2);
    TLLM_CHECK(d == a + getSerializationSize());
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
    mPluginAttributes.emplace_back(PluginField("dim", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("dstate", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("dt_rank", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("nheads", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("ngroups", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("chunk_size", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("delta_softplus", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("remove_input_padding", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("paged_state", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("z_enabled", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("is_mamba2", nullptr, PluginFieldType::kINT8));
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
    int dim{};
    int dstate{};
    int dtRank{};
    int nHeads{};
    int nGroups{};
    int chunkSize{};
    bool deltaSoftplus{};
    bool removePadding{};
    bool pagedState{};
    bool zEnabled{};
    bool isMamab2{};
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
        else if (!strcmp(attrName, "nheads"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            nHeads = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "ngroups"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            nGroups = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "chunk_size"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            chunkSize = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
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
        else if (!strcmp(attrName, "z_enabled"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            zEnabled = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "is_mamba2"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            isMamab2 = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new SelectiveScanPlugin(dim, dstate, dtRank, nHeads, nGroups, chunkSize, deltaSoftplus, type,
            removePadding, pagedState, zEnabled, isMamab2);
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
