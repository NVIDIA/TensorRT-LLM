/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gatedDeltaNetPlugin.h"
#include "tensorrt_llm/common/assert.h"

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::GatedDeltaNetPluginCreator;
using tensorrt_llm::plugins::GatedDeltaNetPlugin;

static char const* GATED_DELTA_NET_PLUGIN_VERSION{"1"};
static char const* GATED_DELTA_NET_PLUGIN_NAME{"GatedDeltaNet"};
PluginFieldCollection GatedDeltaNetPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> GatedDeltaNetPluginCreator::mPluginAttributes;

GatedDeltaNetPlugin::GatedDeltaNetPlugin(int numVHeads, int headKDim, int headVDim, int chunkSize, bool useQkL2norm,
    nvinfer1::DataType type, bool removePadding, bool pagedState)
    : mNumVHeads(numVHeads)
    , mHeadKDim(headKDim)
    , mHeadVDim(headVDim)
    , mChunkSize(chunkSize)
    , mUseQkL2norm(useQkL2norm)
    , mType(type)
    , mRemovePadding(removePadding)
    , mPagedState(pagedState)
{
    // CORRECTNESS-FIRST: the recurrent state is always fp32, and for this first
    // version we only support the fp32 compute path.
    TLLM_CHECK_WITH_INFO(mType == DataType::kFLOAT, "GatedDeltaNetPlugin only supports float (fp32) for now.");
}

// Parameterized constructor
GatedDeltaNetPlugin::GatedDeltaNetPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mNumVHeads);
    read(d, mHeadKDim);
    read(d, mHeadVDim);
    read(d, mChunkSize);
    read(d, mUseQkL2norm);
    read(d, mType);
    read(d, mRemovePadding);
    read(d, mPagedState);
    TLLM_CHECK(d == a + length);
    TLLM_CHECK_WITH_INFO(mType == DataType::kFLOAT, "GatedDeltaNetPlugin only supports float (fp32) for now.");
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* GatedDeltaNetPlugin::clone() const noexcept
{
    auto* plugin = new GatedDeltaNetPlugin(
        mNumVHeads, mHeadKDim, mHeadVDim, mChunkSize, mUseQkL2norm, mType, mRemovePadding, mPagedState);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

// Outputs
//     0. y             [B, T, H_v, D_v] or [num_tokens, H_v, D_v] for remove_input_padding
//     1. present_state [B, H_v, D_k, D_v] fp32 (omitted iff paged_state)
nvinfer1::DimsExprs GatedDeltaNetPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    if (outputIndex == 0)
    {
        // y has the same leading layout as v (which carries D_v as the last dim).
        return inputs[getVIdx()];
    }
    // present_state mirrors the input recurrent state [B, H_v, D_k, D_v].
    return inputs[getStateIdx()];
}

bool GatedDeltaNetPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (pos == getHostRequestTypesIdx() || pos == getLastTokenIdsIdx()
        || (mRemovePadding && pos == getHostContextLengthIdx()) || (mPagedState && pos == getSlotMappingIdx()))
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else if (mPagedState && pos == getStateIdx())
    {
        // host pointer to the paged recurrent state
        return inOut[pos].type == nvinfer1::DataType::kINT64;
    }
    else
    {
        // q, k, v, g, beta, (dense) state, y, present_state are all fp32 / linear.
        return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
}

void GatedDeltaNetPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t GatedDeltaNetPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    // The kernel keeps per-thread local state (one S-column per thread); no
    // device scratch is required.
    return 0;
}

template <typename T>
int GatedDeltaNetPlugin::enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    // inputs
    //     0. q     [B, T, H_v, D_k]
    //     1. k     [B, T, H_v, D_k]
    //     2. v     [B, T, H_v, D_v]
    //     3. g     [B, T, H_v]
    //     4. beta  [B, T, H_v]
    //     5. state_or_ptr [B, H_v, D_k, D_v] fp32 (or host [1] int64 ptr if paged_state)
    //     6. host_request_types [B] int32 (0: context, 1: generation)
    //     7. last_token_ids     [B] int32
    //     8. host_context_lengths [B] int32 (iff remove_input_padding)
    //     9. slot_mapping         [B] int32 (iff paged_state)
    // outputs
    //     0. y             [B, T, H_v, D_v]
    //     1. present_state [B, H_v, D_k, D_v] fp32 (omitted iff paged_state)

    // CORRECTNESS-FIRST: only the dense (non-paged, non-remove-padding) path is
    // wired up for now. The paged_state / remove_input_padding signatures are
    // parsed and stored so the plugin is future-proof, but their kernel handling
    // is not yet implemented.
    TLLM_CHECK_WITH_INFO(!mPagedState && !mRemovePadding,
        "GatedDeltaNetPlugin: paged_state and remove_input_padding paths are not yet implemented; only the dense "
        "[B,T,...] non-paged path is supported.");

    auto const batchSize = inputDesc[getHostRequestTypesIdx()].dims.d[0];
    // Dense path: T is the padded sequence length carried by q.
    int const maxSeqLen = inputDesc[getQIdx()].dims.d[1];

    // host_request_types selects context vs generation for the whole batch (we do
    // not support mixing context and generation in one enqueue, matching
    // selectiveScanPlugin).
    RequestType const* reqTypes = static_cast<RequestType const*>(inputs[getHostRequestTypesIdx()]);

    // present_state is a dedicated output buffer in the dense path; the kernel
    // reads from the input state and writes the updated state here.
    void* statePtrIn = const_cast<void*>(inputs[getStateIdx()]);
    void* statePtrOut = outputs[1];

    // Zero-initialize y so that padding tokens beyond each request's valid length
    // (which the kernel never writes) are well-defined zeros, matching the
    // reference launcher semantics.
    {
        size_t const yElems = static_cast<size_t>(batchSize) * maxSeqLen * mNumVHeads * mHeadVDim;
        TLLM_CUDA_CHECK(cudaMemsetAsync(outputs[0], 0, yElems * sizeof(T), stream));
    }

    GatedDeltaNetParams params;
    memset(&params, 0, sizeof(params));
    params.batch = static_cast<int>(batchSize);
    params.maxSeqLen = maxSeqLen;
    params.numVHeads = mNumVHeads;
    params.headKDim = mHeadKDim;
    params.headVDim = mHeadVDim;
    params.q = inputs[getQIdx()];
    params.k = inputs[getKIdx()];
    params.v = inputs[getVIdx()];
    params.g = inputs[getGIdx()];
    params.beta = inputs[getBetaIdx()];
    params.statePtrIn = statePtrIn;
    // For context, last_token_ids carries the per-request inclusive valid length
    // (non-removePadding case). For generation each request processes exactly one
    // token; the kernel reads seqlens[b] directly, so this maps cleanly.
    params.seqLens = static_cast<int const*>(inputs[getLastTokenIdsIdx()]);
    // RequestType is an int32-backed enum; the params struct carries the raw
    // int32 host pointer (the kernel does not branch on it in this dense path).
    params.hostReqTypes = reinterpret_cast<int const*>(reqTypes);
    params.y = outputs[0];
    params.statePtrOut = statePtrOut;
    params.useQkL2norm = mUseQkL2norm;

    invokeGatedDeltaNet(params, stream);

    sync_check_cuda_error(stream);
    return 0;
}

int GatedDeltaNetPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }
    if (mType == DataType::kFLOAT)
    {
        return enqueueImpl<float>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType GatedDeltaNetPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    if (index == 0)
    {
        return inputTypes[getVIdx()];
    }
    return inputTypes[getStateIdx()];
}

// IPluginV2 Methods

char const* GatedDeltaNetPlugin::getPluginType() const noexcept
{
    return GATED_DELTA_NET_PLUGIN_NAME;
}

char const* GatedDeltaNetPlugin::getPluginVersion() const noexcept
{
    return GATED_DELTA_NET_PLUGIN_VERSION;
}

int GatedDeltaNetPlugin::getNbOutputs() const noexcept
{
    return mPagedState ? 1 : 2;
}

int GatedDeltaNetPlugin::initialize() noexcept
{
    return 0;
}

void GatedDeltaNetPlugin::terminate() noexcept {}

size_t GatedDeltaNetPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNumVHeads) + sizeof(mHeadKDim) + sizeof(mHeadVDim) + sizeof(mChunkSize) + sizeof(mUseQkL2norm)
        + sizeof(mType) + sizeof(mRemovePadding) + sizeof(mPagedState);
}

void GatedDeltaNetPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mNumVHeads);
    write(d, mHeadKDim);
    write(d, mHeadVDim);
    write(d, mChunkSize);
    write(d, mUseQkL2norm);
    write(d, mType);
    write(d, mRemovePadding);
    write(d, mPagedState);
    TLLM_CHECK(d == a + getSerializationSize());
}

void GatedDeltaNetPlugin::destroy() noexcept
{
    delete this;
}

///////////////

GatedDeltaNetPluginCreator::GatedDeltaNetPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_v_heads", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("head_k_dim", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("head_v_dim", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("chunk_size", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("use_qk_l2norm", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("remove_input_padding", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("paged_state", nullptr, PluginFieldType::kINT8));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* GatedDeltaNetPluginCreator::getPluginName() const noexcept
{
    return GATED_DELTA_NET_PLUGIN_NAME;
}

char const* GatedDeltaNetPluginCreator::getPluginVersion() const noexcept
{
    return GATED_DELTA_NET_PLUGIN_VERSION;
}

PluginFieldCollection const* GatedDeltaNetPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* GatedDeltaNetPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int numVHeads{};
    int headKDim{};
    int headVDim{};
    int chunkSize{};
    bool useQkL2norm{};
    bool removePadding{};
    bool pagedState{};
    nvinfer1::DataType type{};
    // Read configurations from each field.
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "num_v_heads"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            numVHeads = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "head_k_dim"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            headKDim = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "head_v_dim"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            headVDim = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "chunk_size"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            chunkSize = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "use_qk_l2norm"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            useQkL2norm = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
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
        auto* obj = new GatedDeltaNetPlugin(
            numVHeads, headKDim, headVDim, chunkSize, useQkL2norm, type, removePadding, pagedState);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* GatedDeltaNetPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call GatedDeltaNetPlugin::destroy()
    try
    {
        auto* obj = new GatedDeltaNetPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
