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
#include "wanAttentionPlugin.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/recoverFromRingAtten.h"
#include "tensorrt_llm/kernels/sageAttentionKernels.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include <cassert>

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;

using tensorrt_llm::plugins::WanAttentionPluginCreator;
using tensorrt_llm::plugins::WanAttentionPlugin;

static char const* WAN_ATTENTION_PLUGIN_VERSION{"1"};
static char const* WAN_ATTENTION_PLUGIN_NAME{"WanAttention"};
PluginFieldCollection WanAttentionPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> WanAttentionPluginCreator::mPluginAttributes;

WanAttentionPlugin::WanAttentionPlugin(
    int num_heads, int head_size, float q_scaling, ContextFMHAType context_fmha_type, nvinfer1::DataType type)
    : mNumHeads(num_heads)
    , mHeadSize(head_size)
    , mQScaling(q_scaling)
    , mType(type)
    , mEnableContextFMHA(context_fmha_type != ContextFMHAType::DISABLED)
    , mFMHAForceFP32Acc(context_fmha_type == ContextFMHAType::ENABLED_WITH_FP32_ACC)
{
    // pre-check whether FMHA is supported in order to save memory allocation
    if (mEnableContextFMHA)
    {
        mEnableContextFMHA = false;
        if (!(mType == DataType::kHALF || mType == DataType::kBF16))
        {
            TLLM_LOG_WARNING("Fall back to unfused MHA because of unsupported data type.");
        }
        else
        {
            mEnableContextFMHA = true;
        }
    }
}

// Parameterized constructor
WanAttentionPlugin::WanAttentionPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mNumHeads);
    read(d, mHeadSize);
    read(d, mQScaling);
    read(d, mQKHalfAccum);
    read(d, mEnableContextFMHA);
    read(d, mFMHAForceFP32Acc);
    read(d, mType);

    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* WanAttentionPlugin::clone() const noexcept
{
    auto* plugin = new WanAttentionPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
}

nvinfer1::DimsExprs WanAttentionPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    TLLM_CHECK(outputIndex == 0);
    auto ret = inputs[0];
    return ret;
}

bool WanAttentionPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    // inputs: [0] query, [1] kv_packed, [2] cu_seqlen_q, [3] cu_seqlen_kv
    // outputs: [X] hidden_states
    if (nbInputs == 4)
    { // BERT
        if (pos == 2 or pos == 3)
        {
            return inOut[pos].type == nvinfer1::DataType::kINT32;
        }

        return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    return false;
}

void WanAttentionPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t WanAttentionPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    // query: [batch_size, q_seq_len, num_head, head_dim]
    // kv: [batch_size, 2, num_head, kv_seq_len, head_dim]
    // cu_seqlen_q: [batch_size + 1]
    // cu_seqlen_kv: [batch_size + 1]

    int const batch_size = inputs[0].dims.d[0];

    bool ret = false;
    if (inputs[0].dims.d[3] != inputs[1].dims.d[4])
    {
        printf("inner dim of q,k and v should match!");
        ret = true;
    }
    if (batch_size + 1 != inputs[2].dims.d[0])
    {
        printf("cu seqlen_q should have dim of (batch_size + 1)");
        ret = true;
    }
    if (batch_size + 1 != inputs[3].dims.d[0])
    {
        printf("cu seqlen_kv should have dim of (batch_size + 1)");
        ret = true;
    }
    if (ret)
    {
        return -1;
    }

    auto const size = tensorrt_llm::runtime::BufferDataType(inputs[0].type).getSize();

    size_t const fmha_scheduler_counter = mEnableContextFMHA ? sizeof(uint32_t) : 0;

    int const NUM_BUFFERS = 1;

    size_t workspaces[NUM_BUFFERS];
    workspaces[0] = fmha_scheduler_counter;

    return tc::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);
}

template <typename T>
int WanAttentionPlugin::enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{

    // inputs
    //     query: [batch_size, q_seq_len, num_head, head_dim]
    //     kv: [batch_size,  kv_seq_len, 2, num_head, head_dim]
    //     cu_seqlen_q: [batch_size + 1]
    //     cu_seqlen_kv: [batch_size + 1]
    // outputs
    //     output_tensor [batch_size, q_seq_len, num_head, head_dim]

    // if remove padding, inputs[0] dim is [num_tokens] which doesn't have workspace info
    // should get max_batch_size from inputs[1] and max_input_length from plugin attribute
    int const batch_size = inputDesc[0].dims.d[0];
    int const q_seq_len = inputDesc[0].dims.d[1];
    int const kv_seq_len = inputDesc[1].dims.d[1];
    int const request_batch_size = batch_size;

    assert(inputs[0].dims.d[3] == inputs[1].dims.d[4] && "inner dim of q,k and v should match!");
    assert(batch_size + 1 == inputs[2].dims.d[0] && "cu seqlen should have dim of (batch_size + 1)");
    assert(batch_size + 1 == inputs[3].dims.d[0] && "cu seqlen should have dim of (batch_size + 1)");

    T const* query = reinterpret_cast<T const*>(inputs[0]);
    T const* kv_packed = reinterpret_cast<T const*>(inputs[1]);
    int const* cu_seqlens_q = reinterpret_cast<int const*>(inputs[2]);
    int const* cu_seqlens_kv = reinterpret_cast<int const*>(inputs[3]);

    T* context_buf_ = (T*) (outputs[0]);

    size_t const fmha_scheduler_counter = mEnableContextFMHA ? sizeof(uint32_t) : 0;

    int8_t* workspace_byte_ptr = reinterpret_cast<int8_t*>(workspace);

    uint32_t* fmha_tile_counter_ptr
        = reinterpret_cast<uint32_t*>(tc::nextWorkspacePtr(workspace_byte_ptr, fmha_scheduler_counter));

    // FMHA doesn't apply to MHA with relative attention bias, i.e. softmax(QK + bias) * V
    // We update mEnableContextFMHA in constructor to check this condition
    assert(mEnableContextFMHA && "mEnableContextFMHA is false!");

    // Construct the fmha params for running kernels.
    MHARunnerParams fmhaParams{};
    fmhaParams.b = request_batch_size;
    fmhaParams.qSeqLen = q_seq_len;
    fmhaParams.kvSeqLen = kv_seq_len;
    fmhaParams.totalQSeqLen = batch_size * q_seq_len;
    fmhaParams.totalKvSeqLen = batch_size * kv_seq_len;
    // Device buffer pointers.
    fmhaParams.qkvPtr = nullptr;
    fmhaParams.qPtr = query;
    fmhaParams.kvPtr = kv_packed;
    fmhaParams.outputPtr = context_buf_;
    fmhaParams.cuQSeqLenPtr = cu_seqlens_q;
    // fmhaParams.cuQSeqLenPtr = nullptr;
    fmhaParams.cuKvSeqLenPtr = cu_seqlens_kv;
    // fmhaParams.cuKvSeqLenPtr = nullptr;
    fmhaParams.tileCounterPtr = fmha_tile_counter_ptr;
    fmhaParams.stream = stream;

    // Run the fmha kernel.
    mFMHARunner->run(fmhaParams);
    sync_check_cuda_error(stream);
    return 0;
}

template int WanAttentionPlugin::enqueueImpl<half>(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream);

template int WanAttentionPlugin::enqueueImpl<float>(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream);

#ifdef ENABLE_BF16
template int WanAttentionPlugin::enqueueImpl<__nv_bfloat16>(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream);
#endif

int WanAttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
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
nvinfer1::DataType WanAttentionPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

char const* WanAttentionPlugin::getPluginType() const noexcept
{
    return WAN_ATTENTION_PLUGIN_NAME;
}

char const* WanAttentionPlugin::getPluginVersion() const noexcept
{
    return WAN_ATTENTION_PLUGIN_VERSION;
}

int WanAttentionPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int WanAttentionPlugin::initialize() noexcept
{
    if (mEnableContextFMHA)
    {
        // Pre-checked during constructing.
        Data_type data_type;
        if (mType == DataType::kHALF)
        {
            data_type = DATA_TYPE_FP16;
        }
        else if (mType == DataType::kBF16)
        {
            data_type = DATA_TYPE_BF16;
        }
        else
        {
            TLLM_CHECK_WITH_INFO(false, "WANAttentionPlugin received wrong data type.");
        }

        // Construct the fmha runner.
        MHARunnerFixedParams fmhaParams{};
        fmhaParams.dataType = data_type;
        fmhaParams.dataTypeKv = data_type;
        fmhaParams.dataTypeOut = data_type;
        fmhaParams.forceFp32Acc = mFMHAForceFP32Acc;
        fmhaParams.attentionMaskType = ContextAttentionMaskType::PADDING;
        fmhaParams.isSPadded = false;
        fmhaParams.numQHeads = mNumHeads;
        fmhaParams.numKvHeads = mNumHeads;
        fmhaParams.headSize = mHeadSize;
        fmhaParams.qScaling = mQScaling;
        fmhaParams.attentionInputLayout = AttentionInputLayout::Q_CONTIGUOUS_KV;
        fmhaParams.numTokensPerBlock = 128;
        fmhaParams.headSizeV = mHeadSize;

        // Load kernels from the pre-compiled cubins.
        mFMHARunner.reset(new FusedMHARunnerV2(fmhaParams));

        // Fall back to unfused MHA kernels if not supported.
        mEnableContextFMHA = mFMHARunner->isFmhaSupported();
        printf("\n%s\n", fmhaParams.convertToStrOutput().c_str());
        printf("\nmEnableContextFMHA: %s\n", mEnableContextFMHA ? "true" : "false");
    }
    return 0;
}

void WanAttentionPlugin::destroy() noexcept
{
    delete this;
}

size_t WanAttentionPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(mQScaling) + sizeof(mQKHalfAccum) + sizeof(mEnableContextFMHA)
        + sizeof(mFMHAForceFP32Acc) + sizeof(mType);
}

void WanAttentionPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mNumHeads);
    write(d, mHeadSize);
    write(d, mQScaling);
    write(d, mQKHalfAccum);
    write(d, mEnableContextFMHA);
    write(d, mFMHAForceFP32Acc);
    write(d, mType);
    TLLM_CHECK(d == a + getSerializationSize());
}

void WanAttentionPlugin::terminate() noexcept {}

///////////////

WanAttentionPluginCreator::WanAttentionPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();

    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("q_scaling", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("context_fmha_type", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* WanAttentionPluginCreator::getPluginName() const noexcept
{
    return WAN_ATTENTION_PLUGIN_NAME;
}

char const* WanAttentionPluginCreator::getPluginVersion() const noexcept
{
    return WAN_ATTENTION_PLUGIN_VERSION;
}

PluginFieldCollection const* WanAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* WanAttentionPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
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
        auto* obj = new WanAttentionPlugin(num_heads, head_size, q_scaling, context_fmha_type, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* WanAttentionPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call BertAttentionPlugin::destroy()
    try
    {
        auto* obj = new WanAttentionPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
