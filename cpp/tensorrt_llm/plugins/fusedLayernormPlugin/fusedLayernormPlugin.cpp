/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "fusedLayernormPlugin.h"
#include "pluginUtils.h"
#include "tensorrt_llm/common/assert.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::FusedLayernormPluginCreator;
using tensorrt_llm::plugins::FusedLayernormPlugin;

static char const* FUSED_LAYERNORM_PLUGIN_VERSION{"1"};
static char const* FUSED_LAYERNORM_PLUGIN_NAME{"FusedLayernorm"};
PluginFieldCollection FusedLayernormPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> FusedLayernormPluginCreator::mPluginAttributes;

FusedLayernormPlugin::FusedLayernormPlugin(float eps, bool needFP32Output, bool needQuantize, nvinfer1::DataType type)
    : mEps(eps)
    , mNeedFP32Output(needFP32Output)
    , mNeedQuantize(needQuantize)
    , mType(type)
{
}

// Parameterized constructor
FusedLayernormPlugin::FusedLayernormPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mEps);
    read(d, mNeedFP32Output);
    read(d, mNeedQuantize);
    read(d, mType);
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* FusedLayernormPlugin::clone() const noexcept
{
    auto* plugin = new FusedLayernormPlugin(mEps, mNeedFP32Output, mNeedQuantize, mType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs FusedLayernormPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Dim should be the same as input hidden states
    if (!mNeedQuantize)
    {
        return inputs[0];
    }

    if (outputIndex == 1) // un-normed output fp16
    {
        return inputs[0];
    }
    if (outputIndex == 0) // quantized normed output
    {
        // Quantized output with int64_t data type (16 FP4 values per element).
        DimsExprs ret;
        ret.nbDims = inputs[0].nbDims;
        for (int di = 0; di < ret.nbDims; ++di)
        {
            ret.d[di] = inputs[0].d[di];
        }
        return ret;
    }

    // Scaling Factors.
    try
    {
        TLLM_CHECK(outputIndex == 2);

        DimsExprs ret;
        ret.nbDims = inputs[0].nbDims;
        for (int di = 0; di < ret.nbDims; ++di)
        {
            ret.d[di] = inputs[0].d[di];
        }
        // Sequence dimension or token dimension.
        // Pad to multiple of 128.
        auto dimM
            = exprBuilder.operation(DimensionOperation::kCEIL_DIV, *ret.d[ret.nbDims - 2], *exprBuilder.constant(128));
        ret.d[ret.nbDims - 2] = exprBuilder.operation(DimensionOperation::kPROD, *dimM, *exprBuilder.constant(128));
        // Hidden size dimension.
        ret.d[ret.nbDims - 1]
            = exprBuilder.operation(DimensionOperation::kCEIL_DIV, *ret.d[ret.nbDims - 1], *exprBuilder.constant(16));
        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool FusedLayernormPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    int const totalPoses = 5 + 2 * static_cast<int>(mNeedQuantize);
    TLLM_CHECK(0 <= pos && pos < totalPoses);
    TLLM_CHECK(nbInputs == 3 + static_cast<int>(mNeedQuantize));
    if (pos < nbInputs)
    {
        switch (pos)
        {
        case 0:
        case 1:
        case 2: return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
        case 3: return (inOut[pos].type == nvinfer1::DataType::kFLOAT);
        }
    }
    if (pos == nbInputs) // Normed output
    {
        if (mNeedQuantize)
        {
            // fp4 quantized output -- fp4 padded tp int64
            return (inOut[pos].type == nvinfer1::DataType::kFP4) && (inOut[pos].format == TensorFormat::kLINEAR);
        }
        return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else if (pos == nbInputs + 1) // Un-normed output
    {
        return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    // fp4 act_per_block_scale -- fp8 padded to int32
    return (inOut[pos].type == nvinfer1::DataType::kFP8) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void FusedLayernormPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t FusedLayernormPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return sizeof(WarpSpecializedCounters);
}

int FusedLayernormPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // inputs
    //     input [M(*), N]
    //     residual [M(*), N]
    //     weight [N, ]
    //     scale [1, ] - if needQuantize
    // outputs
    //     output [M(*), N] - fp4 padded to int64 / fp16
    //     un-normed output [M(*), N] - fp16
    //     act_per_block_scale - fp8 padded to int32 - if needQuantize

#define SETUP_PARAM                                                                                                    \
    Param param;                                                                                                       \
    int64_t m64 = 1;                                                                                                   \
    for (int i = 0; i < inputDesc[0].dims.nbDims - 1; ++i)                                                             \
    {                                                                                                                  \
        m64 *= inputDesc[0].dims.d[i];                                                                                 \
    }                                                                                                                  \
    int const m = TLLM_INT32_CAST(m64);                                                                                \
    int const n = TLLM_INT32_CAST(inputDesc[2].dims.d[0]);                                                             \
    param.m = m;                                                                                                       \
    param.n = n;                                                                                                       \
    param.layernorm_eps = mEps;                                                                                        \
    param.input = const_cast<Input*>(reinterpret_cast<Input const*>(inputs[0]));                                       \
    param.residual = const_cast<Input*>(reinterpret_cast<Input const*>(inputs[1]));                                    \
    param.gamma = const_cast<Input*>(reinterpret_cast<Input const*>(inputs[2]));                                       \
    if (mNeedQuantize)                                                                                                 \
    {                                                                                                                  \
        param.sf_scale = const_cast<float*>(reinterpret_cast<float const*>(inputs[3]));                                \
    }                                                                                                                  \
    param.counters = reinterpret_cast<WarpSpecializedCounters*>(workspace);                                            \
    param.stream = stream;                                                                                             \
    param.normed_output = reinterpret_cast<uint32_t*>(outputs[0]);                                                     \
    param.output = reinterpret_cast<Input*>(outputs[1]);                                                               \
    param.sf_out = reinterpret_cast<uint32_t*>(outputs[2]);

#define CLEANUP_AND_INVOKE                                                                                             \
    TLLM_CUDA_CHECK(cudaMemsetAsync(workspace, 0, sizeof(WarpSpecializedCounters), stream));                           \
    invokeWSLayerNorm(param, true, num_sms);

    int num_sms = tensorrt_llm::common::getMultiProcessorCount();

    if (mType == DataType::kHALF)
    {
        using Input = half;
        using Param = WarpSpecializedParam<GeneralFP4AddBiasResidualPreLayerNormParam<Input>>;
        SETUP_PARAM
        CLEANUP_AND_INVOKE
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16)
    {
        using Input = __nv_bfloat16;
        using Param = WarpSpecializedParam<GeneralFP4AddBiasResidualPreLayerNormParam<Input>>;
        SETUP_PARAM
        CLEANUP_AND_INVOKE
    }
#endif
    else
    {
        TLLM_LOG_ERROR("Unsupported data type");
        return 1;
    }
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType FusedLayernormPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    // assert((mNeedFP32Output && index < 3) || (!mNeedFP32Output && index < 2));
    assert((mNeedQuantize && index < 3) || (!mNeedQuantize && index < 2));
    if (index == 0)
    {
        // Output 0 quantized output of layernorm - fp4 padded to int64
        if (mNeedQuantize)
        {
            return nvinfer1::DataType::kFP4;
        }
        return mType;
    }
    else if (index == 1)
    {
        // Output 1 un-normed output
        return mType;
    }
    // Output 2 act_per_block_scale - fp8 padded to int32
    return nvinfer1::DataType::kFP8;
}

// IPluginV2 Methods

char const* FusedLayernormPlugin::getPluginType() const noexcept
{
    return FUSED_LAYERNORM_PLUGIN_NAME;
}

char const* FusedLayernormPlugin::getPluginVersion() const noexcept
{
    return FUSED_LAYERNORM_PLUGIN_VERSION;
}

int FusedLayernormPlugin::getNbOutputs() const noexcept
{
    return 2 + static_cast<int>(mNeedQuantize);
}

int FusedLayernormPlugin::initialize() noexcept
{
    return 0;
}

void FusedLayernormPlugin::terminate() noexcept {}

size_t FusedLayernormPlugin::getSerializationSize() const noexcept
{
    return sizeof(mEps) + sizeof(mNeedFP32Output) + sizeof(mNeedQuantize) + sizeof(mType);
}

void FusedLayernormPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mEps);
    write(d, mNeedFP32Output);
    write(d, mNeedQuantize);
    write(d, mType);
    TLLM_CHECK(d == a + getSerializationSize());
}

void FusedLayernormPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

FusedLayernormPluginCreator::FusedLayernormPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("need_fp32_output", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("need_quantize", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* FusedLayernormPluginCreator::getPluginName() const noexcept
{
    return FUSED_LAYERNORM_PLUGIN_NAME;
}

char const* FusedLayernormPluginCreator::getPluginVersion() const noexcept
{
    return FUSED_LAYERNORM_PLUGIN_VERSION;
}

PluginFieldCollection const* FusedLayernormPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* FusedLayernormPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    float eps{};
    nvinfer1::DataType type{};
    bool needFP32Output{};
    bool needQuantize{};
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "eps"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            eps = static_cast<float>(*(static_cast<float const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "need_fp32_output"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            needFP32Output = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "need_quantize"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            needQuantize = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new FusedLayernormPlugin(eps, needFP32Output, needQuantize, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* FusedLayernormPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call FusedLayernormPlugin::destroy()
    try
    {
        auto* obj = new FusedLayernormPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
