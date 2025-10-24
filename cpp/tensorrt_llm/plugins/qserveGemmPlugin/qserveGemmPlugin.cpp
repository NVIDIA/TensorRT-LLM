/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
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

#include "qserveGemmPlugin.h"
#include "tensorrt_llm/kernels/qserveGemm.h"
#include <cassert>
#include <numeric>

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::QServeGemmPluginCreator;
using tensorrt_llm::plugins::QServeGemmPlugin;
using tensorrt_llm::plugins::read;
using tensorrt_llm::plugins::write;
using namespace tensorrt_llm::kernels::qserve;

static char const* QSERVE_GEMM_PLUGIN_VERSION{"1"};
static char const* QSERVE_GEMM_PLUGIN_NAME{"QServeGemm"};

PluginFieldCollection QServeGemmPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> QServeGemmPluginCreator::mPluginAttributes;

namespace tensorrt_llm::plugins
{

QServeGemmPlugin::QServeGemmPlugin(
    // QuantMode quantMode,
    nvinfer1::DataType dtype, int groupSize)
{
    init(dtype, groupSize);
}

QServeGemmPlugin::QServeGemmPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;

    nvinfer1::DataType type;
    unsigned int quantMode;
    int groupSize;

    read(d, quantMode);
    read(d, type);
    read(d, groupSize);

    read(d, mDims);

    // mQuantMode = QuantMode(quantMode);

    init(type, groupSize);

    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

void QServeGemmPlugin::init(nvinfer1::DataType dtype, int groupSize)
{
    if (groupSize <= 0)
        groupSize = -1; // Per-channel
    mGroupSize = groupSize;
    mType = dtype;
    mRunner = std::make_shared<QServeGemmRunner>();
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QServeGemmPlugin::clone() const noexcept
{
    auto* plugin = new QServeGemmPlugin(*this);
    return plugin;
}

nvinfer1::DimsExprs QServeGemmPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        TLLM_CHECK(nbInputs == 6);
        TLLM_CHECK(outputIndex == 0);
        int const nbDimsA = inputs[0].nbDims;
        TLLM_CHECK(nbDimsA >= 2);
        DimsExprs ret;
        ret.nbDims = nbDimsA;
        for (int ii = 0; ii < nbDimsA - 1; ++ii)
        {
            ret.d[ii] = inputs[0].d[ii];
        }
        ret.d[nbDimsA - 1] = inputs[1].d[0];
        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool QServeGemmPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (mGroupSize != -1)
    { // Per-group
        switch (pos)
        {
        case 0:
            // activation
            return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
        case 1:
            // uint4 weights packed in int8
            return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
        case 2:
            // int8 weight s2_zeros
            return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
        case 3:
            // int8 weight s2_scales
            return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
        case 4:
            // fp16 weight s1_scales
            return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
        case 5:
            // fp16 activation scales
            return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
        case 6:
            // fp16 output activation
            return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
        default: return false;
        }
    }

    else
    { // Per-channel
        switch (pos)
        {
        case 0:
            // activation
            return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
        case 1:
            // uint4 weights packed in int8
            return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
        case 2:
            // fp16 s1_scales
            return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
        case 3:
            // fp16 s1_szeros
            return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
        case 4:
            // fp16 act_sums
            return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
        case 5:
            // fp16 act_scales
            return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
        case 6:
            // fp16 output activation
            return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
        default: return false;
        }
    }
}

void QServeGemmPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    auto const minM = std::accumulate(in[0].min.d, in[0].min.d + in[0].min.nbDims - 1, 1, std::multiplies<int>());
    auto const maxM = std::accumulate(in[0].max.d, in[0].max.d + in[0].max.nbDims - 1, 1, std::multiplies<int>());

    int const maxK = in[0].max.d[in[0].max.nbDims - 1];
    int const maxN = in[1].max.d[0];
    int const minK = in[0].min.d[in[0].min.nbDims - 1];
    int const minN = in[1].min.d[0];

    TLLM_CHECK_WITH_INFO(minN == maxN, "Variable out channels is not allowed");
    TLLM_CHECK_WITH_INFO(minK == maxK, "Variable in channels is not allowed");

    if (!mDims.isInitialized())
    {
        mDims = {minM, maxM, maxN, maxK};
    }
    m_workspaceMaxSize = mRunner->getWorkspaceSize(maxM, maxN, maxK);
}

size_t QServeGemmPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return m_workspaceMaxSize;
}

int QServeGemmPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // inputs

    // Per group:
    //     activation       [M, K]                  int8_t      Quantized sint8 activations
    //     weights          [N, K/2]                int8_t      Quantized uint4 weights  (packed as int8_t)
    //     s2_zeros         [K/group_size, N]       int8_t      Level-2 sint8 scaled zeros of weights
    //     s2_scales        [K/group_size, N]       int8_t      Level-2 sint8 scales of weights
    //     s1_scales        [N]                     half        Level-1 fp16 scales of weights
    //     act_scales       [M]                     half        Scales of activations

    // Per channel:
    //     activation       [M, K]                  int8_t      Quantized sint8 activations
    //     weights          [N, K/2]                int8_t      Quantized uint4 weights  (packed as int8_t)
    //     s1_scales        [N]                     half        Level-1 scales of weights
    //     s1_szeros        [N]                     half        Level-1 scaled zeros of weights
    //     act_sums         [M]                     half        Per-token sums of activations
    //     act_scales       [M]                     half        Scales of activations

    // outputs
    //     mat              [M(*), N]               half

    int64_t m64 = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m64 *= inputDesc[0].dims.d[ii];
    }
    int const m = TLLM_INT32_CAST(m64);
    int const n = TLLM_INT32_CAST(inputDesc[1].dims.d[0]);
    int const k = TLLM_INT32_CAST(inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1]);

    // TODO: Implement optimized kernels if (m <= 4)

    if (mGroupSize != -1)
    {
        ParamsPerGroup params = {reinterpret_cast<int8_t const*>(inputs[0]), // A
            reinterpret_cast<int8_t const*>(inputs[1]),                      // B
            reinterpret_cast<int8_t const*>(inputs[2]),                      // s2_zeros
            reinterpret_cast<int8_t const*>(inputs[3]),                      // s2_scales
            reinterpret_cast<half const*>(inputs[4]),                        // s1_scales
            reinterpret_cast<half const*>(inputs[5]),                        // act_scales
            reinterpret_cast<half*>(outputs[0]),                             // C
            m, n, k};
        mRunner->gemmPerGroup(params, stream);
    }
    else
    {
        ParamsPerChannel params = {reinterpret_cast<int8_t const*>(inputs[0]), // A
            reinterpret_cast<int8_t const*>(inputs[1]),                        // B
            reinterpret_cast<half const*>(inputs[2]),                          // s1_scales
            reinterpret_cast<half const*>(inputs[3]),                          // s1_szeros
            reinterpret_cast<half const*>(inputs[4]),                          // act_sums
            reinterpret_cast<half const*>(inputs[5]),                          // act_scales
            reinterpret_cast<half*>(outputs[0]),                               // C
            m, n, k};
        mRunner->gemmPerChannel(params, stream);
    }

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType QServeGemmPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == 0);
    return mType;
}

// IPluginV2 Methods

char const* QServeGemmPlugin::getPluginType() const noexcept
{
    return QSERVE_GEMM_PLUGIN_NAME;
}

char const* QServeGemmPlugin::getPluginVersion() const noexcept
{
    return QSERVE_GEMM_PLUGIN_VERSION;
}

int QServeGemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int QServeGemmPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}

void QServeGemmPlugin::terminate() noexcept {}

size_t QServeGemmPlugin::getSerializationSize() const noexcept
{
    return sizeof(mQuantMode) + // QuantMode
        sizeof(mType) +         // dtype
        sizeof(mGroupSize) +    // GroupSize
        sizeof(mDims);          // Dimensions
}

void QServeGemmPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mQuantMode.value());
    write(d, mType);
    write(d, mGroupSize);
    write(d, mDims);

    TLLM_CHECK(d == a + getSerializationSize());
}

void QServeGemmPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void QServeGemmPlugin::configGemm() {}

///////////////

QServeGemmPluginCreator::QServeGemmPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.push_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.push_back(PluginField("group_size", nullptr, PluginFieldType::kINT32));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* QServeGemmPluginCreator::getPluginName() const noexcept
{
    return QSERVE_GEMM_PLUGIN_NAME;
}

char const* QServeGemmPluginCreator::getPluginVersion() const noexcept
{
    return QSERVE_GEMM_PLUGIN_VERSION;
}

PluginFieldCollection const* QServeGemmPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* QServeGemmPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    // We do not use any fields for now.

    PluginField const* fields = fc->fields;

    // bool perTokenScaling, perChannelScaling;
    DataType dtype{};
    int group_size = -1;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            dtype = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
            // Only supports fp16 for now.
            assert(dtype == nvinfer1::DataType::kHALF);
        }
        else if (!strcmp(attrName, "group_size"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            group_size = *static_cast<int const*>(fields[i].data);
            // Currently only support per-channel or g128.
            assert(group_size == -1 || group_size == 128);
        }
    }
    try
    {
        // QServeGemmPluginCreator is unique and shared for an engine generation
        // Create plugin profiler with shared tactics map
        // auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler(/* inference */ false);
        // QuantMode quantMode = QuantMode::fromQuantAlgo("W4A8_QSERVE");
        auto* obj = new QServeGemmPlugin(dtype, group_size);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* QServeGemmPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call QServeGemmPlugin::destroy()
    try
    {
        // Create plugin profiler with private tactics map which is read from the serialized engine
        // auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler(/* inference */ true);
        auto* obj = new QServeGemmPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
} // namespace tensorrt_llm::plugins
