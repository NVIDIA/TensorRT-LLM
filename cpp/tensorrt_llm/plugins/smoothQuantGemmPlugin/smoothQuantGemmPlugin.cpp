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
#include "smoothQuantGemmPlugin.h"
#include <numeric>

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels::cutlass_kernels;
using tensorrt_llm::plugins::SmoothQuantGemmPluginCreator;
using tensorrt_llm::plugins::SmoothQuantGemmPlugin;
using tensorrt_llm::plugins::SmoothQuantGemmPluginProfiler;
using tensorrt_llm::plugins::read;
using tensorrt_llm::plugins::write;

static const char* SQ_GEMM_PLUGIN_VERSION{"1"};
static const char* SQ_GEMM_PLUGIN_NAME{"SmoothQuantGemm"};
PluginFieldCollection SmoothQuantGemmPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> SmoothQuantGemmPluginCreator::mPluginAttributes;

void SmoothQuantGemmPluginProfiler::runTactic(int m, int n, int k, const SmoothQuantGemmPluginProfiler::Config& tactic,
    char* workspace, const cudaStream_t& stream)
{
    int8_t* aTmp = reinterpret_cast<int8_t*>(workspace);
    int8_t* bTmp = nextWorkspacePtr(aTmp, m * k * sizeof(int8_t));
    void* cTmp = reinterpret_cast<void*>(nextWorkspacePtr(bTmp, n * k * sizeof(int8_t)));
    float* alphaRowTmp = reinterpret_cast<float*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(cTmp), m * n * (mType == nvinfer1::DataType::kFLOAT ? 4 : 2)));
    float* alphaColTmp
        = reinterpret_cast<float*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(alphaRowTmp), m * sizeof(float)));
    char* workspaceTmp
        = reinterpret_cast<char*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(alphaColTmp), n * sizeof(float)));

    const int wsSize = mRunner->getWorkspaceSize(m, n, k);

    mRunner->gemm(
        aTmp, bTmp, mQuantMode, alphaColTmp, alphaRowTmp, cTmp, m, n, k, tactic, workspaceTmp, wsSize, stream);
}

void SmoothQuantGemmPluginProfiler::computeTmpSize(int maxM, int n, int k)
{
    std::vector<size_t> workspaces = {
        maxM * k * sizeof(int8_t),                                  // A
        n * k * sizeof(int8_t),                                     // B
        maxM * n * (mType == nvinfer1::DataType::kFLOAT ? 4u : 2u), // C
        maxM * sizeof(float),                                       // alphaRow
        n * sizeof(float),                                          // alphaCol
        mRunner->getWorkspaceSize(maxM, n, k)                       // workspace
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<SmoothQuantGemmPluginProfiler::Config> SmoothQuantGemmPluginProfiler::getTactics(int m, int n, int k) const
{
    return mRunner->getConfigs();
}

SmoothQuantGemmPlugin::SmoothQuantGemmPlugin(
    QuantMode quantMode, nvinfer1::DataType type, const SmoothQuantGemmPlugin::PluginProfilerPtr& pluginProfiler)
    : mQuantMode(quantMode)
    , mPluginProfiler(pluginProfiler)
{
    init(type);
}

// Parameterized constructor
SmoothQuantGemmPlugin::SmoothQuantGemmPlugin(
    const void* data, size_t length, const SmoothQuantGemmPlugin::PluginProfilerPtr& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    bool perChannelScaling = false, perTokenScaling = false;
    nvinfer1::DataType type;
    unsigned int quantMode;
    read(d, quantMode);
    read(d, type);
    read(d, mDims);

    mQuantMode = QuantMode(quantMode);

    init(type);

    mPluginProfiler->deserialize(d, mDims, mGemmId);

    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

void SmoothQuantGemmPlugin::init(nvinfer1::DataType type)
{
    mType = type;
    if (mType == nvinfer1::DataType::kHALF)
    {
        m_sqGemmRunner = std::make_shared<CutlassInt8GemmRunner<half>>();
    }
    else if (mType == nvinfer1::DataType::kFLOAT)
    {
        m_sqGemmRunner = std::make_shared<CutlassInt8GemmRunner<float>>();
    }
    else if (mType == nvinfer1::DataType::kINT32)
    {
        m_sqGemmRunner = std::make_shared<CutlassInt8GemmRunner<int32_t>>();
    }
    else
    {
        // TODO: add bf16 support
        TLLM_THROW("Support for bf16 is missing");
    }

    mPluginProfiler->setQuantMode(mQuantMode);

    mGemmId = GemmIdCore(mDims.n, mDims.k, mType);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* SmoothQuantGemmPlugin::clone() const noexcept
{
    auto* plugin = new SmoothQuantGemmPlugin(*this);
    return plugin;
}

nvinfer1::DimsExprs SmoothQuantGemmPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        TLLM_CHECK(nbInputs == 4);
        TLLM_CHECK(outputIndex == 0);
        const int nbDimsA = inputs[0].nbDims;
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
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool SmoothQuantGemmPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    switch (pos)
    {
    case 0:
        // activation
        return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
    case 1:
        // weights
        // Weights stored in checkpoint must have int8 type
        return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
    case 2:
        // scales channels
    case 3:
        // scales tokens
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    case 4:
        // out
        return inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
    default:
        // Never should be here
        assert(false);
        return false;
    }
}

void SmoothQuantGemmPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    const auto minM = std::accumulate(in[0].min.d, in[0].min.d + in[0].min.nbDims - 1, 1, std::multiplies<int>());
    const auto maxM = std::accumulate(in[0].max.d, in[0].max.d + in[0].max.nbDims - 1, 1, std::multiplies<int>());

    const int maxK = in[0].max.d[in[0].max.nbDims - 1];
    const int maxN = in[1].max.d[0];
    const int minK = in[0].min.d[in[0].min.nbDims - 1];
    const int minN = in[1].min.d[0];

    TLLM_CHECK_WITH_INFO(minN == maxN, "Variable out channels is not allowed");
    TLLM_CHECK_WITH_INFO(minK == maxK, "Variable in channels is not allowed");

    if (!mDims.isInitialized())
    {
        mDims = {minM, maxM, maxN, maxK};
    }
    mGemmId = {maxN, maxK, mType};

    m_workspaceMaxSize = m_sqGemmRunner->getWorkspaceSize(maxM, maxN, maxK);
}

size_t SmoothQuantGemmPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return m_workspaceMaxSize;
}

int SmoothQuantGemmPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // inputs
    //     mat1           [M(*), K]
    //     mat2           [N, K]
    //     scale_tokens   [M, 1] if has_per_token_scaling else [1, 1]
    //     scale_channels [1, N] if has_per_channel_scaling else [1, 1]
    // outputs
    //     mat [M(*), N]
    int m = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m *= inputDesc[0].dims.d[ii];
    }
    const int n = inputDesc[1].dims.d[0];
    const int k = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
    const int wsSize = m_sqGemmRunner->getWorkspaceSize(m, n, k);

    const auto& bestTactic = mPluginProfiler->getBestConfig(m, mGemmId);
    TLLM_CHECK_WITH_INFO(bestTactic, "No valid SQ GEMM tactic");
    m_sqGemmRunner->gemm(reinterpret_cast<const int8_t*>(inputs[0]), reinterpret_cast<const int8_t*>(inputs[1]),
        mQuantMode, reinterpret_cast<const float*>(inputs[3]), reinterpret_cast<const float*>(inputs[2]),
        reinterpret_cast<void*>(outputs[0]), m, n, k, *bestTactic, reinterpret_cast<char*>(workspace), wsSize, stream);

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType SmoothQuantGemmPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == 0);
    return mType;
}

// IPluginV2 Methods

const char* SmoothQuantGemmPlugin::getPluginType() const noexcept
{
    return SQ_GEMM_PLUGIN_NAME;
}

const char* SmoothQuantGemmPlugin::getPluginVersion() const noexcept
{
    return SQ_GEMM_PLUGIN_VERSION;
}

int SmoothQuantGemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int SmoothQuantGemmPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}

void SmoothQuantGemmPlugin::terminate() noexcept {}

size_t SmoothQuantGemmPlugin::getSerializationSize() const noexcept
{
    return sizeof(unsigned int) +                       // QuantMode
        sizeof(nvinfer1::DataType) +                    // dtype
        sizeof(mDims) +                                 // Dimensions
        mPluginProfiler->getSerializationSize(mGemmId); // selected tactics container size
}

void SmoothQuantGemmPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mQuantMode.value());
    write(d, mType);
    write(d, mDims);

    mPluginProfiler->serialize(d, mGemmId);
    assert(d == a + getSerializationSize());
}

void SmoothQuantGemmPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void SmoothQuantGemmPlugin::configGemm()
{
    mPluginProfiler->profileTactics(m_sqGemmRunner, mType, mDims, mGemmId);
}

///////////////

SmoothQuantGemmPluginCreator::SmoothQuantGemmPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("has_per_channel_scaling", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("has_per_token_scaling", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* SmoothQuantGemmPluginCreator::getPluginName() const noexcept
{
    return SQ_GEMM_PLUGIN_NAME;
}

const char* SmoothQuantGemmPluginCreator::getPluginVersion() const noexcept
{
    return SQ_GEMM_PLUGIN_VERSION;
}

const PluginFieldCollection* SmoothQuantGemmPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* SmoothQuantGemmPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    bool perTokenScaling, perChannelScaling;
    nvinfer1::DataType type;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "has_per_channel_scaling"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            perChannelScaling = static_cast<bool>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "has_per_token_scaling"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            perTokenScaling = static_cast<bool>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
    }
    try
    {
        // SmoothQuantGemmPluginCreator is unique and shared for an engine generation
        // Create plugin profiler with shared tactics map
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler(/* inference */ false);
        QuantMode quantMode = QuantMode::fromDescription(true, true, perTokenScaling, perChannelScaling);
        auto* obj = new SmoothQuantGemmPlugin(quantMode, type, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* SmoothQuantGemmPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call SmoothQuantGemmPlugin::destroy()
    try
    {
        // Create plugin profiler with private tactics map which is read from the serialized engine
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler(/* inference */ true);
        auto* obj = new SmoothQuantGemmPlugin(serialData, serialLength, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
