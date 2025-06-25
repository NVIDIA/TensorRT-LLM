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

#include "gemmSwigluPlugin.h"
#include "cutlass_extensions/gemm_configs.h"

#include <NvInferRuntimeBase.h>
#include <numeric>

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels::cutlass_kernels;
using tensorrt_llm::plugins::GemmSwigluPluginCreator;
using tensorrt_llm::plugins::GemmSwigluPlugin;
using tensorrt_llm::plugins::GemmSwigluPluginProfiler;
using tensorrt_llm::plugins::read;
using tensorrt_llm::plugins::write;

static char const* GEMM_SWIGLU_PLUGIN_VERSION{"1"};
static char const* GEMM_SWIGLU_PLUGIN_NAME{"GemmSwiglu"};
PluginFieldCollection GemmSwigluPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> GemmSwigluPluginCreator::mPluginAttributes;

size_t GemmSwigluPluginProfiler::getBytePerElement(nvinfer1::DataType type)
{
    size_t bpe;
    if (type == nvinfer1::DataType::kHALF || type == nvinfer1::DataType::kBF16)
    {
        bpe = 2;
    }
    else if (type == nvinfer1::DataType::kINT8 || type == nvinfer1::DataType::kFP8)
    {
        bpe = 1;
    }
    else
    {
        TLLM_THROW("Not recognized/implemented");
    }
    return bpe;
}

void GemmSwigluPluginProfiler::setQuantMode(tensorrt_llm::common::QuantMode const& quantMode)
{
    mQuantMode = quantMode;
}

void GemmSwigluPluginProfiler::runTactic(
    int m, int n, int k, GemmSwigluPluginProfiler::Config const& tactic, char* workspace, cudaStream_t const& stream)
{
    size_t bpe = getBytePerElement(mType);

    // Workspace size required by gemm runner
    // NB: this function will throw exception when selected tactic exceeds SMEM, which is then
    // caught by gemmPluginProfiler and it will register this tactic as invalid
    size_t wsSizeRunner = mRunner->getWorkspaceSize(m, n, k);

    // Workspace size required by profiling
    size_t wsByteOffset = 0;
    int8_t* wsBytePointer = reinterpret_cast<int8_t*>(workspace);
    void* aTmp = reinterpret_cast<void*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, m * k * bpe));
    void* bTmp = reinterpret_cast<void*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, n * k * bpe));
    void* cTmp = reinterpret_cast<void*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, 1 * n * bpe));
    void* dTmp = reinterpret_cast<void*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, m * (n / 2) * bpe));
    char* workspaceTmp = reinterpret_cast<char*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, wsSizeRunner));

    // Run profiling
    mRunner->gemm(
        dTmp, aTmp, bTmp, cTmp, mQuantMode, m, n, k, 1.0, 1.0, 1.0, tactic, workspaceTmp, wsSizeRunner, stream);
}

int GemmSwigluPluginProfiler::getMaxProfileM() const
{
    return 32768;
}

void GemmSwigluPluginProfiler::computeTmpSize(size_t maxM, size_t n, size_t k)
{
    std::vector<size_t> workspaces = {
        maxM * k * getBytePerElement(mType),       // A
        n * k * getBytePerElement(mType),          // B
        1 * n * getBytePerElement(mType),          // C_bias
        maxM * (n / 2) * getBytePerElement(mType), // D
        mRunner->getWorkspaceSize(maxM, n, k)      // workspace
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<GemmSwigluPluginProfiler::Config> GemmSwigluPluginProfiler::getTactics(int m, int n, int k) const
{
    return mRunner->getConfigs();
}

GemmSwigluPlugin::GemmSwigluPlugin(QuantMode quantMode, nvinfer1::DataType type, bool hasBias, float scale_d0,
    float scale_d1, float scale_output, GemmSwigluPlugin::PluginProfilerPtr const& pluginProfiler)
    : mQuantMode(quantMode)
    , mPluginProfiler(pluginProfiler)
    , mHasBias(hasBias)
    , mScaleD0(scale_d0)
    , mScaleD1(scale_d1)
    , mScaleOutput(scale_output)
{
    init(type);
}

// Parameterized constructor
GemmSwigluPlugin::GemmSwigluPlugin(
    void const* data, size_t length, GemmSwigluPlugin::PluginProfilerPtr const& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    nvinfer1::DataType type;
    unsigned int quantMode;
    read(d, quantMode);
    read(d, type);
    read(d, mHasBias);
    read(d, mScaleD0);
    read(d, mScaleD1);
    read(d, mScaleOutput);
    read(d, mDims);

    mQuantMode = QuantMode(quantMode);

    init(type);

    mPluginProfiler->deserialize(d, mDims, mGemmId);

    TLLM_CHECK(d == a + length);
}

void GemmSwigluPlugin::init(nvinfer1::DataType type)
{
    mType = type;
    if (mType == nvinfer1::DataType::kFP8)
    {
        mGemmRunner = std::make_shared<CutlassFusedGatedGemmRunner<__nv_fp8_e4m3>>();
    }
    else
    {
        TLLM_THROW("Gemm Swiglu plugin only supports fp8 now");
    }

    mPluginProfiler->setQuantMode(mQuantMode);

    mGemmId = GemmIdCore(mDims.n, mDims.k, mType);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* GemmSwigluPlugin::clone() const noexcept
{
    auto* plugin = new GemmSwigluPlugin(*this);
    return plugin;
}

nvinfer1::DimsExprs GemmSwigluPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        TLLM_CHECK(nbInputs == 3);
        TLLM_CHECK(outputIndex == 0);
        int const nbDimsA = inputs[0].nbDims;
        TLLM_CHECK(nbDimsA >= 2);
        DimsExprs ret;
        ret.nbDims = nbDimsA;
        for (int ii = 0; ii < nbDimsA - 1; ++ii)
        {
            ret.d[ii] = inputs[0].d[ii];
        }
        ret.d[nbDimsA - 1] = exprBuilder.constant(inputs[1].d[1]->getConstantValue() / 2);
        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool GemmSwigluPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    switch (pos)
    {
    case 0:
        // activation
        return inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
    case 1:
        // weights
        return inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
    case 2:
        // bias
        return inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
    case 3:
        // out
        return inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
    default:
        // Never should be here
        TLLM_CHECK(false);
        return false;
    }
}

void GemmSwigluPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    auto const minM = std::accumulate(in[0].min.d, in[0].min.d + in[0].min.nbDims - 1, 1, std::multiplies<int>());
    auto const maxM = std::accumulate(in[0].max.d, in[0].max.d + in[0].max.nbDims - 1, 1, std::multiplies<int>());

    int const maxK = in[0].max.d[in[0].max.nbDims - 1];
    int const maxN = in[1].max.d[1];
    int const minK = in[0].min.d[in[0].min.nbDims - 1];
    int const minN = in[1].min.d[1];

    TLLM_CHECK_WITH_INFO(minN == maxN, "Variable out channels is not allowed");
    TLLM_CHECK_WITH_INFO(minK == maxK, "Variable in channels is not allowed");

    if (!mDims.isInitialized())
    {
        mDims = {minM, maxM, maxN, maxK};
    }
    mGemmId = {maxN, maxK, mType};

    mWorkspaceMaxSize = mGemmRunner->getWorkspaceSize(maxM, maxN, maxK);
}

size_t GemmSwigluPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return mWorkspaceMaxSize;
}

int GemmSwigluPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // inputs
    //     mat1           [M(*), K]
    //     mat2           [K, N]
    //     bias           [1, N]
    // outputs
    //     mat [M(*), N / 2]
    int m = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m *= inputDesc[0].dims.d[ii];
    }
    int const n = inputDesc[1].dims.d[1];
    int const k = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
    size_t const wsSize = mGemmRunner->getWorkspaceSize(m, n, k);

    auto const bestTactic = mPluginProfiler->getBestConfig(m, mGemmId);
    TLLM_CHECK_WITH_INFO(bestTactic, "No valid GEMM tactic");
    mGemmRunner->gemm(outputs[0], inputs[0], inputs[1], inputs[2], mQuantMode, m, n, k, mScaleD0, mScaleD1,
        mScaleOutput, *bestTactic, reinterpret_cast<char*>(workspace), wsSize, stream);

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType GemmSwigluPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == 0);
    return mType;
}

// IPluginV2 Methods

char const* GemmSwigluPlugin::getPluginType() const noexcept
{
    return GEMM_SWIGLU_PLUGIN_NAME;
}

char const* GemmSwigluPlugin::getPluginVersion() const noexcept
{
    return GEMM_SWIGLU_PLUGIN_VERSION;
}

int GemmSwigluPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int GemmSwigluPlugin::initialize() noexcept
{
    configGemm(); // gemm profiler in action
    return 0;
}

void GemmSwigluPlugin::terminate() noexcept {}

size_t GemmSwigluPlugin::getSerializationSize() const noexcept
{
    return sizeof(unsigned int) +                       // QuantMode
        sizeof(nvinfer1::DataType) +                    // dtype
        sizeof(bool) +                                  // hasBias
        sizeof(float) * 3 +                             // scales
        sizeof(mDims) +                                 // Dimensions
        mPluginProfiler->getSerializationSize(mGemmId); // selected tactics container size
}

void GemmSwigluPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mQuantMode.value());
    write(d, mType);
    write(d, mHasBias);
    write(d, mScaleD0);
    write(d, mScaleD1);
    write(d, mScaleOutput);
    write(d, mDims);

    mPluginProfiler->serialize(d, mGemmId);
    TLLM_CHECK(d == a + getSerializationSize());
}

void GemmSwigluPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void GemmSwigluPlugin::configGemm()
{
    mPluginProfiler->profileTactics(mGemmRunner, mType, mDims, mGemmId);
}

///////////////

GemmSwigluPluginCreator::GemmSwigluPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("has_bias", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("scale_d0", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("scale_d1", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("scale_output", nullptr, PluginFieldType::kFLOAT32));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* GemmSwigluPluginCreator::getPluginName() const noexcept
{
    return GEMM_SWIGLU_PLUGIN_NAME;
}

char const* GemmSwigluPluginCreator::getPluginVersion() const noexcept
{
    return GEMM_SWIGLU_PLUGIN_VERSION;
}

PluginFieldCollection const* GemmSwigluPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* GemmSwigluPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    TLLM_CHECK(fc->nbFields == 5);
    nvinfer1::DataType type{};
    bool hasBias{};
    float scale_d0{};
    float scale_d1{};
    float scale_output{};
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "has_bias"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            hasBias = static_cast<bool>(*(static_cast<int8_t const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "scale_d0"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            scale_d0 = static_cast<float>(*(static_cast<float const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "scale_d1"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            scale_d1 = static_cast<float>(*(static_cast<float const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "scale_output"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            scale_output = static_cast<float>(*(static_cast<float const*>(fields[i].data)));
        }
    }
    try
    {
        // GemmSwigluPluginCreator is unique and shared for an engine generation
        // Create plugin profiler with shared tactics map
        auto pluginProfiler = mGemmPluginProfileManager.createGemmPluginProfiler(/* inference */ false);
        QuantMode quantMode = QuantMode{};
        auto* obj = new GemmSwigluPlugin(quantMode, type, hasBias, scale_d0, scale_d1, scale_output, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* GemmSwigluPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call GemmSwigluPlugin::destroy()
    try
    {
        // Create plugin profiler with private tactics map which is read from the serialized engine
        auto pluginProfiler = mGemmPluginProfileManager.createGemmPluginProfiler(/* inference */ true);
        auto* obj = new GemmSwigluPlugin(serialData, serialLength, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
