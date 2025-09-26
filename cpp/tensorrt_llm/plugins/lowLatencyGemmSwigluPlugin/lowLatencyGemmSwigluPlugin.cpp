
/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
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

#include "lowLatencyGemmSwigluPlugin.h"
#include "low_latency_gemm_swiglu.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/common/logger.h"
#include <NvInferRuntime.h>
#include <NvInferRuntimeBase.h>
#include <NvInferRuntimePlugin.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <optional>
#include <vector>

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels::internal_cutlass_kernels;
using tensorrt_llm::plugins::LowLatencyGemmSwigluPluginCreator;
using tensorrt_llm::plugins::LowLatencyGemmSwigluPlugin;
using tensorrt_llm::plugins::LowLatencyGemmSwigluPluginProfiler;
using tensorrt_llm::plugins::read;
using tensorrt_llm::plugins::write;

static char const* LOW_LATENCY_GEMM_SWIGLU_PLUGIN_VERSION{"1"};
static char const* LOW_LATENCY_GEMM_SWIGLU_PLUGIN_NAME{"LowLatencyGemmSwiglu"};

PluginFieldCollection LowLatencyGemmSwigluPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> LowLatencyGemmSwigluPluginCreator::mPluginAttributes;

using FP8Type = __nv_fp8_e4m3;

static std::optional<float> getFloatEnv(char const* name)
{
    char const* const env = std::getenv(name);
    if (env == nullptr)
    {
        return std::nullopt;
    }
    try
    {
        float value = std::stof(env);
        return {value};
    }
    catch (std::invalid_argument const& e)
    {
        return std::nullopt;
    }
    catch (std::out_of_range const& e)
    {
        return std::nullopt;
    }
};

static size_t getBytePerElement(nvinfer1::DataType type)
{
    size_t bpe;
    if (type == nvinfer1::DataType::kFLOAT)
    {
        bpe = 4;
    }
    else if (type == nvinfer1::DataType::kHALF || type == nvinfer1::DataType::kBF16)
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

void LowLatencyGemmSwigluPluginProfiler::runTactic(int m, int n, int k,
    LowLatencyGemmSwigluPluginProfiler::Config const& tactic, char* workspace, cudaStream_t const& stream)
{

    float default_pdl_overlap_ratio = 0.5;
    float default_prefetch_ratio = -1.0;
    FP8Type* aTmp = reinterpret_cast<FP8Type*>(workspace);
    FP8Type* bTmp
        = reinterpret_cast<FP8Type*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(aTmp), m * k * sizeof(FP8Type)));
    void* dTmp = reinterpret_cast<void*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(bTmp), n * k * sizeof(FP8Type)));
    size_t workspaceSize = mRunner->getWorkspaceSize(m, n, k);
    char* workspaceTmp = reinterpret_cast<char*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(dTmp), (n / 2 * m * getBytePerElement(mType))));
    mRunner->gemm(aTmp, bTmp, 1.0f, 0.0f, 1.0f, 1.0f, nullptr, dTmp, m, n, k, default_pdl_overlap_ratio,
        default_prefetch_ratio, tactic, workspaceTmp, workspaceSize, stream);
}

int LowLatencyGemmSwigluPluginProfiler::getMaxProfileM() const
{
    return 32768;
}

void LowLatencyGemmSwigluPluginProfiler::computeTmpSize(size_t maxM, size_t n, size_t k)
{

    std::vector<size_t> workspaces = {maxM * k * sizeof(FP8Type), // A
        n * k * sizeof(FP8Type),                                  // B
        maxM * (n / 2) * getBytePerElement(mType),                // D
        mRunner->getWorkspaceSize(maxM, n, k)};                   // workspace

    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<LowLatencyGemmSwigluPluginProfiler::Config> LowLatencyGemmSwigluPluginProfiler::getTactics(
    int m, int n, int k) const
{
    return mRunner->getConfigs();
}

LowLatencyGemmSwigluPlugin::LowLatencyGemmSwigluPlugin(nvinfer1::DataType type, float scale_output, float scale_d0,
    float scale_d1, PluginProfilerPtr const& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
    , mScaleOutput(scale_output)
    , mScaleD0(scale_d0)
    , mScaleD1(scale_d1)
{
    init(type);
}

LowLatencyGemmSwigluPlugin::LowLatencyGemmSwigluPlugin(
    void const* data, size_t length, PluginProfilerPtr const& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{

    char const *d = reinterpret_cast<char const*>(data), *a = d;
    nvinfer1::DataType type;
    read(d, type);
    read(d, mScaleOutput);
    read(d, mScaleD0);
    read(d, mScaleD1);
    read(d, mDims);

    init(type);
    mPluginProfiler->deserialize(d, mDims, mGemmId);
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

void LowLatencyGemmSwigluPlugin::init(nvinfer1::DataType type)
{

    mType = type;

    if (mType == nvinfer1::DataType::kFP8)
    {
        mLowLatencyGemmSwigluRunner = std::make_shared<CutlassLowLatencyFp8GemmSwigluRunner<__nv_fp8_e4m3>>();
    }
    else
    {
        TLLM_THROW("Unsupported data type");
    }
    mGemmId = GemmIdCore(mDims.n, mDims.k, mType);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* LowLatencyGemmSwigluPlugin::clone() const noexcept
{
    auto* plugin = new LowLatencyGemmSwigluPlugin(*this);
    return plugin;
}

nvinfer1::DimsExprs LowLatencyGemmSwigluPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        TLLM_CHECK(nbInputs == 2);
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

bool LowLatencyGemmSwigluPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    switch (pos)
    {
    case 0:
        // activation
        return inOut[pos].type == nvinfer1::DataType::kFP8 && inOut[pos].format == TensorFormat::kLINEAR;
    case 1:
        // weights
        // Weights stored in checkpoint must have fp8 type
        return inOut[pos].type == nvinfer1::DataType::kFP8 && inOut[pos].format == TensorFormat::kLINEAR;
    case 2:
        // out
        return inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
    default:
        // Never should be here
        TLLM_CHECK(false);
        return false;
    }
}

void LowLatencyGemmSwigluPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
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

    mWorkspaceMaxSize = mLowLatencyGemmSwigluRunner->getWorkspaceSize(maxM, maxN, maxK);
}

size_t LowLatencyGemmSwigluPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return mWorkspaceMaxSize;
}

int LowLatencyGemmSwigluPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{

    // input0 activation [M,K] row-major
    // input1 weights [K, N] col-major
    // output0 [M,N / 2] row-major

    int64_t m64 = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m64 *= inputDesc[0].dims.d[ii];
    }
    int const m = TLLM_INT32_CAST(m64);
    int const n = TLLM_INT32_CAST(inputDesc[1].dims.d[1]);
    int const k = TLLM_INT32_CAST(inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1]);
    int const wsSize = mLowLatencyGemmSwigluRunner->getWorkspaceSize(m, n, k);
    auto const& bestTactic = mPluginProfiler->getBestConfig(m, mGemmId);
    TLLM_CHECK_WITH_INFO(bestTactic, "No valid Low Latency GEMM SWIGLU tactic");

    auto env_pdl_overlap_ratio = getFloatEnv("TRTLLM_PDL_OVERLAP_RATIO");
    auto env_prefetch_ratio = getFloatEnv("TRTLLM_PREFETCH_RATIO");
    auto valid_ratio = [](std::optional<float>& env_val, float default_val)
    {
        if (env_val.has_value())
        {
            TLLM_CHECK_WITH_INFO(env_val.value() <= 1.0f, "Valid ratio should be less than or equal to 1.0");
            return env_val.value();
        }
        return default_val;
    };
    float pdl_overlap_ratio = valid_ratio(env_pdl_overlap_ratio, /*default_val=*/0.5);
    float prefetch_ratio = valid_ratio(env_prefetch_ratio, /*default_val=*/-1.0);
    mLowLatencyGemmSwigluRunner->gemm(const_cast<FP8Type*>(reinterpret_cast<FP8Type const*>(inputs[0])),
        const_cast<FP8Type*>(reinterpret_cast<FP8Type const*>(inputs[1])), mScaleOutput, 0.0F, mScaleD0, mScaleD1,
        nullptr, outputs[0], m, n, k, pdl_overlap_ratio, prefetch_ratio, *bestTactic,
        reinterpret_cast<char*>(workspace), wsSize, stream);

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType LowLatencyGemmSwigluPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == 0);
    return mType;
}

// IPluginV2 Methods

char const* LowLatencyGemmSwigluPlugin::getPluginType() const noexcept
{
    return LOW_LATENCY_GEMM_SWIGLU_PLUGIN_NAME;
}

char const* LowLatencyGemmSwigluPlugin::getPluginVersion() const noexcept
{
    return LOW_LATENCY_GEMM_SWIGLU_PLUGIN_VERSION;
}

int LowLatencyGemmSwigluPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int LowLatencyGemmSwigluPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}

void LowLatencyGemmSwigluPlugin::terminate() noexcept {}

size_t LowLatencyGemmSwigluPlugin::getSerializationSize() const noexcept
{
    return sizeof(nvinfer1::DataType) + // dtype
        sizeof(float) * 3 +             // scales
        sizeof(mDims) + mPluginProfiler->getSerializationSize(mGemmId);
}

void LowLatencyGemmSwigluPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    write(d, mScaleOutput);
    write(d, mScaleD0);
    write(d, mScaleD1);
    write(d, mDims);
    mPluginProfiler->serialize(d, mGemmId);
    TLLM_CHECK(d == a + getSerializationSize());
}

void LowLatencyGemmSwigluPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void LowLatencyGemmSwigluPlugin::configGemm()
{
    mPluginProfiler->profileTactics(mLowLatencyGemmSwigluRunner, mType, mDims, mGemmId);
}

//////////////////////////////////////////////////////////////////////////

LowLatencyGemmSwigluPluginCreator::LowLatencyGemmSwigluPluginCreator()
{

    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("scale_output", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("scale_d0", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("scale_d1", nullptr, PluginFieldType::kFLOAT32));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* LowLatencyGemmSwigluPluginCreator::getPluginName() const noexcept
{
    return LOW_LATENCY_GEMM_SWIGLU_PLUGIN_NAME;
}

char const* LowLatencyGemmSwigluPluginCreator::getPluginVersion() const noexcept
{
    return LOW_LATENCY_GEMM_SWIGLU_PLUGIN_VERSION;
}

PluginFieldCollection const* LowLatencyGemmSwigluPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* LowLatencyGemmSwigluPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    TLLM_CHECK(fc->nbFields == 4);
    nvinfer1::DataType type{};
    float scale_output{};
    float scale_d0{};
    float scale_d1{};
    for (int i = 0; i < fc->nbFields; i++)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "scale_output"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            scale_output = static_cast<float>(*(static_cast<float const*>(fields[i].data)));
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
    }

    try
    {

        //
        // LowLatencyGemmSwigluPluginCreator is unique and shared for an engine generation
        // Create plugin profiler with shared tactics map
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler(/*inference=*/false);
        auto* obj = new LowLatencyGemmSwigluPlugin(type, scale_output, scale_d0, scale_d1, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* LowLatencyGemmSwigluPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler(/*inference=*/true);
        auto* obj = new LowLatencyGemmSwigluPlugin(serialData, serialLength, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
