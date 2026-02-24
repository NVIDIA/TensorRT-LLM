
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

#include "lowLatencyGemmPlugin.h"
#include "low_latency_gemm.h"
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
using tensorrt_llm::plugins::LowLatencyGemmPluginCreator;
using tensorrt_llm::plugins::LowLatencyGemmPlugin;
using tensorrt_llm::plugins::LowLatencyGemmPluginProfiler;
using tensorrt_llm::plugins::read;
using tensorrt_llm::plugins::write;

static char const* LOW_LATENCY_GEMM_PLUGIN_VERSION{"1"};
static char const* LOW_LATENCY_GEMM_PLUGIN_NAME{"LowLatencyGemm"};

PluginFieldCollection LowLatencyGemmPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> LowLatencyGemmPluginCreator::mPluginAttributes;

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

void LowLatencyGemmPluginProfiler::runTactic(int m, int n, int k, LowLatencyGemmPluginProfiler::Config const& tactic,
    char* workspace, cudaStream_t const& stream)
{

    float default_pdl_overlap_ratio = 0.5;
    float default_prefetch_ratio = -1.0;
    FP8Type* aTmp = reinterpret_cast<FP8Type*>(workspace);
    FP8Type* bTmp
        = reinterpret_cast<FP8Type*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(aTmp), m * k * sizeof(FP8Type)));
    void* cTmp = reinterpret_cast<void*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(bTmp), n * k * sizeof(FP8Type)));
    size_t workspaceSize = mRunner->getWorkspaceSize(m, n, k);
    char* workspaceTmp = reinterpret_cast<char*>(nextWorkspacePtr(
        reinterpret_cast<int8_t*>(cTmp), m * n * (mType == nvinfer1::DataType::kFLOAT ? sizeof(float) : sizeof(half))));
    mRunner->gemm(aTmp, bTmp, 1.0f, 0.0f, nullptr, cTmp, m, n, k, default_pdl_overlap_ratio, default_prefetch_ratio,
        tactic, workspaceTmp, workspaceSize, stream);
}

void LowLatencyGemmPluginProfiler::computeTmpSize(size_t maxM, size_t n, size_t k)
{

    std::vector<size_t> workspaces = {maxM * k * sizeof(FP8Type), n * k * sizeof(FP8Type),
        maxM * n * (mType == nvinfer1::DataType::kFLOAT ? sizeof(float) : sizeof(half)),
        mRunner->getWorkspaceSize(maxM, n, k)};

    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<LowLatencyGemmPluginProfiler::Config> LowLatencyGemmPluginProfiler::getTactics(int m, int n, int k) const
{
    return mRunner->getConfigs();
}

LowLatencyGemmPlugin::LowLatencyGemmPlugin(
    nvinfer1::DataType type, float alpha, PluginProfilerPtr const& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
    , mAplha(alpha)
{
    init(type);
}

LowLatencyGemmPlugin::LowLatencyGemmPlugin(void const* data, size_t length, PluginProfilerPtr const& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{

    char const *d = reinterpret_cast<char const*>(data), *a = d;
    nvinfer1::DataType type;
    read(d, type);
    read(d, mAplha);
    read(d, mDims);
    init(type);
    mPluginProfiler->deserialize(d, mDims, mGemmId);
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

void LowLatencyGemmPlugin::init(nvinfer1::DataType type)
{

    mType = type;

    if (mType == nvinfer1::DataType::kFLOAT)
    {
        m_lowLatencyGemmRunner = std::make_shared<CutlassLowLatencyFp8GemmRunner<float>>();
    }
    else if (mType == nvinfer1::DataType::kHALF)
    {
        m_lowLatencyGemmRunner = std::make_shared<CutlassLowLatencyFp8GemmRunner<half>>();
    }
#ifdef ENABLE_BF16

    else if (mType == nvinfer1::DataType::kBF16)
    {
        m_lowLatencyGemmRunner = std::make_shared<CutlassLowLatencyFp8GemmRunner<__nv_bfloat16>>();
    }
#endif
    else
    {
        TLLM_THROW("Unsupported data type");
    }
    mGemmId = GemmIdCore(mDims.n, mDims.k, mType);
}

nvinfer1::DimsExprs LowLatencyGemmPlugin::getOutputDimensions(
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
        // input[1] , weights [n,k]
        ret.d[nbDimsA - 1] = inputs[1].d[0];
        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool LowLatencyGemmPlugin::supportsFormatCombination(
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
        assert(false);
        return false;
    }
}

void LowLatencyGemmPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
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
    mGemmId = {maxN, maxK, mType};

    m_workspaceMaxSize = m_lowLatencyGemmRunner->getWorkspaceSize(maxM, maxN, maxK);
}

size_t LowLatencyGemmPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return m_workspaceMaxSize;
}

int LowLatencyGemmPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{

    // input0 activation [M,K]
    // input1 weights [N,K]
    // output0 [M,N]

    int64_t m64 = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m64 *= inputDesc[0].dims.d[ii];
    }
    int const m = TLLM_INT32_CAST(m64);
    int const n = TLLM_INT32_CAST(inputDesc[1].dims.d[0]);
    int const k = TLLM_INT32_CAST(inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1]);
    int const wsSize = m_lowLatencyGemmRunner->getWorkspaceSize(m, n, k);
    auto const& bestTactic = mPluginProfiler->getBestConfig(m, mGemmId);
    TLLM_CHECK_WITH_INFO(bestTactic, "No valid Low Latency GEMM tactic");

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
    m_lowLatencyGemmRunner->gemm(const_cast<FP8Type*>(reinterpret_cast<FP8Type const*>(inputs[0])),
        const_cast<FP8Type*>(reinterpret_cast<FP8Type const*>(inputs[1])), mAplha, 0.0F, nullptr, outputs[0], m, n, k,
        pdl_overlap_ratio, prefetch_ratio, *bestTactic, reinterpret_cast<char*>(workspace), wsSize, stream);

    return 0;
}

nvinfer1::DataType LowLatencyGemmPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == 0);
    return mType;
}

// IPluginV2 Methods

char const* LowLatencyGemmPlugin::getPluginType() const noexcept
{
    return LOW_LATENCY_GEMM_PLUGIN_NAME;
}

char const* LowLatencyGemmPlugin::getPluginVersion() const noexcept
{
    return LOW_LATENCY_GEMM_PLUGIN_VERSION;
}

int LowLatencyGemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int LowLatencyGemmPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}

void LowLatencyGemmPlugin::terminate() noexcept {}

nvinfer1::IPluginV2DynamicExt* LowLatencyGemmPlugin::clone() const noexcept
{
    auto* plugin = new LowLatencyGemmPlugin(*this);
    return plugin;
}

size_t LowLatencyGemmPlugin::getSerializationSize() const noexcept
{
    return sizeof(nvinfer1::DataType) + // dtype
        sizeof(float) * 1 +             // alpha
        sizeof(mDims) + mPluginProfiler->getSerializationSize(mGemmId);
}

void LowLatencyGemmPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    write(d, mAplha);
    write(d, mDims);
    mPluginProfiler->serialize(d, mGemmId);
    TLLM_CHECK(d == a + getSerializationSize());
}

void LowLatencyGemmPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void LowLatencyGemmPlugin::configGemm()
{
    mPluginProfiler->profileTactics(m_lowLatencyGemmRunner, mType, mDims, mGemmId);
}

LowLatencyGemmPluginCreator::LowLatencyGemmPluginCreator()
{

    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("alpha", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* LowLatencyGemmPluginCreator::getPluginName() const noexcept
{
    return LOW_LATENCY_GEMM_PLUGIN_NAME;
}

char const* LowLatencyGemmPluginCreator::getPluginVersion() const noexcept
{
    return LOW_LATENCY_GEMM_PLUGIN_VERSION;
}

PluginFieldCollection const* LowLatencyGemmPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* LowLatencyGemmPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    float alpha{};
    nvinfer1::DataType type{};
    for (int i = 0; i < fc->nbFields; i++)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "alpha"))
        {

            TLLM_CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            alpha = *(static_cast<float const*>(fields[i].data));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
    }

    try
    {

        //
        // GemmPluginCreator is unique and shared for an engine generation
        // Create plugin profiler with shared tactics map

        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler(/*inference=*/false);
        auto* obj = new LowLatencyGemmPlugin(type, alpha, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* LowLatencyGemmPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler(/*inference=*/true);
        auto* obj = new LowLatencyGemmPlugin(serialData, serialLength, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
