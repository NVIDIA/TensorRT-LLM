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

#include "fp8CurrentScalingGemmPlugin.h"
#include "cutlass_extensions/gemm_configs.h"

#include <NvInferRuntimeBase.h>
#include <numeric>

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels::small_m_gemm;
using tensorrt_llm::plugins::Fp8CurrentScalingGemmPluginCreator;
using tensorrt_llm::plugins::Fp8CurrentScalingGemmPlugin;

static char const* FP8_CURRENT_SCALING_GEMM_PLUGIN_VERSION{"1"};
static char const* FP8_CURRENT_SCALING_GEMM_PLUGIN_NAME{"Fp8CurrentScalingGemm"};
PluginFieldCollection Fp8CurrentScalingGemmPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> Fp8CurrentScalingGemmPluginCreator::mPluginAttributes;

Fp8CurrentScalingGemmPlugin::Fp8CurrentScalingGemmPlugin(
    int need_quantize_acts_on_demand, int need_quantize_weights_on_demand, nvinfer1::DataType type)
{
    init(type, need_quantize_acts_on_demand, need_quantize_weights_on_demand);
}

// Parameterized constructor
Fp8CurrentScalingGemmPlugin::Fp8CurrentScalingGemmPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    nvinfer1::DataType type;
    int need_quantize_acts_on_demand = 0;
    int need_quantize_weights_on_demand = 0;

    read(d, need_quantize_acts_on_demand);
    read(d, need_quantize_weights_on_demand);
    read(d, type);

    init(type, need_quantize_acts_on_demand, need_quantize_weights_on_demand);

    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

void Fp8CurrentScalingGemmPlugin::init(
    nvinfer1::DataType type, int need_quantize_acts_on_demand, int need_quantize_weights_on_demand)
{
    mArch = tensorrt_llm::common::getSMVersion();
    mType = type;
    mNeedQuantizeActsOnDemand = need_quantize_acts_on_demand;
    mNeedQuantizeWeightsOnDemand = need_quantize_weights_on_demand;
    mInputIdx = 0;
    mWeightInputIdx = mInputIdx + 1;
    mWeightScalesIdx = mNeedQuantizeWeightsOnDemand ? mWeightInputIdx : mWeightInputIdx + 1;
    mInputScalesIdx = mNeedQuantizeActsOnDemand ? mWeightScalesIdx : mWeightScalesIdx + 1;

    if (mType == nvinfer1::DataType::kBF16)
    {
        if (mNeedQuantizeActsOnDemand && !mNeedQuantizeWeightsOnDemand)
        {
            mGemmRunner
                = std::make_shared<CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>>();
        }
        else if (mNeedQuantizeActsOnDemand && mNeedQuantizeWeightsOnDemand)
        {
            mGemmRunner
                = std::make_shared<CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>>();
        }
        else
        {
            mGemmRunner
                = std::make_shared<CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>>();
        }
    }
    else
    {
        TLLM_THROW("Fp8 current scaling Gemm plugin doesn't support this type now");
    }
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* Fp8CurrentScalingGemmPlugin::clone() const noexcept
{
    auto* plugin = new Fp8CurrentScalingGemmPlugin(*this);
    return plugin;
}

nvinfer1::DimsExprs Fp8CurrentScalingGemmPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // inputs
    // 0 activations [M, K]
    // 1 weights     [K, N]
    // 2 weight scales [K // 128, N // 128]
    // 3 activation scales [M, K // 128] (optional)

    try
    {
        TLLM_CHECK(nbInputs == mInputScalesIdx + 1);
        TLLM_CHECK(outputIndex == 0);
        int const nbDimsA = inputs[0].nbDims;
        int const nbDimsB = inputs[mWeightInputIdx].nbDims;
        TLLM_CHECK(nbDimsA >= 2);
        TLLM_CHECK(nbDimsB == 2);
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

bool Fp8CurrentScalingGemmPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (pos == mInputIdx)
    {
        // activations
        if (mNeedQuantizeActsOnDemand)
        {
            return inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
        }
        else
        {
            return inOut[pos].type == nvinfer1::DataType::kFP8 && inOut[pos].format == TensorFormat::kLINEAR;
        }
    }
    else if (pos == mWeightInputIdx)
    {
        // weights
        if (mNeedQuantizeWeightsOnDemand)
        {
            return inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
        }
        else
        {
            return inOut[pos].type == nvinfer1::DataType::kFP8 && inOut[pos].format == TensorFormat::kLINEAR;
        }
    }
    else if (!mNeedQuantizeWeightsOnDemand && pos == mWeightScalesIdx)
    {
        // weight scales
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if (!mNeedQuantizeActsOnDemand && pos == mInputScalesIdx)
    {
        // input scales
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if (pos == mInputScalesIdx + 1)
    {
        // outputs
        return inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else
    {
        return false;
    }
}

void Fp8CurrentScalingGemmPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
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

    mWorkspaceMaxSize = mGemmRunner->getWorkspaceSize(maxM, maxN, maxK);
}

size_t Fp8CurrentScalingGemmPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return mWorkspaceMaxSize;
}

int Fp8CurrentScalingGemmPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // inputs
    //     0 activations  [M, K]
    //     1 weights      [K, N]
    //     2 weight scales[K // 128, N // 128]
    //     3 scales       [M, K // 128] (optional)
    //
    // outputs
    //     mat            [M, N]
    int m = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m *= inputDesc[0].dims.d[ii];
    }
    int const n = inputDesc[1].dims.d[0];
    int const k = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
    // size_t const wsSize = mGemmRunner->getWorkspaceSize(m, n, k);

    mGemmRunner->gemm(outputs[0], inputs[0], inputs[1], m, n, k, reinterpret_cast<char*>(workspace), stream,
        reinterpret_cast<float const*>(inputs[3]), reinterpret_cast<float const*>(inputs[2]));
    sync_check_cuda_error();

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType Fp8CurrentScalingGemmPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == 0);
    return mType;
}

// IPluginV2 Methods

char const* Fp8CurrentScalingGemmPlugin::getPluginType() const noexcept
{
    return FP8_CURRENT_SCALING_GEMM_PLUGIN_NAME;
}

char const* Fp8CurrentScalingGemmPlugin::getPluginVersion() const noexcept
{
    return FP8_CURRENT_SCALING_GEMM_PLUGIN_VERSION;
}

int Fp8CurrentScalingGemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int Fp8CurrentScalingGemmPlugin::initialize() noexcept
{
    // Modify here, maybe do nothing
    configGemm(); // gemm profiler in action
    return 0;
}

void Fp8CurrentScalingGemmPlugin::terminate() noexcept {}

size_t Fp8CurrentScalingGemmPlugin::getSerializationSize() const noexcept
{
    return sizeof(int) +            // need_quantize_acts_on_demand
        sizeof(int) +               // need_quantize_weights_on_demand
        sizeof(nvinfer1::DataType); // dtype
}

void Fp8CurrentScalingGemmPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mNeedQuantizeActsOnDemand);
    write(d, mNeedQuantizeWeightsOnDemand);
    write(d, mType);

    TLLM_CHECK(d == a + getSerializationSize());
}

void Fp8CurrentScalingGemmPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void Fp8CurrentScalingGemmPlugin::configGemm() {}

Fp8CurrentScalingGemmPluginCreator::Fp8CurrentScalingGemmPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("need_quantize_acts_on_demand", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("need_quantize_weights_on_demand", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* Fp8CurrentScalingGemmPluginCreator::getPluginName() const noexcept
{
    return FP8_CURRENT_SCALING_GEMM_PLUGIN_NAME;
}

char const* Fp8CurrentScalingGemmPluginCreator::getPluginVersion() const noexcept
{
    return FP8_CURRENT_SCALING_GEMM_PLUGIN_VERSION;
}

PluginFieldCollection const* Fp8CurrentScalingGemmPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* Fp8CurrentScalingGemmPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    TLLM_CHECK(fc->nbFields == 3);
    int needQuantizeActsOnDemand;
    int needQuantizeWeightsOnDemand;

    nvinfer1::DataType type;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "need_quantize_acts_on_demand"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            needQuantizeActsOnDemand = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "need_quantize_weights_on_demand"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            needQuantizeWeightsOnDemand = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
    }
    try
    {
        // Fp8CurrentScalingGemmPluginCreator is unique and shared for an engine generation
        // Create plugin profiler with shared tactics map
        auto* obj = new Fp8CurrentScalingGemmPlugin(needQuantizeActsOnDemand, needQuantizeWeightsOnDemand, type);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* Fp8CurrentScalingGemmPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call Fp8CurrentScalingGemmPlugin::destroy()
    try
    {
        // Create plugin profiler with private tactics map which is read from the serialized engine
        auto* obj = new Fp8CurrentScalingGemmPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
