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
#include "weightOnlyQuantMatmulPlugin.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/enabled.h"

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels::cutlass_kernels;
using tensorrt_llm::plugins::WeightOnlyQuantMatmulPluginCreator;
using tensorrt_llm::plugins::WeightOnlyQuantMatmulPlugin;
using tensorrt_llm::plugins::WeightOnlyQuantGemmPluginProfiler;
using tensorrt_llm::plugins::read;
using tensorrt_llm::plugins::write;

static const char* WOQ_MATMUL_PLUGIN_VERSION{"1"};
static const char* WOQ_MATMUL_PLUGIN_NAME{"WeightOnlyQuantMatmul"};
PluginFieldCollection WeightOnlyQuantMatmulPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> WeightOnlyQuantMatmulPluginCreator::mPluginAttributes;

void WeightOnlyQuantGemmPluginProfiler::runTactic(int m, int n, int k,
    const WeightOnlyQuantGemmPluginProfiler::Config& tactic, char* workspace, const cudaStream_t& stream)
{
    const int originalN = n * getWeightTypeMultiplier(mWeightTypeId);
    half* actPtr = reinterpret_cast<half*>(workspace);
    int8_t* weightPtr
        = reinterpret_cast<int8_t*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(actPtr), m * k * sizeof(half)));
    half* scalesPtr = reinterpret_cast<half*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(weightPtr), originalN * k * sizeof(int8_t)));
    half* outputPtr
        = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(scalesPtr), originalN * sizeof(half)));
    char* workspacePtr
        = reinterpret_cast<char*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(outputPtr), m * originalN * sizeof(half)));

    const int wsSize = mRunner->getWorkspaceSize(m, n, k);

    if (mWeightTypeId == WeightTypeId::INT8)
    {
        mRunner->gemm(actPtr, weightPtr, scalesPtr, outputPtr, m, originalN, k, tactic, workspacePtr, wsSize, stream);
    }
    else
    {
        mRunner->gemm(actPtr, reinterpret_cast<cutlass::uint4b_t*>(weightPtr), scalesPtr, outputPtr, m, originalN, k,
            tactic, workspacePtr, wsSize, stream);
    }
}

void WeightOnlyQuantGemmPluginProfiler::computeTmpSize(int maxM, int n, int k)
{
    const int originalN = n * getWeightTypeMultiplier(mWeightTypeId);
    std::vector<size_t> workspaces = {
        maxM * k * sizeof(half),              // A
        originalN * k * sizeof(int8_t),       // B
        originalN * sizeof(half),             // scales
        maxM * originalN * sizeof(half),      // C
        mRunner->getWorkspaceSize(maxM, n, k) // workspace
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<WeightOnlyQuantGemmPluginProfiler::Config> WeightOnlyQuantGemmPluginProfiler::getTactics(
    int m, int n, int k) const
{
    return mRunner->getConfigs();
}

WeightOnlyQuantMatmulPlugin::WeightOnlyQuantMatmulPlugin(nvinfer1::DataType type, WeightTypeId weightTypeId,
    const WeightOnlyQuantMatmulPlugin::PluginProfilerPtr& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    init(type, weightTypeId);
}

// Parameterized constructor
WeightOnlyQuantMatmulPlugin::WeightOnlyQuantMatmulPlugin(
    const void* data, size_t length, const WeightOnlyQuantMatmulPlugin::PluginProfilerPtr& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    nvinfer1::DataType type;
    WeightTypeId weightTypeId;
    read(d, type);
    read(d, weightTypeId);
    read(d, mDims);

    init(type, weightTypeId);

    mPluginProfiler->deserialize(d, mDims, mGemmId);

    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

void WeightOnlyQuantMatmulPlugin::init(nvinfer1::DataType type, WeightTypeId weightTypeId)
{
    mType = type;
    mWeightTypeId = weightTypeId;
    if (mWeightTypeId == WeightTypeId::INT8)
    {
        if (mType == nvinfer1::DataType::kHALF)
        {
            m_weightOnlyGemmRunner = std::make_shared<
                CutlassFpAIntBGemmRunner<half, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
        }
#if defined(ENABLE_BF16)
        else if (mType == nvinfer1::DataType::kBF16)
        {
            m_weightOnlyGemmRunner = std::make_shared<
                CutlassFpAIntBGemmRunner<__nv_bfloat16, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
        }
#endif
        else
        {
            TLLM_CHECK(false);
        }

        mCudaKernelEnabled
            = tensorrt_llm::kernels::isWeightOnlyBatchedGemvEnabled(tensorrt_llm::kernels::WeightOnlyQuantType::Int8b);
    }
    else if (mWeightTypeId == WeightTypeId::INT4)
    {
        if (mType == nvinfer1::DataType::kHALF)
        {
            m_weightOnlyGemmRunner = std::make_shared<
                CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
        }
#if defined(ENABLE_BF16)
        else if (mType == nvinfer1::DataType::kBF16)
        {
            m_weightOnlyGemmRunner = std::make_shared<CutlassFpAIntBGemmRunner<__nv_bfloat16, cutlass::uint4b_t,
                cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
        }
#endif
        else
        {
            TLLM_CHECK(false);
        }
        mCudaKernelEnabled
            = tensorrt_llm::kernels::isWeightOnlyBatchedGemvEnabled(tensorrt_llm::kernels::WeightOnlyQuantType::Int4b);
    }
    else
    {
        TLLM_CHECK(false);
    }

    mPluginProfiler->setWeightTypeId(mWeightTypeId);

    mGemmId = GemmIdCore(mDims.n, mDims.k, mType);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* WeightOnlyQuantMatmulPlugin::clone() const noexcept
{
    auto* plugin = new WeightOnlyQuantMatmulPlugin(*this);
    return plugin;
}

void WeightOnlyQuantMatmulPlugin::configGemm()
{
    mPluginProfiler->profileTactics(m_weightOnlyGemmRunner, mType, mDims, mGemmId);
}

nvinfer1::DimsExprs WeightOnlyQuantMatmulPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // input [m1, m2, m3, ... , k]
    // weight [k, n] for int8, [k, n/2] for int4

    try
    {
        TLLM_CHECK(nbInputs == 3);
        TLLM_CHECK(outputIndex == 0);
        const int nbDimsA = inputs[0].nbDims;
        const int nbDimsB = inputs[1].nbDims;
        TLLM_CHECK(nbDimsA >= 2);
        TLLM_CHECK(nbDimsB == 2);
        DimsExprs ret;
        ret.nbDims = nbDimsA;
        for (int ii = 0; ii < nbDimsA - 1; ++ii)
        {
            ret.d[ii] = inputs[0].d[ii];
        }
        if (mWeightTypeId == WeightTypeId::INT8)
        {
            // int8 weight only quant
            ret.d[nbDimsA - 1] = exprBuilder.constant(inputs[1].d[1]->getConstantValue());
        }
        else
        {
            // int4 weight only quant
            ret.d[nbDimsA - 1] = exprBuilder.constant(inputs[1].d[1]->getConstantValue() * INT8_INT4_RATIO);
        }
        return ret;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool WeightOnlyQuantMatmulPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    switch (pos)
    {
    case 0:
        // activation
        return inOut[0].type == mType && inOut[0].format == TensorFormat::kLINEAR;
    case 1:
        // weights
        // Weights are required to be int8, but will be reinterpreted as int4 in enqueue if required
        // Weights stored in checkpoint should have int8/int4 type
        return inOut[1].type == nvinfer1::DataType::kINT8 && inOut[1].format == TensorFormat::kLINEAR;
    case 2:
        // scales channels
        return inOut[2].type == mType && inOut[2].format == TensorFormat::kLINEAR;
    case 3:
        // out
        return inOut[3].type == mType && inOut[3].format == TensorFormat::kLINEAR;
    default:
        // Never should be here
        assert(false);
        return false;
    }
}

void WeightOnlyQuantMatmulPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    const auto minM = std::accumulate(in[0].min.d, in[0].min.d + in[0].min.nbDims - 1, 1, std::multiplies<int>());
    const auto maxM = std::accumulate(in[0].max.d, in[0].max.d + in[0].max.nbDims - 1, 1, std::multiplies<int>());

    const int maxK = in[0].max.d[in[0].max.nbDims - 1];
    const int maxN = in[1].max.d[1] * getWeightTypeMultiplier(mWeightTypeId);

    const auto K = maxK;
    const auto N = maxN / getWeightTypeMultiplier(mWeightTypeId);

    if (!mDims.isInitialized())
    {
        mDims = {minM, maxM, N, K};
    }

    mGemmId = {N, K, mType};

    m_workspaceMaxSize = m_weightOnlyGemmRunner->getWorkspaceSize(maxM, maxN, maxK);
}

size_t WeightOnlyQuantMatmulPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return m_workspaceMaxSize;
}

int WeightOnlyQuantMatmulPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // inputs
    //     mat1           [M1, M2,..., K]
    //     mat2           [K, N] for int8, [K, N/2] for int4
    //     scale_channels [N]
    // outputs
    //     mat [M, N]

    int m = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m *= inputDesc[0].dims.d[ii];
    }
    const int n = inputDesc[1].dims.d[1];
    const int k = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];

    const bool use_cuda_kernel = m < SMALL_M_FAST_PATH && mCudaKernelEnabled;
#if defined(ENABLE_BF16)
    TLLM_CHECK_WITH_INFO(mType == nvinfer1::DataType::kHALF || mType == nvinfer1::DataType::kBF16,
        "No valid weightOnlyQuantMatmul configuration");
#else
    TLLM_CHECK_WITH_INFO(mType == nvinfer1::DataType::kHALF, "No valid weightOnlyQuantMatmul configuration");
#endif

    tensorrt_llm::kernels::WeightOnlyQuantType weight_only_quant_type;
    tensorrt_llm::kernels::WeightOnlyActivationType weight_only_act_type;
    int real_n;
    if (mType == nvinfer1::DataType::kHALF)
    {
        weight_only_act_type = tensorrt_llm::kernels::WeightOnlyActivationType::FP16;
    }
    else if (mType == nvinfer1::DataType::kBF16)
    {
        weight_only_act_type = tensorrt_llm::kernels::WeightOnlyActivationType::BF16;
    }
    if (mWeightTypeId == WeightTypeId::INT8)
    {
        weight_only_quant_type = tensorrt_llm::kernels::WeightOnlyQuantType::Int8b;
        real_n = n;
    }
    else if (mWeightTypeId == WeightTypeId::INT4)
    {
        weight_only_quant_type = tensorrt_llm::kernels::WeightOnlyQuantType::Int4b;
        real_n = n * INT8_INT4_RATIO;
    }
    if (use_cuda_kernel)
    {
        // Use CUDA kernels for small batch size
        // The CUDA kernel is designed for ColumnMajorTileInterleave weight layout used in fpAIntB cutlass
        // kernel when sm >= 75 and the preprocessing of cutlass on sm70 does not interleave the weights.
        tensorrt_llm::kernels::WeightOnlyParams params{reinterpret_cast<const uint8_t*>(inputs[1]), inputs[2], nullptr,
            inputs[0], nullptr, nullptr, outputs[0], m, real_n, k, 0, weight_only_quant_type,
            tensorrt_llm::kernels::WeightOnlyType::PerChannel,
            tensorrt_llm::kernels::WeightOnlyActivationFunctionType::Identity, weight_only_act_type};
        tensorrt_llm::kernels::weight_only_batched_gemv_launcher(params, stream);
    }
    else
    {
        const int ws_size = m_weightOnlyGemmRunner->getWorkspaceSize(m, real_n, k);

        const auto& bestTactic = mPluginProfiler->getBestConfig(m, mGemmId);
        TLLM_CHECK_WITH_INFO(bestTactic,
            "No valid weight only per-channel GEMM tactic(It is usually caused by the failure to execute all candidate "
            "configurations of the CUTLASS kernel, please pay attention to the warning information when building the "
            "engine.)");

        m_weightOnlyGemmRunner->gemm(inputs[0], inputs[1], inputs[2], outputs[0], m, real_n, k, *bestTactic,
            reinterpret_cast<char*>(workspace), ws_size, stream);
    }

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType WeightOnlyQuantMatmulPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == 0);
    return mType;
}

// IPluginV2 Methods

const char* WeightOnlyQuantMatmulPlugin::getPluginType() const noexcept
{
    return WOQ_MATMUL_PLUGIN_NAME;
}

const char* WeightOnlyQuantMatmulPlugin::getPluginVersion() const noexcept
{
    return WOQ_MATMUL_PLUGIN_VERSION;
}

int WeightOnlyQuantMatmulPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int WeightOnlyQuantMatmulPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}

void WeightOnlyQuantMatmulPlugin::terminate() noexcept {}

size_t WeightOnlyQuantMatmulPlugin::getSerializationSize() const noexcept
{
    return sizeof(mWeightTypeId) +                      // mWeightTypeId
        sizeof(nvinfer1::DataType) +                    // mType
        sizeof(mDims) +                                 // Dimensions
        mPluginProfiler->getSerializationSize(mGemmId); // selected tactics container size
}

void WeightOnlyQuantMatmulPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    write(d, mWeightTypeId);
    write(d, mDims);

    mPluginProfiler->serialize(d, mGemmId);
    assert(d == a + getSerializationSize());
}

void WeightOnlyQuantMatmulPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

WeightOnlyQuantMatmulPluginCreator::WeightOnlyQuantMatmulPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("weight_type_id", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* WeightOnlyQuantMatmulPluginCreator::getPluginName() const noexcept
{
    return WOQ_MATMUL_PLUGIN_NAME;
}

const char* WeightOnlyQuantMatmulPluginCreator::getPluginVersion() const noexcept
{
    return WOQ_MATMUL_PLUGIN_VERSION;
}

const PluginFieldCollection* WeightOnlyQuantMatmulPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* WeightOnlyQuantMatmulPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    nvinfer1::DataType type;
    WeightTypeId weightTypeId;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "weight_type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            weightTypeId = static_cast<WeightTypeId>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
    }
    try
    {
        // WeightOnlyGroupwiseQuantMatmulPluginCreator is unique and shared for an engine generation
        // Create plugin profiler with shared tactics map
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler(/* inference */ false);
        auto* obj = new WeightOnlyQuantMatmulPlugin(type, weightTypeId, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* WeightOnlyQuantMatmulPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call WeightOnlyQuantMatmulPlugin::destroy()
    try
    {
        // Create plugin profiler with private tactics map which is read from the serialized engine
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler(/* inference */ true);
        auto* obj = new WeightOnlyQuantMatmulPlugin(serialData, serialLength, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
