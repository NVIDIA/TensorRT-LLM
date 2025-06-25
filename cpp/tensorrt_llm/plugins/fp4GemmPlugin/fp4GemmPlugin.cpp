/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <numeric>

#include "fp4GemmPlugin.h"
#include "tensorrt_llm/common/assert.h"

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::Fp4GemmPluginCreator;
using tensorrt_llm::plugins::Fp4GemmPlugin;
using tensorrt_llm::plugins::Fp4GemmPluginProfiler;
#if defined(USING_OSS_CUTLASS_FP4_GEMM)
using namespace tensorrt_llm::kernels::cutlass_kernels;
#else
using namespace tensorrt_llm::kernels::internal_cutlass_kernels;
#endif

constexpr nvinfer1::DataType FP4_DTYPE = nvinfer1::DataType::kFP4;
constexpr nvinfer1::DataType FP8_DTYPE = nvinfer1::DataType::kFP8;

static char const* FP4_GEMM_PLUGIN_VERSION{"1"};
static char const* FP4_GEMM_PLUGIN_NAME{"Fp4Gemm"};
PluginFieldCollection Fp4GemmPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> Fp4GemmPluginCreator::mPluginAttributes;

void Fp4GemmPluginProfiler::runTactic(
    int m, int n, int k, Fp4GemmPluginProfiler::Config const& tactic, char* workspace, cudaStream_t const& stream)
{
    // Workspace size required by gemm runner
    // NB: this function will throw exception when selected tactic exceeds SMEM, which is then
    // caught by gemmPluginProfiler and it will register this tactic as invalid
    size_t wsSizeRunner = mRunner->getWorkspaceSize(m, n, k, /* batch_count */ 1);

    // Workspace size required by profiling
    size_t wsByteOffset = 0;
    int8_t* wsBytePointer = reinterpret_cast<int8_t*>(workspace);
    void* aTmp = reinterpret_cast<void*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, (m * k) / 2));
    void* bTmp = reinterpret_cast<void*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, (n * k) / 2));
    void* dTmp = reinterpret_cast<void*>(
        nextWorkspacePtr(wsBytePointer, wsByteOffset, m * n * (mType == nvinfer1::DataType::kFLOAT ? 4u : 2u)));
    // SF M/N is padded along 128 and K is padded along 4.
    int vector_size = 16;
    int sf_round_m = ((m + 127) / 128) * 128;
    int sf_round_n = ((n + 127) / 128) * 128;
    int sf_round_k = ((k / vector_size + 3) / 4) * 4;
    float* a_sf = reinterpret_cast<float*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, sf_round_m * sf_round_k));
    float* b_sf = reinterpret_cast<float*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, sf_round_n * sf_round_k));
    float* global_sf = reinterpret_cast<float*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, sizeof(float)));
    char* workspaceTmp = reinterpret_cast<char*>(nextWorkspacePtr(wsBytePointer, wsByteOffset, wsSizeRunner));

    // Run profiling
    mRunner->gemm(dTmp, aTmp, bTmp, a_sf, b_sf, global_sf, m, n, k, /* batch_count */ 1, tactic, workspaceTmp,
        wsSizeRunner, stream);
    sync_check_cuda_error(stream);
}

void Fp4GemmPluginProfiler::computeTmpSize(size_t maxM, size_t n, size_t k)
{
    size_t vector_size = 16;
    size_t sf_round_m = ((maxM + 127) / 128) * 128;
    size_t sf_round_n = ((n + 127) / 128) * 128;
    size_t sf_round_k = ((k / vector_size + 3) / 4) * 4;
    std::vector<size_t> workspaces = {
        (size_t) (maxM * k / 2),                                    // A
        (size_t) (n * k / 2),                                       // B
        maxM * n * (mType == nvinfer1::DataType::kFLOAT ? 4u : 2u), // D
        (size_t) (sf_round_m * sf_round_k),                         // A_SF
        (size_t) (sf_round_n * sf_round_k),                         // B_SF
        sizeof(float),                                              // Global_SF
        mRunner->getWorkspaceSize(maxM, n, k, /* batch_count */ 1)  // workspace
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<Fp4GemmPluginProfiler::Config> Fp4GemmPluginProfiler::getTactics(int m, int n, int k) const
{
    return mRunner->getConfigs();
}

Fp4GemmPlugin::Fp4GemmPlugin(
    int sfVecSize, nvinfer1::DataType OutputType, Fp4GemmPlugin::PluginProfilerPtr const& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
    , mSfVecSize(sfVecSize)
    , mOutputType(OutputType)
{
    init(OutputType);
}

Fp4GemmPlugin::Fp4GemmPlugin(void const* data, size_t length, Fp4GemmPlugin::PluginProfilerPtr const& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mSfVecSize);
    read(d, mOutputType);
    read(d, mDims);

    init(mOutputType);
    mPluginProfiler->deserialize(d, mDims, mGemmId);

    TLLM_CHECK(d == a + length);
}

void Fp4GemmPlugin::init(nvinfer1::DataType type)
{
    TLLM_CHECK_WITH_INFO((getSMVersion() >= 100), "FP4 Gemm not supported before Blackwell");
    TLLM_CHECK_WITH_INFO(
        (mOutputType == DataType::kBF16) || (mOutputType == DataType::kFLOAT) || (mOutputType == DataType::kHALF),
        "Only support float, half, bfloat16, got %d.", (int) mOutputType);
    mOutputType = type;
    if (mOutputType == nvinfer1::DataType::kHALF)
    {
        mGemmRunner = std::make_shared<CutlassFp4GemmRunner<half>>();
    }
    else if (mOutputType == nvinfer1::DataType::kFLOAT)
    {
        mGemmRunner = std::make_shared<CutlassFp4GemmRunner<float>>();
    }
#ifdef ENABLE_BF16
    else if (mOutputType == nvinfer1::DataType::kBF16)
    {
        mGemmRunner = std::make_shared<CutlassFp4GemmRunner<__nv_bfloat16>>();
    }
#endif

    mGemmId = GemmIdCore(mDims.n, mDims.k, mOutputType);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* Fp4GemmPlugin::clone() const noexcept
{
    auto* plugin = new Fp4GemmPlugin(*this);
    return plugin;
}

nvinfer1::DimsExprs Fp4GemmPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    TLLM_CHECK_WITH_INFO(outputIndex == 0, "Only support one output");
    auto const& dimsInput = inputs[getInputTensorIdx()];
    auto const& dimsWeights = inputs[getWeightsTensorIdx()];
    TLLM_CHECK_WITH_INFO(dimsInput.nbDims >= 2 && dimsWeights.nbDims == 2, "Fp4GemmPlugin input dim=%d, weights dim=%d",
        dimsInput.nbDims, dimsWeights.nbDims);
    nvinfer1::DimsExprs ret;
    if (outputIndex == 0)
    {
        ret.nbDims = dimsInput.nbDims;
        for (int i = 0; i < dimsInput.nbDims - 1; ++i)
        {
            ret.d[i] = dimsInput.d[i];
        }
        ret.d[dimsInput.nbDims - 1] = dimsWeights.d[0];
    }
    else
    {
        TLLM_CHECK_WITH_INFO(outputIndex == 0, "output fp4 not supported now.");
        ret.nbDims = 1;
        auto vecCount = dimsInput.d[0];
        int numDim = dimsInput.nbDims;
        for (int idx = 1; idx < numDim - 1; ++idx)
        {
            vecCount = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *vecCount, *dimsInput.d[idx]);
        }
        auto constant128 = exprBuilder.constant(128);
        auto alignedRowCount = exprBuilder.operation(nvinfer1::DimensionOperation::kCEIL_DIV, *vecCount, *constant128);
        alignedRowCount = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *alignedRowCount, *constant128);
        auto constant4 = exprBuilder.constant(4);
        auto constantSFSize = exprBuilder.constant(mSfVecSize);
        auto sfColumn
            = exprBuilder.operation(nvinfer1::DimensionOperation::kCEIL_DIV, *dimsInput.d[numDim - 1], *constantSFSize);
        auto alignedColumnCount = exprBuilder.operation(nvinfer1::DimensionOperation::kCEIL_DIV, *sfColumn, *constant4);
        alignedColumnCount
            = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *alignedColumnCount, *constant4);
        auto totalSize
            = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *alignedColumnCount, *alignedRowCount);
        ret.d[0] = totalSize;
    }
    return ret;
}

bool Fp4GemmPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (inOut[pos].format != TensorFormat::kLINEAR)
    {
        return false;
    }
    if (pos == getInputTensorIdx())
    {
        return (inOut[pos].type == FP4_DTYPE);
    }
    else if (pos == getWeightsTensorIdx())
    {
        return (inOut[pos].type == FP4_DTYPE);
    }
    else if (pos == getInputSFTensorIdx() || pos == getWeightsSFTensorIdx())
    {
        return (inOut[pos].type == FP8_DTYPE);
    }
    else if (pos == getGlobalSFTensorIdx())
    {
        return (inOut[pos].type == DataType::kFLOAT);
    }
    else if (pos == nbInputs)
    {
        // Output
        return (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kBF16
            || inOut[pos].type == DataType::kHALF);
    }
    return false;
}

void Fp4GemmPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    auto const minM = std::accumulate(in[0].min.d, in[0].min.d + in[0].min.nbDims - 1, 1, std::multiplies<int>());
    auto const maxM = std::accumulate(in[0].max.d, in[0].max.d + in[0].max.nbDims - 1, 1, std::multiplies<int>());

    int const maxK = in[0].max.d[in[0].max.nbDims - 1];
    int const maxN = in[2].max.d[0];
    int const minK = in[0].min.d[in[0].min.nbDims - 1];
    int const minN = in[2].min.d[0];

    TLLM_CHECK_WITH_INFO(minN == maxN, "Variable out channels is not allowed");
    TLLM_CHECK_WITH_INFO(minK == maxK, "Variable in channels is not allowed");

    if (!mDims.isInitialized())
    {
        mDims = {minM, maxM, maxN, maxK};
    }
    mGemmId = {maxN, maxK, mOutputType};
    m_workspaceMaxSize = mGemmRunner->getWorkspaceSize(maxM, maxN, maxK, /* batch_count */ 1);
}

size_t Fp4GemmPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return m_workspaceMaxSize;
}

int Fp4GemmPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // inputs
    //     0.  input_tensor [num_tokens, dim]
    //     1.  input_block_scale  [num_tokens, dim / SFVecSize] (padded)
    //     2.  weights_tensor [out_dim, dim]
    //     3.  weights_block_scale [out_dim, dim / SFVecSize] (padded)
    //     4.  alpha (global scaling factor) [1]
    // outputs
    //     0.  output_tensor [num_tokens, out_dim]
    int64_t m = 1;
    for (int i = 0; i < inputDesc[getInputTensorIdx()].dims.nbDims - 1; ++i)
    {
        m *= inputDesc[getInputTensorIdx()].dims.d[i];
    }
    int const n = inputDesc[getWeightsTensorIdx()].dims.d[0];
    int const k = inputDesc[getWeightsTensorIdx()].dims.d[1];
    TLLM_CHECK_WITH_INFO(k % 32 == 0, "K dim should be aligned to 16 Bytes");
    int N_align = mOutputType == nvinfer1::DataType::kFLOAT ? 4u : 8u;
    TLLM_CHECK_WITH_INFO(n % N_align == 0, "N dim should be aligned to 16 Bytes");
    size_t const wsSize = mGemmRunner->getWorkspaceSize(m, n, k, /* batch_count */ 1);
    auto const bestTactic = mPluginProfiler->getBestConfig(m, mGemmId);
    TLLM_CHECK_WITH_INFO(bestTactic, "No valid FP4 GEMM tactic");
    if (m >= 1)
    {
        mGemmRunner->gemm(outputs[0], inputs[0], inputs[2], inputs[1], inputs[3],
            reinterpret_cast<float const*>(inputs[4]), m, n, k, /* batch_count */ 1, *bestTactic,
            reinterpret_cast<char*>(workspace), wsSize, stream);
    }
    sync_check_cuda_error(stream);
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType Fp4GemmPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK_WITH_INFO(index == 0, "Only support one output");
    return mOutputType;
}

// IPluginV2 Methods

char const* Fp4GemmPlugin::getPluginType() const noexcept
{
    return FP4_GEMM_PLUGIN_NAME;
}

char const* Fp4GemmPlugin::getPluginVersion() const noexcept
{
    return FP4_GEMM_PLUGIN_VERSION;
}

int Fp4GemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int Fp4GemmPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}

void Fp4GemmPlugin::terminate() noexcept {}

size_t Fp4GemmPlugin::getSerializationSize() const noexcept
{
    return sizeof(mSfVecSize) +                         // mSfVecSize
        sizeof(nvinfer1::DataType) +                    // dtype
        sizeof(mDims) +                                 // Dimensions
        mPluginProfiler->getSerializationSize(mGemmId); // selected tactics container size
}

void Fp4GemmPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mSfVecSize);
    write(d, mOutputType);
    write(d, mDims);
    mPluginProfiler->serialize(d, mGemmId);
    TLLM_CHECK(d == a + getSerializationSize());
}

void Fp4GemmPlugin::destroy() noexcept
{
    delete this;
}

void Fp4GemmPlugin::configGemm()
{
    mPluginProfiler->profileTactics(mGemmRunner, mOutputType, mDims, mGemmId);
}

///////////////

Fp4GemmPluginCreator::Fp4GemmPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("sv_vec_size", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("output_type_id", nullptr, PluginFieldType::kINT32));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* Fp4GemmPluginCreator::getPluginName() const noexcept
{
    return FP4_GEMM_PLUGIN_NAME;
}

char const* Fp4GemmPluginCreator::getPluginVersion() const noexcept
{
    return FP4_GEMM_PLUGIN_VERSION;
}

PluginFieldCollection const* Fp4GemmPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* Fp4GemmPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    TLLM_CHECK(fc->nbFields == 2);
    int sf_vec_size{};
    nvinfer1::DataType output_type{};
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "sf_vec_size"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            sf_vec_size = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "output_type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            output_type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
    }
    try
    {
        // Fp4GemmPluginCreator is unique and shared for an engine generation
        // Create plugin profiler with shared tactics map
        auto pluginProfiler = mGemmPluginProfileManager.createGemmPluginProfiler(/* inference */ false);
        auto* obj = new Fp4GemmPlugin(sf_vec_size, output_type, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* Fp4GemmPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call CumsumLastDimPlugin::destroy()
    try
    {
        // Create plugin profiler with private tactics map which is read from the serialized engine
        auto pluginProfiler = mGemmPluginProfileManager.createGemmPluginProfiler(/* inference */ true);
        auto* obj = new Fp4GemmPlugin(serialData, serialLength, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
