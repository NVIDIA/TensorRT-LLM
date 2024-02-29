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
#include "gemmPlugin.h"
#include "plugin.h"
#include <NvInferRuntimeBase.h>

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::GemmPluginCreator;
using tensorrt_llm::plugins::GemmPlugin;
using tensorrt_llm::plugins::CublasLtGemmPluginProfiler;
using tensorrt_llm::plugins::CublasGemmWrapperPtr;
using tensorrt_llm::plugins::read;
using tensorrt_llm::plugins::write;

static const char* GEMM_PLUGIN_VERSION{"1"};
static const char* GEMM_PLUGIN_NAME{"Gemm"};
PluginFieldCollection GemmPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> GemmPluginCreator::mPluginAttributes;

void getProblemParams(cublasOperation_t& transa, cublasOperation_t& transb, int& m, int& n, int& k, int& lda, int& ldb,
    int& ldc, bool transA, bool transB, int M, int N, int K)
{
    transa = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    transb = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    m = N;
    n = M;
    k = K;
    lda = transB ? K : N;
    ldb = transA ? M : K;
    ldc = N;
}

void runGemm(const int M, const int N, const int K, const bool transA, const bool transB, const nvinfer1::DataType type,
    const CublasGemmWrapperPtr& cublasWrapperPtr, const void* act, const void* weight, void* output,
    const std::optional<cublasLtMatmulHeuristicResult_t>& heuristic, void* workspace, cudaStream_t stream)
{
    cublasWrapperPtr->setStream(stream);
    cublasWrapperPtr->setWorkspace(workspace);

    cublasOperation_t transa, transb;
    int m, n, k;
    int lda, ldb, ldc;
    getProblemParams(transa, transb, m, n, k, lda, ldb, ldc, transA, transB, M, N, K);

    cublasWrapperPtr->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);
    cublasWrapperPtr->Gemm(transa, transb, m, n, k, weight, lda, act, ldb, output, ldc, heuristic);
    cublasWrapperPtr->destroyDescriptors();
}

void CublasLtGemmPluginProfiler::runTactic(
    int m, int n, int k, const CublasLtGemmPluginProfiler::Config& tactic, char* workspace, const cudaStream_t& stream)
{
    size_t dataSize = sizeof(half);
    if (mType == DataType::kFLOAT)
    {
        dataSize = sizeof(float);
    }

    void* actPtr = reinterpret_cast<void*>(workspace);
    void* weightPtr = reinterpret_cast<void*>(
        nextWorkspacePtrWithAlignment(reinterpret_cast<int8_t*>(actPtr), m * k * dataSize, ALIGNMENT));
    void* outputPtr = reinterpret_cast<void*>(
        nextWorkspacePtrWithAlignment(reinterpret_cast<int8_t*>(weightPtr), n * k * dataSize, ALIGNMENT));
    char* workspacePtr = reinterpret_cast<char*>(
        nextWorkspacePtrWithAlignment(reinterpret_cast<int8_t*>(outputPtr), m * n * dataSize, ALIGNMENT));
    runGemm(m, n, k, mTransA, mTransB, mType, mRunner, actPtr, weightPtr, outputPtr, {tactic}, workspacePtr, stream);
}

bool CublasLtGemmPluginProfiler::checkTactic(int m, int n, int k, const Config& tactic) const
{
    cublasOperation_t transa, transb;
    int M = m, N = n, K = k;
    int lda, ldb, ldc;
    getProblemParams(transa, transb, m, n, k, lda, ldb, ldc, mTransA, mTransB, M, N, K);

    mRunner->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);

    const auto checkResult = mRunner->checkTactic(transa, transb, m, n, k, lda, ldb, ldc, tactic.algo);

    mRunner->destroyDescriptors();

    return checkResult;
}

void CublasLtGemmPluginProfiler::computeTmpSize(int maxM, int n, int k)
{
    size_t dataSize = typeSize(mType);
    size_t outputDataSize = typeSize(mOutputType);

    std::vector<size_t> workspaces = {
        maxM * k * dataSize,       // A
        n * k * dataSize,          // B
        maxM * n * outputDataSize, // C
        CUBLAS_WORKSPACE_SIZE      // workspace
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size(), ALIGNMENT);
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<CublasLtGemmPluginProfiler::Config> CublasLtGemmPluginProfiler::getTactics(int M, int N, int K) const
{
    cublasOperation_t transa, transb;
    int m, n, k;
    int lda, ldb, ldc;
    getProblemParams(transa, transb, m, n, k, lda, ldb, ldc, mTransA, mTransB, M, N, K);

    mRunner->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);
    const auto heruistics = mRunner->getTactics(transa, transb, m, n, k, lda, ldb, ldc);
    mRunner->destroyDescriptors();

    return heruistics;
}

GemmPlugin::GemmPlugin(
    int transA, int transB, nvinfer1::DataType type, bool useFp8, const GemmPlugin::PluginProfilerPtr& pluginProfiler)
    : mTransA(transA)
    , mTransB(transB)
    , mType(type)
    , mUseFp8(useFp8)
    , mPluginProfiler(pluginProfiler)
    , mOutputType(type)
{
    init();
}

// Parameterized constructor
GemmPlugin::GemmPlugin(const void* data, size_t length, const GemmPlugin::PluginProfilerPtr& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mTransA);
    read(d, mTransB);
    read(d, mType);
    read(d, mUseFp8);
    read(d, mDims);
    read(d, mOutputType);

    init();

    mPluginProfiler->deserialize(d, mDims, mGemmId);

    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

void GemmPlugin::init()
{
    auto cublasHandle = getCublasHandle();
    auto cublasLtHandle = getCublasLtHandle();
    mCublasWrapper = std::make_shared<CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);

    mPluginProfiler->setTranspose(mTransA, mTransB);
    mPluginProfiler->setOutputType(mOutputType);

    mGemmId = GemmIdCublas(mDims.n, mDims.k, mType, mTransA, mTransB);
}

void GemmPlugin::setGemmConfig()
{
    if (mType == DataType::kHALF)
    {
        mCublasWrapper->setFP16GemmConfig(trtToCublasDtype(mOutputType));
    }
    else if (mType == DataType::kFLOAT)
    {
        mCublasWrapper->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16)
    {
        mCublasWrapper->setBF16GemmConfig(trtToCublasDtype(mOutputType));
    }
#endif

#ifdef ENABLE_FP8
    if (mUseFp8)
    {
        mCublasWrapper->setFP8GemmConfig(trtToCublasDtype(mOutputType));
    }
#endif
}

void GemmPlugin::configGemm()
{
    if (!mDims.isInitialized())
    {
        return;
    }

    setGemmConfig();

    mPluginProfiler->profileTactics(mCublasWrapper, mType, mDims, mGemmId);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* GemmPlugin::clone() const noexcept
{
    auto* plugin = new GemmPlugin(*this);
    return plugin;
}

nvinfer1::DimsExprs GemmPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        TLLM_CHECK(nbInputs == 2);
        TLLM_CHECK(outputIndex == 0);
        const int nbDimsA = inputs[0].nbDims;
        const int nbDimsB = inputs[1].nbDims;
        DimsExprs ret;
        ret.nbDims = nbDimsA + nbDimsB - 2;

        if (mTransA)
        {
            for (int i = 1; i < nbDimsA; ++i)
            {
                ret.d[i - 1] = inputs[0].d[i];
            }
        }
        else
        {
            for (int i = 0; i < nbDimsA - 1; ++i)
            {
                ret.d[i] = inputs[0].d[i];
            }
        }
        if (mTransB)
        {
            for (int i = 0; i < nbDimsB - 1; ++i)
            {
                ret.d[nbDimsA - 1 + i] = inputs[1].d[i];
            }
        }
        else
        {
            for (int i = 1; i < nbDimsB; ++i)
            {
                ret.d[nbDimsA - 2 + i] = inputs[1].d[i];
            }
        }
        return ret;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool GemmPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    const auto& desc = inOut[pos];
    if (desc.format != TensorFormat::kLINEAR)
    {
        return false;
    }

    if (pos < nbInputs)
    {
        return desc.type == mType;
    }

    return desc.type == mType || desc.type == nvinfer1::DataType::kFLOAT;
}

int32_t computeMDimension(bool transA, const int32_t nbDims, const int32_t* dims)
{
    int32_t M = 1;
    if (transA)
    {
        for (int i = nbDims - 1; i > 0; --i)
        {
            M *= dims[i];
        }
    }
    else
    {
        for (int i = 0; i < nbDims - 1; ++i)
        {
            M *= dims[i];
        }
    }
    return M;
}

int32_t computeNDimension(bool transB, const int32_t nbDims, const int32_t* dims)
{
    int32_t N = 1;
    if (transB)
    {
        for (int i = 0; i < nbDims - 1; ++i)
        {
            N *= dims[i];
        }
    }
    else
    {
        for (int i = nbDims - 1; i > 0; --i)
        {
            N *= dims[i];
        }
    }
    return N;
}

void GemmPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    const int nbDimsA = in[0].max.nbDims;
    const int nbDimsB = in[1].max.nbDims;

    const auto minM = computeMDimension(mTransA, nbDimsA, in[0].min.d);
    const auto maxM = computeMDimension(mTransA, nbDimsA, in[0].max.d);
    const auto N = computeNDimension(mTransB, nbDimsB, in[1].max.d);
    const auto K = mTransA ? in[0].max.d[0] : in[0].max.d[nbDimsA - 1];

    if (!mDims.isInitialized())
    {
        mDims = {minM, maxM, N, K};
    }
    mGemmId.n = N;
    mGemmId.k = K;

    mOutputType = out[0].desc.type;
}

size_t GemmPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return CUBLAS_WORKSPACE_SIZE;
}

int GemmPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // inputs
    //     mat1 [M, K] (mTransA = False)
    //     mat2 [K, N] (mTransB = False)
    // outputs
    //     mat [M, N]

    setGemmConfig();

    const int nbDimsA = inputDesc[0].dims.nbDims;
    const int nbDimsB = inputDesc[1].dims.nbDims;
    const auto M = computeMDimension(mTransA, nbDimsA, inputDesc[0].dims.d);
    const auto N = computeNDimension(mTransB, nbDimsB, inputDesc[1].dims.d);
    const int K = mTransA ? inputDesc[0].dims.d[0] : inputDesc[0].dims.d[nbDimsA - 1];

    auto bestTactic = mPluginProfiler->getBestConfig(M, mGemmId);
    runGemm(M, N, K, mTransA, mTransB, mType, mCublasWrapper, inputs[0], inputs[1], outputs[0], bestTactic, workspace,
        stream);
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType GemmPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

const char* GemmPlugin::getPluginType() const noexcept
{
    return GEMM_PLUGIN_NAME;
}

const char* GemmPlugin::getPluginVersion() const noexcept
{
    return GEMM_PLUGIN_VERSION;
}

int GemmPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int GemmPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}

void GemmPlugin::destroy() noexcept
{
    delete this;
}

size_t GemmPlugin::getSerializationSize() const noexcept
{
    return sizeof(mTransA) + sizeof(mTransB) + sizeof(mType) + sizeof(mDims) + sizeof(mUseFp8)
        + mPluginProfiler->getSerializationSize(mGemmId) + sizeof(mOutputType); // selected tactics container size
}

void GemmPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mTransA);
    write(d, mTransB);
    write(d, mType);
    write(d, mUseFp8);
    write(d, mDims);
    write(d, mOutputType);
    mPluginProfiler->serialize(d, mGemmId);

    assert(d == a + getSerializationSize());
}

void GemmPlugin::terminate() noexcept {}

///////////////

GemmPluginCreator::GemmPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("transA", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("transB", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("use_fp8", nullptr, PluginFieldType::kINT32, 0));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GemmPluginCreator::getPluginName() const noexcept
{
    return GEMM_PLUGIN_NAME;
}

const char* GemmPluginCreator::getPluginVersion() const noexcept
{
    return GEMM_PLUGIN_VERSION;
}

const PluginFieldCollection* GemmPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* GemmPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int transA, transB;
    nvinfer1::DataType type;
    int useFp8;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "transa"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            transA = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "transb"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            transB = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "use_fp8"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            useFp8 = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
    }
    try
    {
        // GemmPluginCreator is unique and shared for an engine generation
        // Create plugin profiler with shared tactics map
        // FIXME enable tactic profiler
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler(/* inference */ false, /* skip */ true);
        auto* obj = new GemmPlugin(transA, transB, type, useFp8, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* GemmPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call GemmPlugin::destroy()
    try
    {
        // GemmPluginCreator is unique and shared for an engine generation
        // Create plugin profiler with shared tactics map
        // FIXME enable tactic profiler
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler(/* inference */ true, /* skip */ true);
        auto* obj = new GemmPlugin(serialData, serialLength, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
