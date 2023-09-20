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
#include "tensorrt_llm/plugins/gemmPlugin/gemmPlugin.h"

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using nvinfer1::plugin::GemmPluginCreator;
using nvinfer1::plugin::GemmPlugin;

static const char* GEMM_PLUGIN_VERSION{"1"};
static const char* GEMM_PLUGIN_NAME{"Gemm"};
PluginFieldCollection GemmPluginCreator::mFC{};
std::vector<PluginField> GemmPluginCreator::mPluginAttributes;

GemmPlugin::GemmPlugin(int transA, int transB, nvinfer1::DataType type)
    : mTransA(transA)
    , mTransB(transB)
    , mType(type)
{
}

// Parameterized constructor
GemmPlugin::GemmPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mTransA);
    read(d, mTransB);
    read(d, mType);
    PLUGIN_ASSERT(d == a + length);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* GemmPlugin::clone() const noexcept
{
    auto* plugin = new GemmPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
}

nvinfer1::DimsExprs GemmPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_ASSERT(nbInputs == 2);
        PLUGIN_ASSERT(outputIndex == 0);
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
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void GemmPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
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

    auto cublasHandle = mCublasWrapper->getCublasHandle();
    PLUGIN_CUBLASASSERT(cublasSetStream(cublasHandle, stream));
    mCublasWrapper->setStream(stream);
    mCublasWrapper->setWorkspace(workspace);
    if (mType == DataType::kHALF)
    {
        mCublasWrapper->setFP16GemmConfig();
    }
    else if (mType == DataType::kFLOAT)
    {
        mCublasWrapper->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16)
    {
        mCublasWrapper->setBF16GemmConfig();
    }
#endif

    const int nbDimsA = inputDesc[0].dims.nbDims;
    int M = 1, N = 1;
    const int K = mTransA ? inputDesc[0].dims.d[0] : inputDesc[0].dims.d[nbDimsA - 1];
    if (mTransA)
    {
        for (int i = nbDimsA - 1; i > 0; --i)
        {
            M *= inputDesc[0].dims.d[i];
        }
    }
    else
    {
        for (int i = 0; i < nbDimsA - 1; ++i)
        {
            M *= inputDesc[0].dims.d[i];
        }
    }
    const int nbDimsB = inputDesc[1].dims.nbDims;
    if (mTransB)
    {
        for (int i = 0; i < nbDimsB - 1; ++i)
        {
            N *= inputDesc[1].dims.d[i];
        }
    }
    else
    {
        for (int i = nbDimsB - 1; i > 0; --i)
        {
            N *= inputDesc[1].dims.d[i];
        }
    }

    cublasOperation_t transa = mTransB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = mTransA ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int m = N;
    const int n = M;
    const int k = K;
    const auto lda = mTransB ? K : N;
    const auto ldb = mTransA ? M : K;
    const auto ldc = N;
    mCublasWrapper->Gemm(transa, transb, m, n, k, inputs[1], lda, inputs[0], ldb, outputs[0], ldc);

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType GemmPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(index == 0);
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
    auto cublasHandle = getCublasHandle();
    auto cublasLtHandle = getCublasLtHandle();
    mCublasAlgoMap = new cublasAlgoMap(GEMM_CONFIG);
    mCublasWrapperMutex = new std::mutex();
    mCublasWrapper
        = new cublasMMWrapper(cublasHandle, cublasLtHandle, nullptr, mCublasAlgoMap, mCublasWrapperMutex, nullptr);
    return 0;
}

void GemmPlugin::destroy() noexcept
{
    delete mCublasAlgoMap;
    delete mCublasWrapperMutex;
    delete mCublasWrapper;

    mCublasAlgoMap = nullptr;
    mCublasWrapperMutex = nullptr;
    mCublasWrapper = nullptr;
    delete this;
}

size_t GemmPlugin::getSerializationSize() const noexcept
{
    return sizeof(mTransA) + sizeof(mTransB) + sizeof(mType);
}

void GemmPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mTransA);
    write(d, mTransB);
    write(d, mType);
    assert(d == a + getSerializationSize());
}

void GemmPlugin::terminate() noexcept {}

void GemmPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* GemmPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

///////////////

GemmPluginCreator::GemmPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("transA", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("transB", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
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
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "transa"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            transA = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "transb"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            transB = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new GemmPlugin(transA, transB, type);
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
        auto* obj = new GemmPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void GemmPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* GemmPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
