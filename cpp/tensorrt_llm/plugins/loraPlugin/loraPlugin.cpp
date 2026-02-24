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

#include "loraPlugin.h"

#include "pluginUtils.h"
#include "tensorrt_llm/common/assert.h"

#include <vector>

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::LoraPluginCreator;
using tensorrt_llm::plugins::LoraPlugin;
using tensorrt_llm::plugins::read;
using tensorrt_llm::plugins::write;

static char const* LORA_PLUGIN_VERSION{"1"};
static char const* LORA_PLUGIN_NAME{"Lora"};
PluginFieldCollection LoraPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> LoraPluginCreator::mPluginAttributes;

LoraPlugin::LoraPlugin(int in_hidden_size, std::vector<int> out_hidden_sizes, int transA, int transB,
    int num_lora_modules, nvinfer1::DataType type, LoraPlugin::PluginProfilerPtr const& pluginProfiler,
    bool remove_input_padding, int max_low_rank, int weight_index)
    : mTransA(transA)
    , mTransB(transB)
    , mType(type)
    , mRemoveInputPadding(remove_input_padding)
    , mNumLoraModules(num_lora_modules)
    , mInHiddenSize(in_hidden_size)
    , mMaxLowRank(max_low_rank)
    , mWeightIndex(weight_index)
    , mPluginProfiler(pluginProfiler)
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    mOutHiddenSizes.resize(mNumLoraModules);
    mOutHiddenSizes.assign(out_hidden_sizes.begin(), out_hidden_sizes.end());
    init();
}

// Parameterized constructor
LoraPlugin::LoraPlugin(void const* data, size_t length, LoraPlugin::PluginProfilerPtr const& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mInHiddenSize);
    read(d, mTransA);
    read(d, mTransB);
    read(d, mNumLoraModules);
    read(d, mType);
    read(d, mRemoveInputPadding);
    read(d, mMaxLowRank);
    read(d, mWeightIndex);
    mOutHiddenSizes.resize(mNumLoraModules);
    for (int i = 0; i < mNumLoraModules; i++)
    {
        read(d, mOutHiddenSizes[i]);
    }
    init();

    mPluginProfiler->deserialize(d, mDims, mGemmId);

    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

void LoraPlugin::init()
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);

    auto cublasHandle = getCublasHandle();
    auto cublasLtHandle = getCublasLtHandle();
    auto cublasWraper = std::make_shared<CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);

    mLoraImpl = std::make_shared<kernels::LoraImpl>(
        mInHiddenSize, mOutHiddenSizes, mTransA, mTransB, mNumLoraModules, mType, mMaxLowRank, cublasWraper);

    mPluginProfiler->setTranspose(mTransA, mTransB);
    mGemmId = GemmIdCublas(mDims.n, mDims.k, mType, mTransA, mTransB, mType);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* LoraPlugin::clone() const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    auto* plugin = new LoraPlugin(*this);
    return plugin;
}

nvinfer1::DimsExprs LoraPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    try
    {
        TLLM_CHECK(outputIndex < mNumLoraModules);
        int const nbDimsA = inputs[getInputTensorIdx()].nbDims;
        DimsExprs ret;
        ret.nbDims = nbDimsA;

        for (int i = 0; i < ret.nbDims; ++i)
        {
            ret.d[0] = 0;
        }

        if (mTransA)
        {
            for (int i = 1; i < nbDimsA; ++i)
            {
                ret.d[i - 1] = inputs[getInputTensorIdx()].d[i];
            }
        }
        else
        {
            for (int i = 0; i < nbDimsA - 1; ++i)
            {
                ret.d[i] = inputs[getInputTensorIdx()].d[i];
            }
        }

        auto const* outHiddenSize = exprBuilder.constant(mOutHiddenSizes.at(outputIndex));
        TLLM_CHECK(outHiddenSize != nullptr);
        ret.d[ret.nbDims - 1] = outHiddenSize;
        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool LoraPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    if (pos == getHostRequestTypesIdx())
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else if (pos >= getLoraRanksIdx() && pos < getLoraRanksIdx() + mNumLoraModules)
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else if (pos >= getLoraWeightsPtrsIdx() && pos < getLoraWeightsPtrsIdx() + mNumLoraModules)
    {
        return inOut[pos].type == nvinfer1::DataType::kINT64;
    }
    else if (mRemoveInputPadding && pos == getHostContextLengthsIdx())
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else
    {
        return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
}

void LoraPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);

    auto const input = in[getInputTensorIdx()];

    int const nbDimsA = input.max.nbDims;

    auto const minM = utils::computeMDimension(mTransA, input.min);
    auto const maxM = utils::computeMDimension(mTransA, input.max);
    auto const N = utils::computeNDimension(mTransB, in[getHostRequestTypesIdx()].max);
    auto const K = static_cast<utils::DimType64>(mTransA ? input.max.d[0] : input.max.d[nbDimsA - 1]);

    if (!mDims.isInitialized())
    {
        mDims = {minM, maxM, N, K};
    }
    mGemmId.n = N;
    mGemmId.k = K;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

size_t LoraPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);

    int const nbReq = inputs[getLoraRanksIdx()].dims.d[0];
    auto const type = inputs[getInputTensorIdx()].type;
    auto const numTokens = getNumTokens(inputs);
    return mLoraImpl->getWorkspaceSize(numTokens, nbReq, type);
}

int64_t LoraPlugin::getNumTokens(nvinfer1::PluginTensorDesc const* input_tensors) const
{
    int ndim = input_tensors[getInputTensorIdx()].dims.nbDims;
    TLLM_CHECK_WITH_INFO(
        3 == ndim || 2 == ndim, "hidden_state dimension should be either 2 [numTokens, hidden], or 3 [b, s, hidden]");
    int64_t num_tokens = input_tensors[getInputTensorIdx()].dims.d[0];
    if (ndim == 3)
    {
        num_tokens *= input_tensors[getInputTensorIdx()].dims.d[1];
    }
    return num_tokens;
}

int LoraPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (isBuilding())
    {
        return 0;
    }

    auto const numReqs = inputDesc[getLoraRanksIdx()].dims.d[0];
    void const* input = inputs[getInputTensorIdx()];
    int const seqLen = mRemoveInputPadding ? 0 : inputDesc[getInputTensorIdx()].dims.d[1];
    int32_t const* reqTypes = static_cast<int32_t const*>(inputs[getHostRequestTypesIdx()]);
    void const* const* loraRanks = &inputs[getLoraRanksIdx()];
    void const* const* loraWeightPtrs = &inputs[getLoraWeightsPtrsIdx()];
    int32_t const* hostContextLengths
        = mRemoveInputPadding ? static_cast<int32_t const*>(inputs[getHostContextLengthsIdx()]) : nullptr;

    int numTokens = getNumTokens(inputDesc);
    mExpandLoraWeightPtrs.clear();
    mExpandLoraRanks.clear();
    mExpandLoraWeightPtrs.reserve(mNumLoraModules * numTokens * 2);
    mExpandLoraRanks.reserve(mNumLoraModules * numTokens);

    for (int loraModuleIdx = 0; loraModuleIdx < mNumLoraModules; loraModuleIdx++)
    {
        auto const loraWeightModulePtrs = static_cast<int64_t const*>(loraWeightPtrs[loraModuleIdx]);
        auto const loraRankModule = static_cast<int32_t const*>(loraRanks[loraModuleIdx]);

        int idx = 0;
        for (int reqId = 0; reqId < numReqs; reqId++)
        {
            // loraWeightModulePtrs has 3 pointers for each module: A,B, and an optional DoRA magnitude
            // the current LoRA plugin does not apply DoRA scaling, so the magnitude is ignored
            RequestType const reqType = static_cast<RequestType>(reqTypes[reqId]);
            if (reqType == RequestType::kGENERATION)
            {
                mExpandLoraWeightPtrs.push_back(reinterpret_cast<void const*>(loraWeightModulePtrs[reqId * 3]));
                mExpandLoraWeightPtrs.push_back(reinterpret_cast<void const*>(loraWeightModulePtrs[reqId * 3 + 1]));
                mExpandLoraRanks.push_back(loraRankModule[reqId]);
                idx += 1;
            }
            else
            {
                int contextLen = (mRemoveInputPadding ? hostContextLengths[reqId] : seqLen);

                for (int contextId = 0; contextId < contextLen; contextId++)
                {
                    mExpandLoraWeightPtrs.push_back(reinterpret_cast<void const*>(loraWeightModulePtrs[reqId * 3]));
                    mExpandLoraWeightPtrs.push_back(reinterpret_cast<void const*>(loraWeightModulePtrs[reqId * 3 + 1]));
                    mExpandLoraRanks.push_back(loraRankModule[reqId]);
                    idx += 1;
                }
            }
        }

        // In 1st generation phase cross attention qkv lora, cross qkv is skipped by passing an empty encoder_output
        // (passing 0 to dim) getNumTokens() will get in cross qkv_lora. Skipping the check for this case.
        if (numTokens > 0)
        {
            TLLM_CHECK_WITH_INFO(idx == numTokens,
                fmtstr("LoraParams and input dims don't match, lora tokens %d input tokens %d", idx, numTokens));
        }
    }

    // only used for unified gemm
    auto bestTactic = mPluginProfiler->getBestConfig(numTokens, mGemmId);
    mLoraImpl->setBestTactic(bestTactic);
    mLoraImpl->run(numTokens, numReqs, input, mExpandLoraRanks.data(), mExpandLoraWeightPtrs.data(), mWeightIndex,
        outputs, workspace, stream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType LoraPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    TLLM_CHECK(index < mNumLoraModules);
    return mType;
}

// IPluginV2 Methods

char const* LoraPlugin::getPluginType() const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    return LORA_PLUGIN_NAME;
}

char const* LoraPlugin::getPluginVersion() const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    return LORA_PLUGIN_VERSION;
}

int LoraPlugin::getNbOutputs() const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    return mNumLoraModules;
}

int LoraPlugin::initialize() noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    if (!mDims.isInitialized())
    {
        return 0;
    }

    mLoraImpl->setGemmConfig();

    mPluginProfiler->profileTactics(mLoraImpl->getCublasWrapper(), mType, mDims, mGemmId);
    return 0;
}

void LoraPlugin::destroy() noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    delete this;
}

size_t LoraPlugin::getSerializationSize() const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    return sizeof(mInHiddenSize) + sizeof(mTransA) + sizeof(mTransB) + sizeof(mNumLoraModules) + sizeof(mType)
        + mPluginProfiler->getSerializationSize(mGemmId) + sizeof(mRemoveInputPadding) + sizeof(mMaxLowRank)
        + sizeof(mWeightIndex) + sizeof(int) * mNumLoraModules; // selected tactics container size
}

void LoraPlugin::serialize(void* buffer) const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mInHiddenSize);
    write(d, mTransA);
    write(d, mTransB);
    write(d, mNumLoraModules);
    write(d, mType);
    write(d, mRemoveInputPadding);
    write(d, mMaxLowRank);
    write(d, mWeightIndex);
    for (int i = 0; i < mNumLoraModules; i++)
    {
        write(d, mOutHiddenSizes.at(i));
    }
    mPluginProfiler->serialize(d, mGemmId);
    TLLM_CHECK(d == a + getSerializationSize());
}

void LoraPlugin::terminate() noexcept {}

///////////////

LoraPluginCreator::LoraPluginCreator()
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("transA", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("transB", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("num_lora_modules", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("weight_index", nullptr, PluginFieldType::kINT32));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* LoraPluginCreator::getPluginName() const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    return LORA_PLUGIN_NAME;
}

char const* LoraPluginCreator::getPluginVersion() const noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    return LORA_PLUGIN_VERSION;
}

PluginFieldCollection const* LoraPluginCreator::getFieldNames() noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    return &mFC;
}

IPluginV2* LoraPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);

    PluginField const* fields = fc->fields;
    nvinfer1::DataType type{};
    int num_lora_modules{};
    int in_hidden_size{};
    int transA{};
    int transB{};
    bool remove_input_padding{};
    int max_low_rank{};
    int weight_index{};
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "in_hidden_size"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            in_hidden_size = *(static_cast<int32_t const*>(fields[i].data));
        }
        else if (!strcmp(attrName, "transa"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            transA = *(static_cast<int const*>(fields[i].data));
        }
        else if (!strcmp(attrName, "transb"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            transB = *(static_cast<int const*>(fields[i].data));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "remove_input_padding"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            remove_input_padding = static_cast<bool>(*(static_cast<int8_t const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "max_low_rank"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            max_low_rank = *(static_cast<int const*>(fields[i].data));
        }
        else if (!strcmp(attrName, "num_lora_modules"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            num_lora_modules = *(static_cast<int const*>(fields[i].data));
        }
        else if (!strcmp(attrName, "weight_index"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            weight_index = *(static_cast<int const*>(fields[i].data));
        }
    }
    std::vector<int> out_hidden_sizes;
    out_hidden_sizes.resize(num_lora_modules);
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        for (int j = 0; j < num_lora_modules; j++)
        {
            if (!strcmp(attrName, fmtstr("out_hidden_size_%d", j).c_str()))
            {
                TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
                out_hidden_sizes.at(j) = *(static_cast<int const*>(fields[i].data));
            }
        }
    }
    try
    {
        // LoraPluginCreator is unique and shared for an engine generation
        // Create plugin profiler with shared tactics map
        // FIXME enable tactic profiler
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler(/* inference */ false, /* skip */ true);
        auto* obj = new LoraPlugin(in_hidden_size, out_hidden_sizes, transA, transB, num_lora_modules, type,
            pluginProfiler, remove_input_padding, max_low_rank, weight_index);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* LoraPluginCreator::deserializePlugin(char const* name, void const* serialData, size_t serialLength) noexcept
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
    // This object will be deleted when the network is destroyed, which will
    // call LoraPlugin::destroy()
    try
    {
        // LoraPluginCreator is unique and shared for an engine generation
        // Create plugin profiler with shared tactics map
        // FIXME enable tactic profiler
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler(/* inference */ true, /* skip */ true);
        auto* obj = new LoraPlugin(serialData, serialLength, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
