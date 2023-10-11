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
#include "allreducePlugin.h"
#include "mpi.h"
#include "plugin.h"

using namespace nvinfer1;
using tensorrt_llm::plugins::AllreducePluginCreator;
using tensorrt_llm::plugins::AllreducePlugin;

static const char* ALLREDUCE_PLUGIN_VERSION{"1"};
static const char* ALLREDUCE_PLUGIN_NAME{"AllReduce"};
PluginFieldCollection AllreducePluginCreator::mFC{};
std::vector<nvinfer1::PluginField> AllreducePluginCreator::mPluginAttributes;

AllreducePlugin::AllreducePlugin(std::set<int> group, nvinfer1::DataType type, AllReduceStrategyType strategy)
    : mGroup(group)
    , mType(type)
    , mStrategy(strategy)
    , mCustomARBufferSize(0)
{
}

// Parameterized constructor
AllreducePlugin::AllreducePlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mType);
    read(d, mStrategy);
    read(d, mCustomARBufferSize);
    mGroup.clear();
    int groupItem = 0;
    while (d != a + length)
    {
        read(d, groupItem);
        mGroup.insert(groupItem);
    }
    TLLM_CHECK(d == a + length);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* AllreducePlugin::clone() const noexcept
{
    auto* plugin = new AllreducePlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs AllreducePlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    return inputs[0];
}

bool AllreducePlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
}

void AllreducePlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    if (mStrategy == AllReduceStrategyType::NCCL)
    {
        return;
    }

    size_t sizePerElem = 0;
    switch (mType)
    {
    case DataType::kFLOAT: sizePerElem = sizeof(float); break;
    case DataType::kHALF: sizePerElem = sizeof(half); break;
#ifdef ENABLE_BF16
    case DataType::kBF16: sizePerElem = sizeof(__nv_bfloat16); break;
#endif
    default: break;
    }

    size_t inputSize = 1;
    for (int i = 0; i < in[0].max.nbDims; i++)
    {
        inputSize *= in[0].max.d[i];
    }

    if (isBuilding())
    {
        mCustomARBufferSize = std::max(mCustomARBufferSize, inputSize * sizePerElem);
    }
}

size_t AllreducePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

size_t AllreducePlugin::ncclEstimatedThreshold(int worldSize) const noexcept
{
    // returns the message size over which it's more interesting to use NCCL
    // 0.60 * TP_SIZE * 10MB
    return 0.60 * (10 * 1000 * 1000 * worldSize);
}

int AllreducePlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }
    int size = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        size *= inputDesc[0].dims.d[i];
    }
    size_t sizePerElem = 0;
    switch (mType)
    {
    case DataType::kFLOAT: sizePerElem = sizeof(float); break;
    case DataType::kHALF: sizePerElem = sizeof(half); break;
#ifdef ENABLE_BF16
    case DataType::kBF16: sizePerElem = sizeof(__nv_bfloat16); break;
#endif
    default: break;
    }

    auto runtimeStrategy = mStrategy;
    if (runtimeStrategy == AllReduceStrategyType::AUTO)
    {
        runtimeStrategy = size * sizePerElem > ncclEstimatedThreshold(mGroup.size()) ? AllReduceStrategyType::NCCL
                                                                                     : AllReduceStrategyType::CUSTOM;
    }

    if (runtimeStrategy == AllReduceStrategyType::NCCL)
    {
        NCCLCHECK(ncclAllReduce(inputs[0], outputs[0], size, (*getDtypeMap())[inputDesc[0].type], ncclSum,
            (*getCommMap())[mGroup], stream));
    }
    else if (runtimeStrategy == AllReduceStrategyType::CUSTOM)
    {
        auto shareBuffer = mCustomAllReduceContext->getShareBuffer();
        cudaMemcpyAsync(shareBuffer, inputs[0], size * sizePerElem, cudaMemcpyDeviceToDevice, stream);

        mCustomAllReduceContext->customAllReduce(outputs[0], size, sizePerElem, mType, stream);
    }

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType AllreducePlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

const char* AllreducePlugin::getPluginType() const noexcept
{
    return ALLREDUCE_PLUGIN_NAME;
}

const char* AllreducePlugin::getPluginVersion() const noexcept
{
    return ALLREDUCE_PLUGIN_VERSION;
}

int AllreducePlugin::getNbOutputs() const noexcept
{
    return 1;
}

bool AllreducePlugin::isCustomAllReduceSuported(int ranks_per_node) const noexcept
{
    constexpr bool isCudaVersionSupported =
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
        true;
#else
        false;
#endif

    return isCudaVersionSupported && (ranks_per_node % 2 == 0) && (ranks_per_node <= MAX_RANKS_PER_NODE)
        && (ranks_per_node > 0);
}

int AllreducePlugin::initialize() noexcept
{
    if (isBuilding())
    {
        return 0;
    }

    int myRank, nRanks;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    int deviceId;
    cudaGetDevice(&deviceId);
    TLLM_CHECK_WITH_INFO(myRank == deviceId, "MPI rank != cudaDeviceId, check if cudaSetDevice has been called");

    if (mStrategy == AllReduceStrategyType::NCCL || mStrategy == AllReduceStrategyType::AUTO)
    {
        auto* commMap = getCommMap();
        // [] operator inserts T() if it does not exist
        if ((*commMap)[mGroup] == nullptr)
        {
            int groupRank = 0;
            for (auto it = mGroup.begin(); it != mGroup.end(); ++it)
            {
                if (*it == myRank)
                {
                    break;
                }
                ++groupRank;
            }

            ncclUniqueId id;
            if (myRank == *mGroup.begin())
            {
                ncclGetUniqueId(&id);
                for (auto it = std::next(std::begin(mGroup), 1); it != mGroup.end(); ++it)
                {
                    MPICHECK(MPI_Send(&id, sizeof(id), MPI_BYTE, *it, 0, MPI_COMM_WORLD));
                }
            }
            else
            {
                MPI_Status status;
                MPICHECK(MPI_Recv(&id, sizeof(id), MPI_BYTE, *mGroup.begin(), 0, MPI_COMM_WORLD, &status));
            }
            (*commMap)[mGroup] = nullptr;
            NCCLCHECK(ncclCommInitRank(&((*commMap)[mGroup]), mGroup.size(), id, groupRank));
        }
    }

    if (mStrategy == AllReduceStrategyType::CUSTOM || mStrategy == AllReduceStrategyType::AUTO)
    {
        TLLM_CHECK_WITH_INFO(tensorrt_llm::CustomAllReduceComm::isAvailable(), "Custom all reduce isn't available.");
        auto allocSize = mCustomARBufferSize > 0 ? mCustomARBufferSize : CUSTOM_AR_SIZE_THRESHOLD;
        mCustomAllReduceContext = std::make_shared<tensorrt_llm::CustomAllReduceComm>(nRanks, 0, myRank, allocSize);
    }

    return 0;
}

void AllreducePlugin::terminate() noexcept
{
    if (mStrategy == AllReduceStrategyType::NCCL || mStrategy == AllReduceStrategyType::AUTO)
    {
        auto* commMap = getCommMap();
        // [] operator inserts T() if it does not exist
        if (isBuilding() || (*commMap)[mGroup] == nullptr)
        {
            return;
        }
        NCCLCHECK(ncclCommDestroy((*commMap)[mGroup]));
        (*commMap)[mGroup] = nullptr;
    }
}

size_t AllreducePlugin::getSerializationSize() const noexcept
{
    return sizeof(int) * mGroup.size() + sizeof(mType) + sizeof(mStrategy) + sizeof(mCustomARBufferSize);
}

void AllreducePlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    write(d, mStrategy);
    write(d, mCustomARBufferSize);
    for (auto it = mGroup.begin(); it != mGroup.end(); ++it)
    {
        write(d, *it);
    }
    assert(d == a + getSerializationSize());
}

void AllreducePlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

AllreducePluginCreator::AllreducePluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("group", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("strategy", nullptr, PluginFieldType::kINT8, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* AllreducePluginCreator::getPluginName() const noexcept
{
    return ALLREDUCE_PLUGIN_NAME;
}

const char* AllreducePluginCreator::getPluginVersion() const noexcept
{
    return ALLREDUCE_PLUGIN_VERSION;
}

const PluginFieldCollection* AllreducePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* AllreducePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    std::set<int> group;
    nvinfer1::DataType type;
    AllreducePlugin::AllReduceStrategyType strategy;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "group"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            const auto* r = static_cast<const int*>(fields[i].data);
            for (int j = 0; j < fields[i].length; ++j)
            {
                group.insert(*r);
                ++r;
            }
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "strategy"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            strategy = static_cast<AllreducePlugin::AllReduceStrategyType>(*static_cast<const int8_t*>(fields[i].data));
        }
    }

    try
    {
        auto* obj = new AllreducePlugin(group, type, strategy);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* AllreducePluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call AllreducePlugin::destroy()
    try
    {
        auto* obj = new AllreducePlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
