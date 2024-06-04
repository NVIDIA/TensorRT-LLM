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

#include "tensorrt_llm/common/customAllReduceUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/kernels/customAllReduceKernels.h"
#include <nccl.h>
#include <unordered_set>

using namespace nvinfer1;
using tensorrt_llm::plugins::AllreducePluginCreator;
using tensorrt_llm::plugins::AllreducePlugin;
using tensorrt_llm::kernels::AllReduceStrategyType;
using tensorrt_llm::kernels::AllReduceStrategyConfig;

static char const* ALLREDUCE_PLUGIN_VERSION{"1"};
static char const* ALLREDUCE_PLUGIN_NAME{"AllReduce"};
PluginFieldCollection AllreducePluginCreator::mFC{};
std::vector<nvinfer1::PluginField> AllreducePluginCreator::mPluginAttributes;

AllreducePlugin::AllreducePlugin(std::set<int> group, nvinfer1::DataType type, AllReduceStrategyType strategy,
    AllReduceStrategyConfig config, int32_t counter)
    : mGroup(std::move(group))
    , mType(type)
    , mStrategy(strategy)
    , mConfig(config)
    , mCounter(counter)
{
}

// Parameterized constructor
AllreducePlugin::AllreducePlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mType);
    read(d, mStrategy);
    read(d, mConfig);
    read(d, mCounter);
    mGroup.clear();
    int groupItem = 0;
    while (d != a + length)
    {
        read(d, groupItem);
        mGroup.insert(groupItem);
    }
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* AllreducePlugin::clone() const noexcept
{
    auto* plugin = new AllreducePlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs AllreducePlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    return inputs[0];
}

bool AllreducePlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (mStrategy == AllReduceStrategyType::NCCL)
    {
        TLLM_CHECK_WITH_INFO(nbInputs == 1, "NCCL strategy only accepts one input.");
    }
    else
    {
        TLLM_CHECK_WITH_INFO(nbInputs == 2, "Non-NCCL strategies require a workspace tensor.");
    }

    if (nbInputs == 2 && pos == 1)
    {
        return (inOut[pos].type == nvinfer1::DataType::kINT64) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else
    {
        return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
}

void AllreducePlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t AllreducePlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

AllReduceStrategyType AllreducePlugin::selectImplementation(
    size_t messageSize, int worldSize, nvinfer1::DataType type) noexcept
{
    bool const isAuto = (mStrategy == AllReduceStrategyType::AUTO);

    if (!mIsP2PSupported)
    {
        if (!isAuto)
        {
            TLLM_LOG_WARNING("Since Peer to Peer not supported, fallback to AllReduceStrategy: NCCL");
        }
        return AllReduceStrategyType::NCCL;
    }

    if (isAuto && !mIsNVLINKSupported)
    {
        return AllReduceStrategyType::NCCL;
    }

    auto const maxWorkspaceSize = utils::customAllReduceUtils::getMaxRequiredWorkspaceSize(worldSize);

    AllReduceStrategyType strat = AllReduceStrategyType::NCCL;
    auto const messageSizeBytes = messageSize * common::getDTypeSize(type);

    if (messageSizeBytes <= maxWorkspaceSize)
    {
        if (!isAuto)
        {
            strat = mStrategy;
        }
        else if (worldSize <= 2)
        {
            strat = AllReduceStrategyType::ONESHOT;
        }
        else if (worldSize <= 4)
        {
            if (messageSizeBytes < 1 * 1000 * 1000)
            {
                strat = AllReduceStrategyType::ONESHOT;
            }
            else
            {
                strat = AllReduceStrategyType::TWOSHOT;
            }
        }
        else
        {
            if (messageSizeBytes < 500 * 1000)
            {
                strat = AllReduceStrategyType::ONESHOT;
            }
            else
            {
                strat = AllReduceStrategyType::TWOSHOT;
            }
        }

        if (!kernels::configurationSupported(strat, messageSize, worldSize, type))
        {
            if (!isAuto)
            {
                TLLM_LOG_WARNING("Since not alignment, fallback to AllReduceStrategy: NCCL");
            }
            strat = AllReduceStrategyType::NCCL;
        }
    }
    else
    {
        if (!isAuto)
        {
            TLLM_LOG_WARNING("Since messageSize > maxWorkspace, fallback to AllReduceStrategy: NCCL");
        }
        strat = AllReduceStrategyType::NCCL;
    }

    return strat;
}

int AllreducePlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }
    size_t size = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
    {
        size *= inputDesc[0].dims.d[i];
    }
    auto const sizePerElem = common::getDTypeSize(mType);

    kernels::AllReduceStrategyType runtimeStrategy;

    if (mStrategy == AllReduceStrategyType::NCCL)
    {
        runtimeStrategy = AllReduceStrategyType::NCCL;
    }
    else
    {
        runtimeStrategy = selectImplementation(size, mGroup.size(), mType);
    }

    // Log runtime strategy
    switch (runtimeStrategy)
    {
    case AllReduceStrategyType::NCCL:
    {
        TLLM_LOG_DEBUG("AllReducePlugin strategy: AllReduceStrategyType::NCCL");
        break;
    }
    case AllReduceStrategyType::ONESHOT:
    {
        TLLM_LOG_DEBUG("AllReducePlugin strategy: AllReduceStrategyType::ONESHOT");
        break;
    }
    case AllReduceStrategyType::TWOSHOT:
    {
        TLLM_LOG_DEBUG("AllReducePlugin strategy: AllReduceStrategyType::TWOSHOT");
        break;
    }
    default: break;
    }

    if (runtimeStrategy == AllReduceStrategyType::NCCL)
    {
        NCCLCHECK(ncclAllReduce(
            inputs[0], outputs[0], size, (*getDtypeMap())[mType], ncclSum, (*getCommMap())[mGroup], stream));
    }
    else
    {
        auto myRank = COMM_SESSION.getRank();
        int nRanks = inputDesc[1].dims.d[0] / utils::customAllReduceUtils::NUM_POINTERS_PER_RANK;
        // FIXME: pass world config here
        myRank = myRank % nRanks;

        auto params = tensorrt_llm::kernels::AllReduceParams::deserialize(
            reinterpret_cast<int32_t const*>(inputs[1]), nRanks, myRank, mCounter);

        params.local_output_buffer_ptr = outputs[0];
        params.local_input_buffer_ptr = inputs[0];
        params.elts_total = size;
        tensorrt_llm::kernels::customAllReduce(params, mType, runtimeStrategy, mConfig, stream);
    }

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType AllreducePlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

char const* AllreducePlugin::getPluginType() const noexcept
{
    return ALLREDUCE_PLUGIN_NAME;
}

char const* AllreducePlugin::getPluginVersion() const noexcept
{
    return ALLREDUCE_PLUGIN_VERSION;
}

int AllreducePlugin::getNbOutputs() const noexcept
{
    return 1;
}

bool AllreducePlugin::isCustomAllReduceSupported(int ranks_per_node) const noexcept
{
    constexpr bool isCudaVersionSupported =
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
        true;
#else
        false;
#endif

    return isCudaVersionSupported && (ranks_per_node % 2 == 0) && (ranks_per_node <= kernels::MAX_RANKS_PER_NODE)
        && (ranks_per_node > 0);
}

class NvmlManager
{
public:
    NvmlManager()
    {
        NVML_CHECK(nvmlInit());
    }

    ~NvmlManager()
    {
        NVML_CHECK(nvmlShutdown());
    }
};

std::set<int> getLocalGroup(std::set<int> const& group)
{
    auto const myRank = COMM_SESSION.getRank();
    auto const myLocalRank = LOCAL_COMM_SESSION.getRank();
    auto const localSize = LOCAL_COMM_SESSION.getSize();

    std::vector<int32_t> ranks(localSize, 0);
    std::vector<int32_t> localRanks(localSize, 0);
    if (group.size() >= localSize)
    {
        LOCAL_COMM_SESSION.allgather(&myRank, ranks.data(), 1, tensorrt_llm::mpi::MpiType::kINT32);
        LOCAL_COMM_SESSION.allgather(&myLocalRank, localRanks.data(), 1, tensorrt_llm::mpi::MpiType::kINT32);
    }
    else
    {
        if (myRank == *group.begin())
        {
            ranks.clear();
            int rank;
            ranks.push_back(myRank);
            for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
            {
                LOCAL_COMM_SESSION.recvValue(rank, *it, 0);
                ranks.push_back(rank);
            }
            for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
            {
                LOCAL_COMM_SESSION.send(ranks.data(), localSize, tensorrt_llm::mpi::MpiType::kINT32, *it, 0);
            }

            localRanks.clear();
            localRanks.push_back(myLocalRank);
            for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
            {
                LOCAL_COMM_SESSION.recvValue(rank, *it, 0);
                localRanks.push_back(rank);
            }
            for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
            {
                LOCAL_COMM_SESSION.send(localRanks.data(), localSize, tensorrt_llm::mpi::MpiType::kINT32, *it, 0);
            }
        }
        else
        {
            LOCAL_COMM_SESSION.sendValue(myRank, *group.begin(), 0);
            LOCAL_COMM_SESSION.recv(ranks.data(), localSize, tensorrt_llm::mpi::MpiType::kINT32, *group.begin(), 0);

            LOCAL_COMM_SESSION.sendValue(myLocalRank, *group.begin(), 0);
            LOCAL_COMM_SESSION.recv(
                localRanks.data(), localSize, tensorrt_llm::mpi::MpiType::kINT32, *group.begin(), 0);
        }
    }

    std::set<int> localGroup;
    for (size_t i = 0; i < ranks.size(); ++i)
    {
        auto rank = ranks[i];
        if (group.find(rank) != group.end())
        {
            localGroup.insert(localRanks[i]);
        }
    }
    return localGroup;
}

void AllreducePlugin::initGroupTopology() noexcept
{
    static std::map<std::set<int>, std::tuple<bool, bool>> cache;
    if (cache.find(mGroup) != cache.end())
    {
        auto [isNVLINKSupported, isP2PSupported] = cache[mGroup];
        mIsNVLINKSupported = isNVLINKSupported;
        mIsP2PSupported = isP2PSupported;
        return;
    }
    setGroupTopology();
    cache[mGroup] = {mIsNVLINKSupported, mIsP2PSupported};
}

void AllreducePlugin::setGroupTopology() noexcept
{
    auto const rank = COMM_SESSION.getRank();
    TLLM_LOG_INFO("Detecting local TP group for rank %d", rank);
    std::set<int> localGroup = getLocalGroup(mGroup);
    if (mGroup.size() != localGroup.size())
    {
        mIsP2PSupported = false;
        mIsNVLINKSupported = false;
        TLLM_LOG_INFO("Found inter-node TP group for rank %d", rank);
        return;
    }
    TLLM_LOG_INFO("TP group is intra-node for rank %d", rank);

    NvmlManager nvmlManager;
    std::unordered_set<int> visitedDevice;
    mIsP2PSupported = true;
    mIsNVLINKSupported = true;

    // Use cudaDeviceCanAccessPeer to determine whether p2p is supported,
    // and use nvml to determine whether there are nvlink links between ranks.
    for (int firstDeviceId : localGroup)
    {
        for (int secondDeviceId : localGroup)
        {
            if (firstDeviceId == secondDeviceId || visitedDevice.find(secondDeviceId) != visitedDevice.end())
            {
                continue;
            }

            int canAccessPeer = 0;
            TLLM_CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, firstDeviceId, secondDeviceId));

            if (!canAccessPeer)
            {
                mIsP2PSupported = false;
                mIsNVLINKSupported = false;

                return;
            }

            nvmlDevice_t firstDevice;
            NVML_CHECK(nvmlDeviceGetHandleByIndex(firstDeviceId, &firstDevice));

            bool isNVLINK = false;

            for (unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; link++)
            {
                nvmlPciInfo_t remotePciInfo;
                if (nvmlDeviceGetNvLinkRemotePciInfo_v2(firstDevice, link, &remotePciInfo) != NVML_SUCCESS)
                {
                    continue;
                }

                nvmlDevice_t remoteDevice;
                auto const result = nvmlDeviceGetHandleByPciBusId_v2(remotePciInfo.busId, &remoteDevice);

                if (result == NVML_SUCCESS)
                {
                    // Two GPUs are connected directly through nvlink
                    unsigned int remoteDeviceId;
                    NVML_CHECK(nvmlDeviceGetIndex(remoteDevice, &remoteDeviceId));

                    if (remoteDeviceId == secondDeviceId)
                    {
                        isNVLINK = true;
                    }
                }
                else if (result == NVML_ERROR_NOT_FOUND)
                {
                    // Maybe Two GPUs are connected via nvswitch,
                    // now remotePciInfo represents the pci information of nvswitch,
                    // determine whether nvlink is supported by whether two GPUs are connected to the same nvswitch.
                    nvmlDevice_t secondDevice;
                    NVML_CHECK(nvmlDeviceGetHandleByIndex(secondDeviceId, &secondDevice));

                    for (unsigned int secondLink = 0; secondLink < NVML_NVLINK_MAX_LINKS; secondLink++)
                    {
                        nvmlPciInfo_t secondRemotePciInfo;
                        if (nvmlDeviceGetNvLinkRemotePciInfo_v2(secondDevice, secondLink, &secondRemotePciInfo)
                            != NVML_SUCCESS)
                        {
                            continue;
                        }

                        if (strcmp(remotePciInfo.busId, secondRemotePciInfo.busId) == 0)
                        {
                            isNVLINK = true;
                            break;
                        }
                    }
                }
                else
                {
                    NVML_CHECK(result);
                }

                if (isNVLINK)
                {
                    break;
                }
            }

            mIsNVLINKSupported &= isNVLINK;
        }
        visitedDevice.insert(firstDeviceId);
    }
}

int AllreducePlugin::initialize() noexcept
{
    if (isBuilding())
    {
        return 0;
    }

    initCommMap(mGroup);
    if (mStrategy != AllReduceStrategyType::NCCL)
    {
        initGroupTopology();
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
    return sizeof(int) * mGroup.size() + sizeof(mType) + sizeof(mStrategy) + sizeof(mConfig) + sizeof(mCounter);
}

void AllreducePlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    write(d, mStrategy);
    write(d, mConfig);
    write(d, mCounter);
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
    mPluginAttributes.emplace_back(PluginField("config", nullptr, PluginFieldType::kINT8, 1));
    mPluginAttributes.emplace_back(PluginField("counter", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* AllreducePluginCreator::getPluginName() const noexcept
{
    return ALLREDUCE_PLUGIN_NAME;
}

char const* AllreducePluginCreator::getPluginVersion() const noexcept
{
    return ALLREDUCE_PLUGIN_VERSION;
}

PluginFieldCollection const* AllreducePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* AllreducePluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    std::set<int> group;
    nvinfer1::DataType type;
    AllReduceStrategyType strategy;
    AllReduceStrategyConfig config;
    int32_t counter;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "group"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            auto const* r = static_cast<int const*>(fields[i].data);
            for (int j = 0; j < fields[i].length; ++j)
            {
                group.insert(*r);
                ++r;
            }
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "strategy"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            strategy = static_cast<AllReduceStrategyType>(*static_cast<int8_t const*>(fields[i].data));
        }
        else if (!strcmp(attrName, "config"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            config = static_cast<AllReduceStrategyConfig>(*static_cast<int8_t const*>(fields[i].data));
        }
        else if (!strcmp(attrName, "counter"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            counter = *static_cast<int32_t const*>(fields[i].data);
        }
    }

    try
    {
        auto* obj = new AllreducePlugin(group, type, strategy, config, counter);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* AllreducePluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call AllreducePlugin::destroy()
    try
    {
        auto* obj = new AllreducePlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
