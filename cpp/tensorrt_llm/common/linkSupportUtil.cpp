#include "tensorrt_llm/common/linkSupportUtil.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/mcastDeviceMemory.h"
#include "tensorrt_llm/runtime/utils/mpiTags.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <map>
#include <nccl.h>
#include <nvml.h>

using namespace tensorrt_llm::mpi;

namespace tensorrt_llm::common
{
std::set<int> getLocalGroup(std::set<int> const& group)
{
    auto const myRank = COMM_SESSION.getRank();
    auto const myLocalRank = LOCAL_COMM_SESSION.getRank();
    auto const localSize = static_cast<uint32_t>(LOCAL_COMM_SESSION.getSize());

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
                LOCAL_COMM_SESSION.recvValue(rank, *it, MpiTag::kDefault);
                ranks.push_back(rank);
            }
            for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
            {
                LOCAL_COMM_SESSION.send(
                    ranks.data(), localSize, tensorrt_llm::mpi::MpiType::kINT32, *it, MpiTag::kDefault);
            }

            localRanks.clear();
            localRanks.push_back(myLocalRank);
            for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
            {
                LOCAL_COMM_SESSION.recvValue(rank, *it, MpiTag::kDefault);
                localRanks.push_back(rank);
            }
            for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
            {
                LOCAL_COMM_SESSION.send(
                    localRanks.data(), localSize, tensorrt_llm::mpi::MpiType::kINT32, *it, MpiTag::kDefault);
            }
        }
        else
        {
            LOCAL_COMM_SESSION.sendValue(myRank, *group.begin(), MpiTag::kDefault);
            LOCAL_COMM_SESSION.recv(
                ranks.data(), localSize, tensorrt_llm::mpi::MpiType::kINT32, *group.begin(), MpiTag::kDefault);

            LOCAL_COMM_SESSION.sendValue(myLocalRank, *group.begin(), MpiTag::kDefault);
            LOCAL_COMM_SESSION.recv(
                localRanks.data(), localSize, tensorrt_llm::mpi::MpiType::kINT32, *group.begin(), MpiTag::kDefault);
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

std::tuple<bool, bool> setGroupTopology(std::set<int> group)
{
    auto const rank = COMM_SESSION.getRank();
    TLLM_LOG_INFO("Detecting local TP group for rank %d", rank);
    std::set<int> local_group = getLocalGroup(group);
    if (group.size() != local_group.size())
    {
        TLLM_LOG_INFO("Found inter-node TP group for rank %d", rank);
        return {false, false};
    }
    TLLM_LOG_INFO("TP group is intra-node for rank %d", rank);

    std::unordered_set<int> visited_device;

    bool is_P2P_supported = true;
    bool is_NVLINK_supported = true;

    // Use cudaDeviceCanAccessPeer to determine whether p2p is supported,
    // and use nvml to determine whether there are nvlink links between ranks.
    for (int first_device_id : local_group)
    {
        for (int second_device_id : local_group)
        {
            if (first_device_id == second_device_id || visited_device.find(second_device_id) != visited_device.end())
            {
                continue;
            }

            int can_access_peer = 0;
            TLLM_CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, first_device_id, second_device_id));

            if (!can_access_peer)
            {
                is_P2P_supported = false;
                is_NVLINK_supported = false;

                return {is_NVLINK_supported, is_P2P_supported};
            }

            nvmlDevice_t first_device;
            NVML_CHECK_THROW(nvmlDeviceGetHandleByIndex(first_device_id, &first_device));

            bool is_NVLINK = false;

            for (unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; link++)
            {
                nvmlPciInfo_t remote_pci_info;
                if (nvmlDeviceGetNvLinkRemotePciInfo_v2(first_device, link, &remote_pci_info) != NVML_SUCCESS)
                {
                    continue;
                }

                nvmlDevice_t remote_device;
                auto const result = nvmlDeviceGetHandleByPciBusId_v2(remote_pci_info.busId, &remote_device);

                if (result == NVML_SUCCESS)
                {
                    // Two GPUs are connected directly through nvlink
                    unsigned int remote_device_id;
                    NVML_CHECK_THROW(nvmlDeviceGetIndex(remote_device, &remote_device_id));

                    if (remote_device_id == static_cast<unsigned int>(second_device_id))
                    {
                        is_NVLINK = true;
                    }
                }
                else if (result == NVML_ERROR_NOT_FOUND)
                {
                    // Maybe Two GPUs are connected via nvswitch,
                    // now remotePciInfo represents the pci information of nvswitch,
                    // determine whether nvlink is supported by whether two GPUs are connected to the same
                    // nvswitch.
                    nvmlDevice_t second_device;
                    NVML_CHECK_THROW(nvmlDeviceGetHandleByIndex(second_device_id, &second_device));

                    for (unsigned int second_link = 0; second_link < NVML_NVLINK_MAX_LINKS; second_link++)
                    {
                        nvmlPciInfo_t second_remote_pci_info;
                        if (nvmlDeviceGetNvLinkRemotePciInfo_v2(second_device, second_link, &second_remote_pci_info)
                            != NVML_SUCCESS)
                        {
                            continue;
                        }

                        if (strcmp(remote_pci_info.busId, second_remote_pci_info.busId) == 0)
                        {
                            is_NVLINK = true;
                            break;
                        }
                    }
                }
                else
                {
                    NVML_CHECK_THROW(result);
                }

                if (is_NVLINK)
                {
                    break;
                }
            }

            is_NVLINK_supported &= is_NVLINK;
        }
        visited_device.insert(first_device_id);
    }
    return {is_NVLINK_supported, is_P2P_supported};
}

std::vector<bool> initGroupTopology(std::set<int> group)
{
    static std::map<std::set<int>, std::tuple<bool, bool>> cache;
    if (cache.find(group) != cache.end())
    {
        auto [is_NVLINK_supported, is_P2P_supported] = cache[group];
        return std::vector{is_NVLINK_supported, is_P2P_supported};
    }
    auto link_supported = setGroupTopology(group);
    cache[group] = link_supported;
    return {std::get<0>(link_supported), std::get<1>(link_supported)};
}

} // namespace tensorrt_llm::common
