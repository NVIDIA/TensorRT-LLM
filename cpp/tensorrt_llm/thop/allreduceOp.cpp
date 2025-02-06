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

#include "tensorrt_llm/common/customAllReduceUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/customAllReduceKernels.h"
#include "tensorrt_llm/kernels/userbuffers/ub_interface.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <nvml.h>
#include <torch/extension.h>
#include <unordered_set>
#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE

// using namespace nvinfer1;
using tensorrt_llm::kernels::AllReduceFusionOp;
using tensorrt_llm::kernels::AllReduceStrategyType;
using tensorrt_llm::kernels::AllReduceStrategyConfig;

namespace torch_ext
{

#if ENABLE_MULTI_DEVICE

namespace
{

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

class AllreduceOp
{
public:
    AllreduceOp(std::set<int> group, nvinfer1::DataType type, AllReduceStrategyType strategy,
        AllReduceStrategyConfig config, AllReduceFusionOp op, float eps, bool affine, bool bias, bool scale)
        : mGroup(std::move(group))
        , mType(type)
        , mStrategy(strategy)
        , mConfig(config)
        , mOp(op)
        , mEps(eps)
        , mAffine(affine)
        , mBias(bias)
        , mScale(scale)
    {
    }

    ~AllreduceOp() = default;

    std::tuple<torch::Tensor, torch::Tensor> run(
        torch::Tensor input, torch::optional<torch::Tensor> workspace, torch::TensorList reduce_fusion_inputs) noexcept
    {
        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
        torch::Tensor output;
        torch::Tensor finalOutput;
        size_t size = input.numel();
        auto const sizePerElem = tensorrt_llm::common::getDTypeSize(mType);

        AllReduceStrategyType runtimeStrategy;

        static char* forceNcclAllReduceStrategyChar = std::getenv("FORCE_NCCL_ALL_REDUCE_STRATEGY");
        bool forceNcclAllReduceStrategy = (forceNcclAllReduceStrategyChar != nullptr);
        // If strategy is set to UB, UB must be used as UB impl output is special and cannot be used
        // by others.
        if (mStrategy == AllReduceStrategyType::UB)
        {
            runtimeStrategy = AllReduceStrategyType::UB;
        }
        else if (forceNcclAllReduceStrategy || mStrategy == AllReduceStrategyType::NCCL)
        {
            runtimeStrategy = AllReduceStrategyType::NCCL;
        }
        else
        {
            runtimeStrategy = selectImplementation(size, mGroup.size(), mType);
        }

        // Log runtime strategy
        auto const rank = COMM_SESSION.getRank();
        switch (runtimeStrategy)
        {
        case AllReduceStrategyType::NCCL:
        {
            TLLM_LOG_DEBUG("AllReducePlugin strategy for rank %d: NCCL", rank);
            break;
        }
        case AllReduceStrategyType::ONESHOT:
        {
            TLLM_LOG_DEBUG("AllReducePlugin strategy for rank %d: ONESHOT", rank);
            break;
        }
        case AllReduceStrategyType::TWOSHOT:
        {
            TLLM_LOG_DEBUG("AllReducePlugin strategy for rank %d: TWOSHOT", rank);
            break;
        }
        case AllReduceStrategyType::UB:
        {
            TLLM_LOG_DEBUG("AllReducePlugin strategy for rank %d: UB", rank);
            break;
        }
        default: break;
        }

        if (runtimeStrategy == AllReduceStrategyType::UB)
        {
            output = torch::empty_like(input);
            // Only support fp8 fusion
            TLLM_CHECK(mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM);
            TLLM_CHECK(mScale);
            TLLM_CHECK(mAffine);
            TLLM_CHECK(!mBias);

            int hidden_size = input.size(-1);

            TLLM_CHECK_WITH_INFO(
                tensorrt_llm::runtime::ub::ub_is_initialized(), "UserBuffer has not been initialized!");
            auto ub_buffer0 = tensorrt_llm::runtime::ub::ub_get(0);
            auto ub_buffer1 = tensorrt_llm::runtime::ub::ub_get(1);
            TLLM_CHECK(input.data_ptr() == ub_buffer0.addr);
            auto ub_comm = tensorrt_llm::runtime::ub::ub_comm();

            void* residual = reduce_fusion_inputs[0].data_ptr();
            void* gamma = reduce_fusion_inputs[1].data_ptr();
            float* scale = static_cast<float*>(reduce_fusion_inputs[2].data_ptr());
            tensorrt_llm::kernels::ub::allreduce2_userbuff_inplace_rmsnorm_quant_launcher(ub_buffer0.handle, 0,
                ub_buffer1.handle, 0, size, hidden_size, nullptr, gamma, mEps, scale, residual, output.data_ptr(),
                mType, ub_comm, stream);
            finalOutput = torch::from_blob(ub_buffer1.addr, input.sizes(), input.strides(),
                torch::dtype(torch::kFloat8_e4m3fn).device(torch::kCUDA));
        }
        else if (runtimeStrategy == AllReduceStrategyType::NCCL)
        {
            output = torch::empty_like(input);
            if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
            {
                finalOutput = torch::empty_like(input);

                NCCLCHECK(ncclAllReduce(input.data_ptr(), output.mutable_data_ptr(), size, (*getDtypeMap())[mType],
                    ncclSum, *mNcclComm, stream));
                tensorrt_llm::kernels::AllReduceParams params;
                int fusion_ptr_idx = 0;
                params.fusion_params.bias_buffer = mBias ? reduce_fusion_inputs[fusion_ptr_idx++].data_ptr() : nullptr;
                params.fusion_params.residual_buffer = reduce_fusion_inputs[fusion_ptr_idx++].data_ptr();
                params.fusion_params.weight_buffer
                    = mAffine ? reduce_fusion_inputs[fusion_ptr_idx++].data_ptr() : nullptr;
                params.local_output_buffer_ptr = finalOutput.mutable_data_ptr();
                params.elts_total = size;
                params.fusion_params.hidden_size = input.size(-1);
                params.fusion_params.eps = mEps;
                params.fusion_params.intermediate_buffer = output.mutable_data_ptr();
                tensorrt_llm::kernels::residualRmsNorm(params, mType, stream, mOp);
            }
            else
            {
                NCCLCHECK(ncclAllReduce(input.data_ptr(), output.mutable_data_ptr(), size, (*getDtypeMap())[mType],
                    ncclSum, *mNcclComm, stream));
            }
        }
        else
        {
            auto const tpSize = mGroup.size();
            int tpRank = 0;
            output = torch::empty_like(input);
            for (auto const& currentRank : mGroup)
            {
                if (rank == currentRank)
                    break;
                ++tpRank;
            }

            int token_num = size / input.size(-1);
            int hidden_size = input.size(-1);
            auto workspace_ptr = workspace.value().mutable_data_ptr();
            auto params = tensorrt_llm::kernels::AllReduceParams::deserialize(
                reinterpret_cast<int64_t*>(workspace_ptr), tpSize, tpRank, mType, token_num, hidden_size, mOp);

            params.local_input_buffer_ptr = input.data_ptr();
            params.elts_total = size;
            if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
            {
                finalOutput = torch::empty_like(input);
                int fusion_ptr_idx = 0;
                params.local_output_buffer_ptr = finalOutput.mutable_data_ptr();
                params.fusion_params.bias_buffer = mBias ? reduce_fusion_inputs[fusion_ptr_idx++].data_ptr() : nullptr;
                params.fusion_params.residual_buffer = reduce_fusion_inputs[fusion_ptr_idx++].data_ptr();
                params.fusion_params.weight_buffer
                    = mAffine ? reduce_fusion_inputs[fusion_ptr_idx++].data_ptr() : nullptr;
                params.fusion_params.hidden_size = hidden_size;
                params.fusion_params.eps = mEps;
                params.fusion_params.intermediate_buffer = output.mutable_data_ptr();
                for (int i = 0; i < tpSize; ++i)
                {
                    params.fusion_params.lamport_peer_comm_buffer_ptrs[i]
                        = reinterpret_cast<void**>(workspace_ptr)[tpSize * 4 + i];
                    params.fusion_params.lamport_peer_comm_buffer_ptrs[i + tensorrt_llm::kernels::MAX_RANKS_PER_NODE]
                        = reinterpret_cast<void**>(workspace_ptr)[tpSize * 5 + i];
                    params.fusion_params
                        .lamport_peer_comm_buffer_ptrs[i + tensorrt_llm::kernels::MAX_RANKS_PER_NODE * 2]
                        = reinterpret_cast<void**>(workspace_ptr)[tpSize * 6 + i];
                }
            }
            else
            {
                params.local_output_buffer_ptr = output.mutable_data_ptr();
            }
            tensorrt_llm::kernels::customAllReduce(params, mType, runtimeStrategy, mConfig, mOp, stream);
        }

        if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
        {
            return std::make_tuple(finalOutput, output);
        }
        else
        {
            return std::make_tuple(output, output);
        }
    }

    int initialize() noexcept
    {
        TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        mNcclComm = getComm(mGroup);
        if (mStrategy != AllReduceStrategyType::NCCL)
        {
            initGroupTopology();
        }

        TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        return 0;
    }

private:
    void initGroupTopology() noexcept
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

    void setGroupTopology() noexcept
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

    AllReduceStrategyType selectImplementation(size_t messageSize, int worldSize, nvinfer1::DataType type) noexcept
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

        auto const maxWorkspaceSize = tensorrt_llm::utils::customAllReduceUtils::getMaxRequiredWorkspaceSize(worldSize);

        AllReduceStrategyType strat = AllReduceStrategyType::NCCL;
        auto const messageSizeBytes = messageSize * tensorrt_llm::common::getDTypeSize(type);

        if (messageSizeBytes <= maxWorkspaceSize)
        {
            // In some instances, the two-shot strategy has exhibited significant performance issues.
            // As a temporary measure, we have disabled the two-shot strategy.
            // TODO: remove this WAR after https://nvbugspro.nvidia.com/bug/4718747 is fixed.
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
                    strat = AllReduceStrategyType::NCCL;
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
                    strat = AllReduceStrategyType::NCCL;
                }
            }

            if (!tensorrt_llm::kernels::configurationSupported(strat, messageSize, worldSize, type))
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

private:
    std::set<int> mGroup;
    bool mIsNVLINKSupported;
    bool mIsP2PSupported;
    nvinfer1::DataType mType;
    AllReduceStrategyType mStrategy;
    AllReduceStrategyConfig mConfig;
    AllReduceFusionOp mOp;
    float mEps;
    std::shared_ptr<ncclComm_t> mNcclComm;
    bool mAffine;
    bool mBias;
    bool mScale;
};

} // namespace

#endif // ENABLE_MULTI_DEVICE

std::tuple<torch::Tensor, torch::Tensor> allreduce(torch::Tensor input, torch::optional<torch::Tensor> workspace,
    torch::TensorList reduce_fusion_inputs, torch::List<int64_t> group_, int64_t const strategy_, int64_t const config_,
    int64_t const fusion_op_, double const eps_, bool const affine_, bool const bias_, bool const scale_)
{
#if ENABLE_MULTI_DEVICE
    auto const dtype = tensorrt_llm::runtime::TorchUtils::dataType(input.scalar_type());
    auto const strategy = static_cast<AllReduceStrategyType>(int8_t(strategy_));
    auto const config = static_cast<AllReduceStrategyConfig>(int8_t(config_));
    auto const fusion_op = static_cast<AllReduceFusionOp>(int8_t(fusion_op_));
    float const eps = eps_;
    std::set<int> group;
    for (int64_t rank : group_)
    {
        group.insert(static_cast<int>(rank));
    }
    AllreduceOp op(group, dtype, strategy, config, fusion_op, eps, affine_, bias_, scale_);
    op.initialize();
    auto output = op.run(input, workspace, reduce_fusion_inputs);
    return output;
#else
    return std::make_tuple(input, input);
#endif // ENABLE_MULTI_DEVICE
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "allreduce(Tensor input, Tensor? workspace, Tensor[] reduce_fusion_inputs, int[] group, int "
        "strategy, int config, int op, float eps, bool affine, bool bias, bool scale) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("allreduce", &torch_ext::allreduce);
}
