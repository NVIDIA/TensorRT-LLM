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

#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/customAllReduceUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/mcastDevMemUtils.h"
#include "tensorrt_llm/common/ncclUtils.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.h"
#include "tensorrt_llm/kernels/communicationKernels/customLowPrecisionAllReduceKernels.h"
#include "tensorrt_llm/kernels/communicationKernels/mnnvlAllreduceKernels.h"
#include "tensorrt_llm/kernels/communicationKernels/moeAllReduceFusionKernels.h"
#include "tensorrt_llm/kernels/customAllReduceKernels.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/kernels/userbuffers/ub_interface.h"
#include "tensorrt_llm/runtime/mcastDeviceMemory.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/pgUtils.h"
#include "tensorrt_llm/thop/fp4Quantize.h"
#include "tensorrt_llm/thop/fp8Op.h"
#include "tensorrt_llm/thop/thUtils.h"
#include "tensorrt_llm/thop/userbuffersTensor.h"

#if ENABLE_MULTI_DEVICE
#include <ATen/cuda/EmptyTensor.h>
#include <c10/util/irange.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#endif // ENABLE_MULTI_DEVICE
#include <nvml.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_set>

// using namespace nvinfer1;
using tensorrt_llm::kernels::AllReduceFusionOp;
using tensorrt_llm::kernels::AllReduceStrategyType;
using tensorrt_llm::mpi::MpiTag;
using tensorrt_llm::pg_utils::get_world_pg;
using tensorrt_llm::pg_utils::get_local_pg;
using tensorrt_llm::pg_utils::PgHelper;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

#if ENABLE_MULTI_DEVICE

namespace
{

template <class... Ts>
struct overloaded : Ts...
{
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

class NvmlManager
{
public:
    NvmlManager()
    {
        NVML_CHECK_THROW(nvmlInit());
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

std::set<int> getLocalGroupTorch(std::set<int> const& group)
{
    auto const worldPg = get_world_pg();
    auto const myRank = worldPg->getRank();
    auto const localPg = get_local_pg();
    auto const myLocalRank = localPg->getRank();
    auto const localSize = static_cast<uint32_t>(localPg->getSize());

    PgHelper pgh_local{localPg};
    PgHelper pgh_world{worldPg}; // for p2p

    std::vector<int32_t> ranks(localSize, -1);
    std::vector<int32_t> localRanks(localSize, -1);

    if (group.size() >= localSize)
    {
        PGCHECK_THROW(pgh_local.allgather(&myRank, ref(ranks), {}));
        PGCHECK_THROW(pgh_local.allgather(&myLocalRank, ref(localRanks), {}));
    }
    else
    {
        int tag = static_cast<int>(MpiTag::kDefault);

        if (myRank == *group.begin())
        {
            // Leader: gather from peers (world ranks), then broadcast full localSize arrays.
            size_t cnt = 0;
            ranks[cnt++] = myRank;
            int tmp;
            for (auto it = std::next(group.begin()); it != group.end(); ++it)
            {
                PGCHECK_THROW(pgh_world.recv(&tmp, *it, tag));
                ranks[cnt++] = tmp;
            }
            for (auto it = std::next(group.begin()); it != group.end(); ++it)
            {
                PGCHECK_THROW(pgh_world.send(ref(ranks), *it, tag));
            }

            cnt = 0;
            localRanks[cnt++] = myLocalRank;
            for (auto it = std::next(group.begin()); it != group.end(); ++it)
            {
                PGCHECK_THROW(pgh_world.recv(&tmp, *it, tag));
                localRanks[cnt++] = tmp;
            }
            for (auto it = std::next(group.begin()); it != group.end(); ++it)
            {
                PGCHECK_THROW(pgh_world.send(ref(localRanks), *it, tag));
            }
        }
        else
        {
            int leader = *group.begin();

            PGCHECK_THROW(pgh_world.send(&myRank, leader, tag));
            PGCHECK_THROW(pgh_world.recv(ref(ranks), leader, tag));

            PGCHECK_THROW(pgh_world.send(&myLocalRank, leader, tag));
            PGCHECK_THROW(pgh_world.recv(ref(localRanks), leader, tag));
        }
    }

    std::set<int> localGroup;
    for (size_t i = 0; i < ranks.size(); ++i)
    {
        int world_r = ranks[i];
        if (group.find(world_r) != group.end())
            localGroup.insert(localRanks[i]);
    }
    return localGroup;
}

class AllreduceOp
{
public:
    AllreduceOp(
        std::set<int> group, nvinfer1::DataType type, AllReduceStrategyType strategy, AllReduceFusionOp op, float eps)
        : mGroup(std::move(group))
        , mIsNVLINKSupported(false)
        , mIsP2PSupported(false)
        , mIsMNNVLSupported(false)
        , mType(type)
        , mStrategy(strategy)
        , mOp(op)
        , mEps(eps)
    {
    }

    AllreduceOp(std::set<int> group, c10::intrusive_ptr<c10d::ProcessGroup> const& process_group_,
        nvinfer1::DataType type, AllReduceStrategyType strategy, AllReduceFusionOp op, float eps)
        : mGroup(std::move(group))
        , mIsNVLINKSupported(false)
        , mIsP2PSupported(false)
        , mIsMNNVLSupported(false)
        , mType(type)
        , mStrategy(strategy)
        , mOp(op)
        , mEps(eps)
        , mNcclComm(process_group_)
    {
    }

    ~AllreduceOp() = default;

    int getRank() const
    {
        return std::visit(
            overloaded{[&](std::shared_ptr<ncclComm_t> const&) { return COMM_SESSION.getRank(); },
                [&](c10::intrusive_ptr<c10d::ProcessGroup> const& torchPg) { return get_world_pg()->getRank(); }},
            mNcclComm);
    }

    std::vector<torch::Tensor> run(torch::Tensor const& input, torch::optional<torch::Tensor> const& residual,
        torch::optional<torch::Tensor> const& norm_weight, torch::optional<torch::Tensor> const& scale,
        torch::optional<torch::Tensor> const& bias, bool trigger_completion_at_end,
        torch::optional<torch::Tensor> workspace)
    {
        size_t size = input.numel();
        size_t seq_len = input.size(0);
        size_t hidden_size = input.size(-1);
        size_t bytes_per_element = input.element_size();
        TLLM_LOG_DEBUG("All reduce message size is %zu", size * bytes_per_element);

        AllReduceStrategyType runtime_strategy = selectImplementation(seq_len, hidden_size);

        // Log runtime strategy
        auto const rank = getRank();
        TLLM_LOG_DEBUG(
            "AllReduceOp runtime strategy for rank %d: " + tensorrt_llm::kernels::toString(runtime_strategy), rank);
        // Dispatch to different allreduce implementations
        switch (runtime_strategy)
        {
        case AllReduceStrategyType::UB: return runUBAllReduce(input, residual, norm_weight, scale, bias);
        case AllReduceStrategyType::NCCL: return runNCCLAllReduce(input, residual, norm_weight, scale, bias);
        case AllReduceStrategyType::NCCL_SYMMETRIC:
            return runNCCLAllReduceSymmetric(input, residual, norm_weight, scale, bias);
        case AllReduceStrategyType::MIN_LATENCY:
        case AllReduceStrategyType::ONESHOT:
        case AllReduceStrategyType::TWOSHOT:
            return runFusionAllReduce(
                input, residual, norm_weight, scale, bias, trigger_completion_at_end, workspace, runtime_strategy);
        case AllReduceStrategyType::LOWPRECISION:
            return runLowPrecisionAllReduce(input, residual, norm_weight, scale, bias);
        default: TORCH_CHECK(false, "Invalid runtime strategy"); return {};
        }
    }

    int initialize()
    {
        TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, getRank());
        if (mNcclComm.index() == 0)
        {
            mNcclComm = getComm(mGroup);
        }
        if (mStrategy != AllReduceStrategyType::NCCL && mStrategy != AllReduceStrategyType::UB)
        {

            initGroupTopology();
        }

        TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, getRank());
        return 0;
    }

private:
    std::vector<torch::Tensor> runUBAllReduce(torch::Tensor const& input,
        torch::optional<torch::Tensor> const& residual, torch::optional<torch::Tensor> const& norm_weight,
        torch::optional<torch::Tensor> const& scale, torch::optional<torch::Tensor> const& bias)
    {
        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
        int size = input.numel();
        int hidden_size = input.size(-1);

        torch::Tensor residual_out = torch::empty_like(input);

        TLLM_CHECK(mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM || mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_FP8
            || mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4);
        TLLM_CHECK_WITH_INFO(tensorrt_llm::runtime::ub::ub_is_initialized(), "UserBuffer has not been initialized!");
        auto& ub_manager = tensorrt_llm::runtime::ub::UserBuffersManager::get_instance();
        auto ub_buffer0 = ub_manager.search_buffer(input.data_ptr());
        TLLM_CHECK(!ub_buffer0.invalid());

        auto ub_comm = ub_manager.comm();
        int m = size / hidden_size;

        if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
        {
            TORCH_CHECK(norm_weight, "norm_weight is required for residual rms norm allreduce");
            TORCH_CHECK(!bias, "bias is not supported for residual rms norm allreduce");
            TORCH_CHECK(mType == nvinfer1::DataType::kHALF || mType == nvinfer1::DataType::kBF16);
            auto [norm_out, ub_buffer1] = torch_ext::create_userbuffers_tensor(input.sizes(), input.scalar_type());
            tensorrt_llm::kernels::ub::allreduce2_userbuff_rmsnorm_launcher(ub_buffer0.handle, 0, ub_buffer1.handle, 0,
                size, hidden_size, nullptr, norm_weight.value().data_ptr(), mEps, residual.value().data_ptr(),
                residual_out.data_ptr(), mType, ub_comm, stream);

            return {norm_out, residual_out};
        }
        else if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_FP8)
        {
            TORCH_CHECK(scale, "scale is required for FP8 allreduce");
            TORCH_CHECK(norm_weight, "norm_weight is required for FP8 allreduce");
            TORCH_CHECK(!bias, "bias is not supported for FP8 allreduce");
            auto [norm_out, ub_buffer1] = torch_ext::create_userbuffers_tensor(input.sizes(), torch::kFloat8_e4m3fn);
            tensorrt_llm::kernels::ub::allreduce2_userbuff_inplace_rmsnorm_quant_launcher(ub_buffer0.handle, 0,
                ub_buffer1.handle, 0, size, hidden_size, nullptr, norm_weight.value().data_ptr(), mEps,
                static_cast<float*>(scale.value().data_ptr()), residual.value().data_ptr(), residual_out.data_ptr(),
                mType, ub_comm, stream);

            return {norm_out, residual_out};
        }
        else if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4)
        {
            TORCH_CHECK(scale, "scale is required for FP4 allreduce");
            TORCH_CHECK(norm_weight, "norm_weight is required for FP4 allreduce");
            TORCH_CHECK(!bias, "bias is not supported for FP4 allreduce");

            int const sfVecSize = 16;
            int scale_size
                = tensorrt_llm::common::roundUp(m, 128) * tensorrt_llm::common::roundUp(hidden_size / sfVecSize, 4);

            TORCH_CHECK(hidden_size % sfVecSize == 0, "hidden_size must be divisible by 16 for FP4 allreduce");

            auto output_shape = input.sizes().vec();
            output_shape.back() /= 2;
            auto output_strides = input.strides().vec();
            for (size_t i = 0; i < output_shape.size() - 1; i++)
            {
                output_strides[i] /= 2;
            }

            auto [quant_out, ub_buffer1] = torch_ext::create_userbuffers_tensor(output_shape, torch::kByte);
            auto [scale_out, ub_buffer2] = torch_ext::create_userbuffers_tensor({scale_size}, torch::kByte);

            tensorrt_llm::kernels::ub::allreduce2_userbuff_inplace_rmsnorm_quant_fp4_launcher(ub_buffer0.handle, 0,
                ub_buffer1.handle, 0, ub_buffer2.handle, 0, size, hidden_size, nullptr, norm_weight.value().data_ptr(),
                mEps, static_cast<float*>(scale.value().data_ptr()), residual.value().data_ptr(),
                residual_out.data_ptr(), mType, ub_comm, stream);

            return {quant_out, scale_out, residual_out};
        }
        TORCH_CHECK(
            false, "UBAllreduce does not support the fusion operation: " + tensorrt_llm::kernels::toString(mOp));
        return {};
    }

    std::vector<torch::Tensor> runNCCLAllReduce(torch::Tensor const& input,
        torch::optional<torch::Tensor> const& residual, torch::optional<torch::Tensor> const& norm_weight,
        torch::optional<torch::Tensor> const& scale, torch::optional<torch::Tensor> const& bias)
    {
        torch::Tensor reduce_output;

        std::visit(overloaded{[&](std::shared_ptr<ncclComm_t>& rawComm)
                       {
                           auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
                           int size = input.numel();
                           reduce_output = torch::empty_like(input);
                           NCCLCHECK_THROW(ncclAllReduce(input.data_ptr(), reduce_output.mutable_data_ptr(), size,
                               (*getDtypeMap())[mType], ncclSum, *rawComm, stream));
                       },
                       [&](c10::intrusive_ptr<c10d::ProcessGroup>& torchPg)
                       {
                           reduce_output = input.clone();
                           // TLLM_LOG_INFO("AllReduce Rank: %d, tensor numel: %d", torchPg->getRank(),
                           // reduce_output.numel());
                           std::vector tensors{reduce_output};
                           PGCHECK_THROW(torchPg->allreduce(tensors, {c10d::ReduceOp::SUM}));
                       }},
            mNcclComm);

        if (mOp == AllReduceFusionOp::NONE)
        {
            return {reduce_output};
        }

        // Treat any other patterns as fallback cases.
        return fallbackRunSubsequentOps(input, residual, norm_weight, scale, bias, reduce_output);
    }

    std::vector<torch::Tensor> runNCCLAllReduceSymmetric(torch::Tensor const& input,
        torch::optional<torch::Tensor> const& residual, torch::optional<torch::Tensor> const& norm_weight,
        torch::optional<torch::Tensor> const& scale, torch::optional<torch::Tensor> const& bias)
    {
        // Handle ProcessGroup path first - cannot extract NCCL comm for window registration
        // Use ProcessGroup's allreduce directly and return early
        if (mNcclComm.index() == 1)
        {
            auto torchPg = std::get<1>(mNcclComm);

            torch::Tensor reduceOutput = input.clone();
            std::vector tensors{reduceOutput};
            PGCHECK_THROW(torchPg->allreduce(tensors, {c10d::ReduceOp::SUM}));

            if (mOp == AllReduceFusionOp::NONE)
            {
                return {reduceOutput};
            }

            // Treat any other patterns as fallback cases.
            return fallbackRunSubsequentOps(input, residual, norm_weight, scale, bias, reduceOutput);
        }

        // From here on, we have a raw NCCL comm - can proceed with window registration
        auto rawComm = std::get<0>(mNcclComm);
        ncclComm_t comm = *rawComm;
        TLLM_CHECK_WITH_INFO(comm != nullptr, "NCCL communicator is null");
        TLLM_LOG_DEBUG("[runNCCLAllReduceSymmetric] Using raw NCCL comm path (not ProcessGroup)");

        using tensorrt_llm::common::nccl_util::NCCLWindowAllocator;
        using tensorrt_llm::common::nccl_util::createNCCLWindowTensor;

        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
        int size = input.numel();
        size_t bufferSizeBytes = size * input.element_size();

        // Using unregistered input buffers with NCCL symmetric, requires a memcpy
        // This is an overhead introduced with using NCCL_SYMMTRIC over NCCL.
        // Both the memcpy and the perf benefit from using NCCL_SYMMETRIC scale linear with the message size.
        // But a local memcpy is cheaper than the remote operations, so with larger message sizes the benefit is
        // stronger. Additionally, the perf benefit scales with the number of ranks, since multimem enables O(const.)
        // versus O(N) complexity. Hence we model this cutoff with a linear model. The numbers below were obtained on
        // GB200, scanning different message sizes and ranks. You can determine the regression onset for each number of
        // ranks to a single message size. And the following formula was obtained by fitting a linear model to the
        // regression onset. It is possible to override this empirical heuristic with the TLLM_NCCL_MIN_REGISTRATION
        // environment variable.
        double const a = -4986.43478503;
        double const b = 156716.52177552;
        int nRanks;
        NCCLCHECK_THROW(ncclCommCount(comm, &nRanks));
        size_t minRegistrationThreshold = static_cast<size_t>(std::max(0.0, a * nRanks + b)) * input.element_size();
        // Disable window registration if neither NVLink nor MNNVL is supported
        // TODO replace in NCCL 2.29 with comm query
        if (!mIsNVLINKSupported && !mIsMNNVLSupported)
        {
            minRegistrationThreshold = std::numeric_limits<size_t>::max();
        }
        char const* envThreshold = std::getenv("TLLM_NCCL_MIN_REGISTRATION");
        if (envThreshold != nullptr)
        {
            minRegistrationThreshold = static_cast<size_t>(std::atoi(envThreshold)) * input.element_size();
        }

        auto& allocator = NCCLWindowAllocator::getInstance();

        // Search for existing buffer
        auto windowBuffer0 = allocator.searchBuffer(comm, input.data_ptr());

        torch::Tensor inputTensor = input;
        void* inputPtr = input.data_ptr();

        // If buffer is not registered, decide whether to register based on size
        if (!windowBuffer0.isValid())
        {
            if (bufferSizeBytes < minRegistrationThreshold)
            {
                // Small buffer: use input directly without window registration
                TLLM_LOG_DEBUG(
                    "[runNCCLAllReduceSymmetric] Buffer size %zu bytes < threshold %zu bytes, "
                    "skipping window registration",
                    bufferSizeBytes, minRegistrationThreshold);
                // inputTensor and inputPtr remain pointing to original input
            }
            else
            {
                // Large buffer: create window buffer and copy input (can swap inputTensor reference)
                auto [symmetricInput, symmetricBuffer0]
                    = createNCCLWindowTensor(comm, input.sizes(), input.scalar_type());
                if (!symmetricBuffer0.isValid())
                {
                    TLLM_LOG_DEBUG(
                        "[runNCCLAllReduceSymmetric] No valid symmetric buffer available; "
                        "falling back to non-symmetric ncclAllReduce (input buffer)");
                    // inputTensor and inputPtr remain pointing to original input
                }
                else
                {
                    TLLM_CUDA_CHECK(cudaMemcpyAsync(
                        symmetricBuffer0.ptr, input.data_ptr(), bufferSizeBytes, cudaMemcpyDeviceToDevice, stream));

                    windowBuffer0 = symmetricBuffer0;
                    inputTensor = symmetricInput; // Swap to window-backed tensor
                    inputPtr = windowBuffer0.ptr;
                }
            }
        }
        else
        {
            // Buffer already registered - use it directly
            inputPtr = windowBuffer0.ptr;
        }

        // Use window-backed output buffer
        auto [normOut, windowBuffer1] = createNCCLWindowTensor(comm, input.sizes(), input.scalar_type());
        torch::Tensor outputTensor = windowBuffer1.isValid() ? normOut : torch::empty_like(inputTensor);
        void* outputPtr = windowBuffer1.isValid() ? windowBuffer1.ptr : outputTensor.data_ptr();
        if (!windowBuffer1.isValid())
        {
            TLLM_LOG_DEBUG(
                "[runNCCLAllReduceSymmetric] No valid symmetric buffer available; "
                "using plain CUDA tensor for output");
        }

        // Perform allreduce
        NCCLCHECK_THROW(ncclAllReduce(inputPtr, outputPtr, size, (*getDtypeMap())[mType], ncclSum, comm, stream));

        if (mOp == AllReduceFusionOp::NONE)
        {
            return {outputTensor};
        }

        // Treat any other patterns as fallback cases.
        return fallbackRunSubsequentOps(input, residual, norm_weight, scale, bias, outputTensor);
    }

    std::vector<torch::Tensor> runLowPrecisionAllReduce(torch::Tensor const& input,
        torch::optional<torch::Tensor> const& residual, torch::optional<torch::Tensor> const& norm_weight,
        torch::optional<torch::Tensor> const& scale, torch::optional<torch::Tensor> const& bias)
    {
#ifdef ENABLE_FP8
        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
        int size = input.numel();
        int hidden_size = input.size(-1);

        auto const tp_size = mGroup.size();
        auto const cur_rank = getRank();
        int tp_rank = 0;

        for (auto const& currentRank : mGroup)
        {
            if (cur_rank == currentRank)
                break;
            ++tp_rank;
        }

        int bytes_per_element = input.element_size();

        int token_num = size / hidden_size;

        auto parts = tensorrt_llm::kernels::splitNumber(size);

        torch::Tensor reduce_output = torch::empty_like(input);

        size_t global_offset = 0;
        for (size_t i = 0; i < parts.size(); ++i)
        {
            size_t tmp_size = parts[i];
            tensorrt_llm::kernels::LowPrecisionAllReduceParams tmp_param;
            if (tp_size <= 4)
            {
                tmp_param = tensorrt_llm::kernels::LowPrecisionAllReduceParams::deserialize(
                    tp_size, tp_rank, mType, token_num, hidden_size);
            }
            else
            {
                tmp_param = tensorrt_llm::kernels::LowPrecisionAllReduceParams::deserialize_hier(
                    tp_size, tp_rank, mType, token_num, hidden_size);
            }

            tmp_param.local_input_buffer_ptr = reinterpret_cast<void const*>(
                reinterpret_cast<char const*>(input.data_ptr()) + global_offset * bytes_per_element);
            tmp_param.local_output_buffer_ptr = reinterpret_cast<void*>(
                reinterpret_cast<char*>(reduce_output.mutable_data_ptr()) + global_offset * bytes_per_element);
            tmp_param.elts_total = tmp_size;
            tensorrt_llm::kernels::customLowPrecisionAllReduce(tmp_param, mType, stream);

            global_offset += tmp_size;
        }

        if (mOp == AllReduceFusionOp::NONE)
        {
            return {reduce_output};
        }

        // Treat any other patterns as fallback cases.
        return fallbackRunSubsequentOps(input, residual, norm_weight, scale, bias, reduce_output);

#else
        C10_THROW_ERROR(NotImplementedError, "Can't use LOWPRECISION without compile with ENABLE FP8.");
#endif
    }

    std::vector<torch::Tensor> runFusionAllReduce(torch::Tensor const& input,
        torch::optional<torch::Tensor> const& residual, torch::optional<torch::Tensor> const& norm_weight,
        torch::optional<torch::Tensor> const& scale, torch::optional<torch::Tensor> const& bias,
        bool trigger_completion_at_end, torch::optional<torch::Tensor> workspace, AllReduceStrategyType strategy)
    {
        // Should handle only Lamport implementation
        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
        int size = input.numel();
        int hidden_size = input.size(-1);
        int seq_len = input.size(0);

        auto const tp_size = mGroup.size();
        auto const cur_rank = getRank();
        int tp_rank = 0;

        for (auto const& currentRank : mGroup)
        {
            if (cur_rank == currentRank)
                break;
            ++tp_rank;
        }

        // Use cleaner output assigning
        torch::Tensor reduce_out;
        torch::Tensor residual_out;
        torch::Tensor norm_out;
        torch::Tensor quant_out;
        torch::Tensor scale_out;

        tensorrt_llm::kernels::ar_fusion::AllReduceFusionParams allreduce_fusion_params;

        allreduce_fusion_params.residual_in = nullptr;
        allreduce_fusion_params.rms_gamma = nullptr;

        allreduce_fusion_params.allreduce_out = nullptr;
        allreduce_fusion_params.quant_out = nullptr;
        allreduce_fusion_params.scale_out = nullptr;
        allreduce_fusion_params.residual_out = nullptr;
        allreduce_fusion_params.norm_out = nullptr;
        allreduce_fusion_params.trigger_completion_at_end = trigger_completion_at_end;

        // Determine if using oneshot or twoshot allreduce kernel in case using MIN_LATENCY strategy.
        if (strategy == AllReduceStrategyType::MIN_LATENCY)
        {
            allreduce_fusion_params.use_oneshot = seq_len <= tensorrt_llm::kernels::ar_fusion::kOneShotMaxToken
                || hidden_size < static_cast<int64_t>(tp_size);
        }
        else
        {
            allreduce_fusion_params.use_oneshot = strategy == AllReduceStrategyType::ONESHOT;
        }

        // Check for some kernel constraints if using TWOSHOT kernel
        if (!allreduce_fusion_params.use_oneshot)
        {
            TORCH_CHECK(input.size(0) >= static_cast<int64_t>(tp_size),
                "Sequence length must be greater than or equal to TP size");
        }

        // Handle no fusion allreduce here
        if (mOp == AllReduceFusionOp::NONE)
        {
            reduce_out = torch::empty_like(input);
            allreduce_fusion_params.allreduce_out = reduce_out.mutable_data_ptr();
            allreduce_fusion_params.pattern = tensorrt_llm::kernels::ar_fusion::AllReduceFusionPattern::kAllReduce;
        }
        // Handle allreduce fusion here
        // Prepare required output tensors for each fusion pattern
        else if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
        {
            norm_out = torch::empty_like(input);
            residual_out = torch::empty_like(residual.value());

            allreduce_fusion_params.norm_out = norm_out.mutable_data_ptr();
            allreduce_fusion_params.residual_out = residual_out.mutable_data_ptr();
            allreduce_fusion_params.pattern
                = tensorrt_llm::kernels::ar_fusion::AllReduceFusionPattern::kARResidualRMSNorm;
        }
        else if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_FP8
            || mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_FP8)
        {
            quant_out = at::detail::empty_cuda(input.sizes(), torch::kFloat8_e4m3fn, input.device(), std::nullopt);
            residual_out = torch::empty_like(residual.value());

            allreduce_fusion_params.quant_out = quant_out.mutable_data_ptr();
            allreduce_fusion_params.residual_out = residual_out.mutable_data_ptr();
            allreduce_fusion_params.pattern
                = tensorrt_llm::kernels::ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP8Quant;

            // norm out is required
            if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_FP8)
            {
                norm_out = torch::empty_like(input);
                allreduce_fusion_params.norm_out = norm_out.mutable_data_ptr();
                allreduce_fusion_params.pattern
                    = tensorrt_llm::kernels::ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant;
            }
        }
        else if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4
            || mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4)
        {
            // TODO: Better check for each pattern
            int64_t sf_vec_size = 16;
            int64_t m = 1;
            auto const& input_shape = input.sizes();
            auto const& r = input_shape.size();
            TORCH_CHECK(r >= 2, "Input should be >=2D tensor.");
            for (size_t i = 0; i < r - 1; i++)
            {
                m *= input_shape[i];
            }
            auto const k = input_shape[r - 1];
            TORCH_CHECK(k % sf_vec_size == 0, "Input should be divisible by sfVecSize.");
            std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
            output_shape[r - 1] = k / 2;

            quant_out = at::detail::empty_cuda(output_shape, FLOAT4_E2M1X2, input.device(), std::nullopt);
            scale_out = at::detail::empty_cuda({tensorrt_llm::computeSwizzledLayoutSFSize(m, k / sf_vec_size)},
                SF_DTYPE, input.device(), std::nullopt);
            residual_out = torch::empty_like(residual.value());

            allreduce_fusion_params.quant_out = quant_out.mutable_data_ptr();
            allreduce_fusion_params.scale_out = scale_out.mutable_data_ptr();
            allreduce_fusion_params.residual_out = residual_out.mutable_data_ptr();
            allreduce_fusion_params.pattern
                = tensorrt_llm::kernels::ar_fusion::AllReduceFusionPattern::kARResidualRMSNormFP4Quant;

            // norm out is required
            if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4)
            {
                norm_out = torch::empty_like(input);
                allreduce_fusion_params.norm_out = norm_out.mutable_data_ptr();
                allreduce_fusion_params.pattern
                    = tensorrt_llm::kernels::ar_fusion::AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant;
            }
        }
        else
        {
            TORCH_CHECK(false, "Unsupported fusion operation: " + tensorrt_llm::kernels::toString(mOp));
            return {};
        }

        allreduce_fusion_params.nranks = tp_size;
        allreduce_fusion_params.rank = tp_rank;
        allreduce_fusion_params.dtype = mType;
        allreduce_fusion_params.size = size;
        allreduce_fusion_params.hidden_dim = hidden_size;
        allreduce_fusion_params.workspace = reinterpret_cast<void**>(workspace.value().mutable_data_ptr());
        allreduce_fusion_params.allreduce_in = input.data_ptr();

        if (mOp != AllReduceFusionOp::NONE)
        {
            allreduce_fusion_params.residual_in = residual.value().data_ptr();
            allreduce_fusion_params.rms_gamma = norm_weight.value().data_ptr();
            allreduce_fusion_params.rms_eps = mEps;
        }

        allreduce_fusion_params.stream = stream;

        bool const is_scale_factor_required = mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_FP8
            || mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_FP8
            || mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4
            || mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4;

        allreduce_fusion_params.scale_factor
            = is_scale_factor_required ? static_cast<float*>(scale.value().data_ptr()) : nullptr;

        tensorrt_llm::kernels::ar_fusion::allreduce_fusion_op(allreduce_fusion_params);

        // Pack output tensors
        switch (mOp)
        {
        case AllReduceFusionOp::NONE: return {reduce_out};
        case AllReduceFusionOp::RESIDUAL_RMS_NORM: return {norm_out, residual_out};
        case AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_FP8: return {quant_out, residual_out};
        case AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_FP8: return {norm_out, quant_out, residual_out};
        case AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4: return {quant_out, scale_out, residual_out};
        case AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4:
            return {norm_out, quant_out, scale_out, residual_out};
        default: TORCH_CHECK(false, "Unsupported fusion operation: " + tensorrt_llm::kernels::toString(mOp));
        }
        return {};
    }

    std::vector<torch::Tensor> fallbackRunSubsequentOps(torch::Tensor const& input,
        torch::optional<torch::Tensor> const& residual, torch::optional<torch::Tensor> const& norm_weight,
        torch::optional<torch::Tensor> const& scale, torch::optional<torch::Tensor> const& bias,
        torch::Tensor& reduce_output)
    {
        // If we reach here, it means the extra fallback operations are required.
        // All patterns are broken into ALlReduce + residual_rms_norm + following operations (quantization, etc.)
        auto const size = input.numel();
        auto const hidden_size = input.size(-1);
        auto const stream = at::cuda::getCurrentCUDAStream(input.get_device());

        torch::Tensor norm_out = torch::empty_like(input);

        tensorrt_llm::kernels::AllReduceParams params;
        params.fusion_params.bias_buffer = bias ? bias.value().data_ptr() : nullptr;
        params.fusion_params.residual_buffer = residual ? residual.value().data_ptr() : nullptr;
        params.fusion_params.weight_buffer = norm_weight ? norm_weight.value().data_ptr() : nullptr;
        params.local_output_buffer_ptr = norm_out.mutable_data_ptr();
        params.elts_total = size;

        params.fusion_params.hidden_size = hidden_size;
        params.fusion_params.eps = mEps;
        params.fusion_params.intermediate_buffer = reduce_output.mutable_data_ptr();
        tensorrt_llm::kernels::residualRmsNorm(params, mType, stream, AllReduceFusionOp::RESIDUAL_RMS_NORM);

        // If no quantization is needed, return the norm and residual outputs.
        if (mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM)
        {
            return {norm_out, reduce_output};
        }

        int64_t const sf_vecsize = 16;
        bool const sf_use_ue8m0 = false;
        bool const is_sf_swizzled_layout = true;
        TORCH_CHECK(scale, "scale is required for quantization ops");

        // Attach the subsequent operations after the residual RMS norm all-reduce and return the final outputs.
        switch (mOp)
        {
        case AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_FP8:
        {
            auto [quant_out, scale_out] = torch_ext::symmetric_static_quantize_per_tensor(norm_out, scale.value());
            return {quant_out, reduce_output};
        }
        case AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4:
        {
            auto [quant_out, scale_out]
                = torch_ext::fp4_quantize(norm_out, scale.value(), sf_vecsize, sf_use_ue8m0, is_sf_swizzled_layout);
            return {quant_out, scale_out, reduce_output};
        }
        case AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_FP8:
        {
            auto [quant_out, scale_out] = torch_ext::symmetric_static_quantize_per_tensor(norm_out, scale.value());
            return {norm_out, quant_out, reduce_output};
        }
        case AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4:
        {
            auto [quant_out, scale_out]
                = torch_ext::fp4_quantize(norm_out, scale.value(), sf_vecsize, sf_use_ue8m0, is_sf_swizzled_layout);
            return {norm_out, quant_out, scale_out, reduce_output};
        }
        default: break;
        }

        TORCH_CHECK(false, "Unsupported fusion operation: " + tensorrt_llm::kernels::toString(mOp));
        return {};
    }

    void initGroupTopology()
    {
        static std::map<std::set<int>, std::tuple<bool, bool, bool>> cache;
        if (cache.find(mGroup) != cache.end())
        {
            auto [is_NVLINK_supported, is_P2P_supported, is_MNNVL_supported] = cache[mGroup];
            mIsNVLINKSupported = is_NVLINK_supported;
            mIsP2PSupported = is_P2P_supported;
            mIsMNNVLSupported = is_MNNVL_supported;
            return;
        }
        setGroupTopology();
        cache[mGroup] = {mIsNVLINKSupported, mIsP2PSupported, mIsMNNVLSupported};
    }

    bool checkMNNVLSupport(int device_id)
    {
#if ENABLE_MULTI_DEVICE
        // 1. Check CUDA driver version (needs >= 12.0.10)
        int cuda_driver_version = -1;
        TLLM_CUDA_CHECK(cudaDriverGetVersion(&cuda_driver_version));
        if (cuda_driver_version < 12010)
        {
            TLLM_LOG_DEBUG("MNNVL check: CUDA Driver version %d < 12010", cuda_driver_version);
            return false;
        }

        // 2. Check multicast support
        CUdevice cu_device;
        TLLM_CU_CHECK(cuDeviceGet(&cu_device, device_id));
        auto cuda_driver = tensorrt_llm::common::CUDADriverWrapper::getInstance();

        int multicast_supported = 0;
        TLLM_CU_CHECK(cuda_driver->cuDeviceGetAttribute(
            &multicast_supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, cu_device));
        if (!multicast_supported)
        {
            TLLM_LOG_DEBUG("MNNVL check: Device %d does not support multicast", device_id);
            return false;
        }

        // 3. Check fabric handle support
        int fabric_handle_supported = 0;
        TLLM_CU_CHECK(cuda_driver->cuDeviceGetAttribute(
            &fabric_handle_supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, cu_device));
        if (!fabric_handle_supported)
        {
            TLLM_LOG_DEBUG("MNNVL check: Device %d does not support fabric handles", device_id);
            return false;
        }

        // 4. Check NVML GPU Fabric Info
        nvmlDevice_t nvml_device;
        NVML_CHECK_THROW(nvmlDeviceGetHandleByIndex(device_id, &nvml_device));

        nvmlGpuFabricInfo_t fabric_info;
        NVML_CHECK_THROW(nvmlDeviceGetGpuFabricInfo(nvml_device, &fabric_info));

        // Check if fabric is fully initialized
        if (fabric_info.state != NVML_GPU_FABRIC_STATE_COMPLETED || fabric_info.status != NVML_SUCCESS)
        {
            TLLM_LOG_DEBUG(
                "MNNVL check: Fabric state not complete - state=%u status=%u", fabric_info.state, fabric_info.status);
            return false;
        }

        // 5. Check NVLink links are active (similar to Python support_nvlink(True))
        unsigned int active_links = 0;
        unsigned int available_links = 0;

        for (unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; link++)
        {
            unsigned int cap_p2p = 0;
            nvmlReturn_t cap_result
                = nvmlDeviceGetNvLinkCapability(nvml_device, link, NVML_NVLINK_CAP_P2P_SUPPORTED, &cap_p2p);
            if (cap_result == NVML_SUCCESS && cap_p2p)
            {
                available_links++;
                nvmlEnableState_t link_state;
                if (nvmlDeviceGetNvLinkState(nvml_device, link, &link_state) == NVML_SUCCESS
                    && link_state == NVML_FEATURE_ENABLED)
                {
                    active_links++;
                }
            }
        }

        bool all_links_up = (active_links == available_links && available_links > 0);
        if (!all_links_up)
        {
            TLLM_LOG_DEBUG(
                "MNNVL check: Not all NVLink links active - active=%u available=%u", active_links, available_links);
            return false;
        }

        TLLM_LOG_INFO("MNNVL check: Device %d supports MNNVL (fabric_clique=%u)", device_id, fabric_info.cliqueId);
        return true;
#else
        return false;
#endif
    }

    void setGroupTopology()
    {
        auto const rank = getRank();
        TLLM_LOG_INFO("Detecting local TP group for rank %d", rank);
        std::set<int> local_group = std::visit(
            overloaded{[&](std::shared_ptr<ncclComm_t>&) { return getLocalGroup(mGroup); },
                [&](c10::intrusive_ptr<c10d::ProcessGroup>& torchPg) { return getLocalGroupTorch(mGroup); }},
            mNcclComm);

        bool is_inter_node = (mGroup.size() != local_group.size());

        NvmlManager nvml_manager;
        mIsP2PSupported = true;
        mIsNVLINKSupported = true;
        mIsMNNVLSupported = false;

        // First, check NVLink within local group (intra-node)
        if (!local_group.empty())
        {
            for (int first_device_id : local_group)
            {
                for (int second_device_id : local_group)
                {
                    if (first_device_id >= second_device_id)
                    {
                        continue;
                    }

                    int can_access_peer = 0;
                    TLLM_CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, first_device_id, second_device_id));

                    if (!can_access_peer)
                    {
                        mIsP2PSupported = false;
                        mIsNVLINKSupported = false;
                        TLLM_LOG_INFO(
                            "P2P not supported between local devices %d and %d", first_device_id, second_device_id);
                        // Continue checking other pairs, but mark as not supported
                        continue;
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
                                if (nvmlDeviceGetNvLinkRemotePciInfo_v2(
                                        second_device, second_link, &second_remote_pci_info)
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

                    mIsNVLINKSupported &= is_NVLINK;
                }
            }
        }

        // For inter-node groups, check MNNVL support
        if (is_inter_node)
        {
            TLLM_LOG_INFO("Found inter-node TP group for rank %d, checking MNNVL support", rank);

            // Check MNNVL support on local device(s)
            bool local_mnnvl_supported = false;
            if (!local_group.empty())
            {
                // Check MNNVL on first device in local group (all devices on same node should have same MNNVL status)
                int check_device = *local_group.begin();
                local_mnnvl_supported = checkMNNVLSupport(check_device);
            }

            // Gather MNNVL status from all ranks in the group
            int local_mnnvl_status = local_mnnvl_supported ? 1 : 0;
            std::vector<int> all_mnnvl_status(mGroup.size());

            std::visit(overloaded{[&](std::shared_ptr<ncclComm_t>& comm_ptr)
                           {
                               // For NCCL comm, use MPI to gather status
                               // Use MPI allgather to collect MNNVL status
                               // Create a sub-communicator for the group
                               std::vector<int> group_ranks(mGroup.begin(), mGroup.end());
                               MPI_Group world_group, new_group;
                               MPI_Comm group_comm;
                               MPI_Comm_group(COMM_SESSION, &world_group);
                               MPI_Group_incl(world_group, group_ranks.size(), group_ranks.data(), &new_group);
                               MPI_Comm_create_group(COMM_SESSION, new_group, 0, &group_comm);

                               if (group_comm != MPI_COMM_NULL)
                               {
                                   MPI_Allgather(&local_mnnvl_status, 1, MPI_INT, all_mnnvl_status.data(), 1, MPI_INT,
                                       group_comm);
                                   MPI_Comm_free(&group_comm);
                               }
                               MPI_Group_free(&new_group);
                               MPI_Group_free(&world_group);
                           },
                           [&](c10::intrusive_ptr<c10d::ProcessGroup>& torchPg)
                           {
                               // For ProcessGroup, use allgather directly
                               // Note: This assumes the ProcessGroup is already set up for the correct group
                               std::vector<torch::Tensor> input_tensors
                                   = {torch::tensor({local_mnnvl_status}, torch::kInt32)};
                               std::vector<std::vector<torch::Tensor>> output_tensors(1);
                               output_tensors[0].resize(mGroup.size());
                               auto work = torchPg->allgather(output_tensors, input_tensors);
                               if (work)
                               {
                                   work->wait();
                                   for (size_t i = 0; i < mGroup.size(); ++i)
                                   {
                                       all_mnnvl_status[i] = output_tensors[0][i].item<int>();
                                   }
                               }
                           }},
                mNcclComm);

            // Check if all ranks support MNNVL
            bool all_ranks_support_mnnvl = true;
            for (int status : all_mnnvl_status)
            {
                if (status == 0)
                {
                    all_ranks_support_mnnvl = false;
                    break;
                }
            }

            // For inter-node: MNNVL support means all nodes have MNNVL
            // Also need local NVLink for optimal performance
            mIsMNNVLSupported = mIsNVLINKSupported && all_ranks_support_mnnvl;
            mIsP2PSupported = false; // P2P doesn't work across nodes

            TLLM_LOG_INFO("Inter-node topology: local_NVLink=%d, local_MNNVL=%d, all_ranks_MNNVL=%d, final_MNNVL=%d",
                mIsNVLINKSupported ? 1 : 0, local_mnnvl_status, all_ranks_support_mnnvl ? 1 : 0,
                mIsMNNVLSupported ? 1 : 0);
        }
        else
        {
            TLLM_LOG_INFO("TP group is intra-node for rank %d", rank);
        }
    }

    AllReduceStrategyType selectImplementation(size_t seq_len, size_t hidden_size)
    {
        if (mStrategy != AllReduceStrategyType::AUTO)
        {
            // For UB,NCCL,NCCL_SYMMETRIC, the correctness of the strategy dispatching is guaranteed by the user.
            if (mStrategy == AllReduceStrategyType::UB || mStrategy == AllReduceStrategyType::NCCL
                || mStrategy == AllReduceStrategyType::NCCL_SYMMETRIC)
            {
                return mStrategy;
            }
        }

        // For ONESHOT, TWOSHOT, LOWPRECISION, fallback is allowed.
        auto const message_size = seq_len * hidden_size;

        // Check if LOWPRECISION is supported.
        if (isUsingLowPrecision(hidden_size))
        {
            return AllReduceStrategyType::LOWPRECISION;
        }

        auto const message_size_bytes = message_size * tensorrt_llm::common::getDTypeSize(mType);
        auto const max_workspace_size
            = tensorrt_llm::utils::customAllReduceUtils::getMaxRequiredWorkspaceSize(mGroup.size());

        if (ifFallbackToNCCL(seq_len, message_size_bytes, max_workspace_size))
        {
            return AllReduceStrategyType::NCCL;
        }

        // This rule based heuristic only chooses between NCCL_SYMMETRIC and MIN_LATENCY strategies.
        // From this point, all fusion patterns are supported by all these strategies: NCCL_SYMMETRIC, ONESHOT, TWOSHOT
        // and MIN_LATENCY.
        if (mStrategy != AllReduceStrategyType::AUTO)
        {
            // Check TWOSHOT constraint: seq_len >= tp_size
            if (mStrategy == AllReduceStrategyType::TWOSHOT && seq_len < mGroup.size())
            {
                TLLM_LOG_WARNING("TWOSHOT strategy requires seq_len >= tp_size (%zu < %zu), falling back to ONESHOT",
                    seq_len, mGroup.size());
                return AllReduceStrategyType::ONESHOT;
            }
            return mStrategy;
        }
        else
        {
            return tensorrt_llm::utils::customAllReduceUtils::selectStrategyLookUpTable(
                seq_len, hidden_size, mOp, mGroup.size());
        }
    }

    bool ifFallbackToNCCL(size_t seq_len, size_t message_size_bytes, size_t max_workspace_size)
    {
        // If messageSize is greater than maxWorkspaceSize or topology is unsuitable, use NCCL fallback.
        // TODO: Use NCCL_SYMMETRIC once the memory allocation issue is resolved.
        if (message_size_bytes > max_workspace_size || !mIsP2PSupported || !mIsNVLINKSupported)
        {
            return true;
        }

        return false;
    }

    bool isUsingLowPrecision(size_t message_size) const noexcept
    {
        bool force_low_precision = mStrategy == AllReduceStrategyType::LOWPRECISION;

#ifdef ENABLE_FP8
        // Use LowPrecision if PCIe and p2p support and message size is larger than 2MB
        constexpr int LowPrecisionMinMessageSize = 2 * 1024 * 1024;
        return force_low_precision && !mIsNVLINKSupported && mIsP2PSupported
            && message_size >= LowPrecisionMinMessageSize;
#else
        // Low precision is not available when FP8 is not enabled
        return false;
#endif
    }

private:
    std::set<int> mGroup;
    bool mIsNVLINKSupported;
    bool mIsP2PSupported;
    bool mIsMNNVLSupported;
    nvinfer1::DataType mType;
    AllReduceStrategyType mStrategy;
    AllReduceFusionOp mOp;
    float mEps;
    std::variant<std::shared_ptr<ncclComm_t>, c10::intrusive_ptr<c10d::ProcessGroup>> mNcclComm;
};

} // namespace

#endif // ENABLE_MULTI_DEVICE

std::vector<torch::Tensor> allreduce_raw(torch::Tensor const& input, torch::optional<torch::Tensor> const& residual,
    torch::optional<torch::Tensor> const& norm_weight, torch::optional<torch::Tensor> const& scale,
    torch::optional<torch::Tensor> const& bias, torch::optional<torch::Tensor> workspace,
    torch::List<int64_t> const& group_, int64_t const strategy_, int64_t const fusion_op_, double const eps_,
    bool const trigger_completion_at_end_)
{
#if ENABLE_MULTI_DEVICE
    auto const dtype = tensorrt_llm::runtime::TorchUtils::dataType(input.scalar_type());
    auto const strategy = static_cast<AllReduceStrategyType>(int8_t(strategy_));
    auto const fusion_op = static_cast<AllReduceFusionOp>(int8_t(fusion_op_));
    float const eps = eps_;
    std::set<int> group;
    for (int64_t rank : group_)
    {
        group.insert(static_cast<int>(rank));
    }
    AllreduceOp op(group, dtype, strategy, fusion_op, eps);
    op.initialize();
    return op.run(input, residual, norm_weight, scale, bias, trigger_completion_at_end_, workspace);
#else
    return {input};
#endif // ENABLE_MULTI_DEVICE
}

std::vector<torch::Tensor> allreduce_pg(torch::Tensor const& input, torch::optional<torch::Tensor> const& residual,
    torch::optional<torch::Tensor> const& norm_weight, torch::optional<torch::Tensor> const& scale,
    torch::optional<torch::Tensor> const& bias, torch::optional<torch::Tensor> const& workspace,
    torch::List<int64_t> const& group_, int64_t rank, c10::intrusive_ptr<c10d::ProcessGroup> const& pg,
    int64_t const strategy_, int64_t const fusion_op_, double const eps_, bool const trigger_completion_at_end_)
{
#if ENABLE_MULTI_DEVICE
    auto const dtype = tensorrt_llm::runtime::TorchUtils::dataType(input.scalar_type());
    auto const strategy = static_cast<AllReduceStrategyType>(int8_t(strategy_));
    auto const fusion_op = static_cast<AllReduceFusionOp>(int8_t(fusion_op_));
    float const eps = eps_;
    std::set<int> group;

    for (int64_t my_rank : group_)
    {
        group.insert(static_cast<int>(my_rank));
    }

    // Get nccl rank for this process process_group_
    auto it = group.find(rank);
    if (it == group.end())
    {
        throw std::runtime_error("Rank not found in group");
    }
    int nccl_rank = std::distance(group.begin(), it);

    if (nccl_rank != pg->getRank())
    {
        throw std::runtime_error("nccl_rank != pg->getRank()");
    }

    AllreduceOp op(group, pg, dtype, strategy, fusion_op, eps);
    op.initialize();
    auto ret = op.run(input, residual, norm_weight, scale, bias, trigger_completion_at_end_, workspace);
    return ret;
#else
    return {input};
#endif // ENABLE_MULTI_DEVICE
}

// residual [m, hidden_dim]
// norm_weight [hidden_dim]
// device_num_experts [1]
// scale_input [global_num_experts, m]
// active_experts_token_input [device_num_experts, m, hidden_dim]
// token_input [m, hidden_dim]
std::vector<torch::Tensor> moe_allreduce(torch::Tensor const& residual, torch::Tensor const& norm_weight,
    torch::Tensor const& device_num_experts, torch::Tensor const& scale_input,
    torch::Tensor const& active_experts_token_input, torch::Tensor const& token_input, torch::Tensor workspace,
    int64_t const rank, int64_t const nranks, double const eps)
{
    auto allreduce_fusion_params = tensorrt_llm::kernels::ar_fusion::moe::MoeReductionAllReduceFusionParams();

    allreduce_fusion_params.quant_out = nullptr;
    allreduce_fusion_params.scale_out = nullptr;
    allreduce_fusion_params.residual_out = nullptr;
    allreduce_fusion_params.norm_out = nullptr;

    allreduce_fusion_params.nranks = static_cast<int>(nranks);
    allreduce_fusion_params.rank = static_cast<int>(rank);
    allreduce_fusion_params.dtype = tensorrt_llm::runtime::TorchUtils::dataType(token_input.scalar_type());
    // size: num_token * hidden_dim
    allreduce_fusion_params.size = static_cast<int>(token_input.numel());
    allreduce_fusion_params.hidden_dim = static_cast<int>(active_experts_token_input.size(-1));

    // workspace: AR scratch space
    allreduce_fusion_params.workspace = reinterpret_cast<void**>(workspace.mutable_data_ptr());

    allreduce_fusion_params.rms_gamma = norm_weight.data_ptr();
    allreduce_fusion_params.rms_eps = static_cast<float>(eps);
    allreduce_fusion_params.stream = at::cuda::getCurrentCUDAStream(norm_weight.get_device());

    allreduce_fusion_params.residual_in = residual.data_ptr();

    // MOE Reduction specific params
    allreduce_fusion_params.allreduce_in = nullptr; // for safety, set nullptr
    allreduce_fusion_params.moe_reduction_device_num_experts = static_cast<int*>(device_num_experts.data_ptr());
    allreduce_fusion_params.moe_reduction_scale_input = static_cast<float*>(scale_input.data_ptr());
    allreduce_fusion_params.moe_reduction_active_experts_token_input = active_experts_token_input.data_ptr();
    allreduce_fusion_params.moe_reduction_token_input = token_input.data_ptr();

    // output tensors
    torch::Tensor norm_out = torch::empty_like(token_input);
    torch::Tensor residual_out = torch::empty_like(residual);

    allreduce_fusion_params.norm_out = norm_out.mutable_data_ptr();
    allreduce_fusion_params.residual_out = residual_out.mutable_data_ptr();

    tensorrt_llm::kernels::ar_fusion::moe::moereduction_allreduce_fusion_op(allreduce_fusion_params);

    return {norm_out, residual_out};
}

// Pattern: MoE Reduction + Add + AR + ADD_RMS, see this torch reference implementation:
// expert_reduction = finalize(input, expanded_idx_to_permuted_idx, expert_scale_factor)
// output_add = expert_reduction + shared_expert_output
// output_residual = output_add + residual
// output_hidden_states = rms_norm(output_residual, norm_weight, eps)
//
// Note:
//     input is the output of MoE FC2
//     input [maxPermutedPaddedCount, hidden_dim]
//     residual [m, hidden_dim]
//     norm_weight [hidden_dim]
//     expanded_idx_to_permuted_idx [m, top_k]
//     expert_scale_factor [m, top_k]
//     shared_expert_output [m, hidden_dim]
std::vector<torch::Tensor> moe_finalize_allreduce(torch::Tensor const& input, torch::Tensor const& residual,
    torch::Tensor const& norm_weight, torch::Tensor const& expanded_idx_to_permuted_idx,
    torch::optional<torch::Tensor> const& shared_expert_output,
    torch::optional<torch::Tensor> const& expert_scale_factor, torch::Tensor workspace, int64_t const rank,
    int64_t const nranks, double const eps)
{
    auto allreduce_fusion_params = tensorrt_llm::kernels::ar_fusion::moe::MoeFinalizeAllReduceFusionParams();

    int hidden_dim = residual.size(-1);
    int top_k = expanded_idx_to_permuted_idx.size(-1);

    allreduce_fusion_params.quant_out = nullptr;
    allreduce_fusion_params.scale_out = nullptr;

    allreduce_fusion_params.nranks = static_cast<int>(nranks);
    allreduce_fusion_params.rank = static_cast<int>(rank);
    allreduce_fusion_params.dtype = tensorrt_llm::runtime::TorchUtils::dataType(input.scalar_type());
    // size: num_token * hidden_dim
    allreduce_fusion_params.size = residual.numel();
    allreduce_fusion_params.hidden_dim = hidden_dim;

    // workspace: AR scratch space
    allreduce_fusion_params.workspace = reinterpret_cast<void**>(workspace.mutable_data_ptr());
    allreduce_fusion_params.rms_gamma = norm_weight.data_ptr();
    allreduce_fusion_params.rms_eps = static_cast<float>(eps);
    allreduce_fusion_params.residual_in = residual.data_ptr();
    allreduce_fusion_params.stream = at::cuda::getCurrentCUDAStream(norm_weight.get_device());

    // MOE Reduction specific params
    allreduce_fusion_params.top_k = top_k;
    allreduce_fusion_params.allreduce_in = input.data_ptr();
    allreduce_fusion_params.expert_scale_factor
        = expert_scale_factor.has_value() ? expert_scale_factor.value().data_ptr() : nullptr;
    allreduce_fusion_params.scale_dtype = tensorrt_llm::runtime::TorchUtils::dataType(
        expert_scale_factor.has_value() ? expert_scale_factor.value().scalar_type() : input.scalar_type());
    TORCH_CHECK(
        expanded_idx_to_permuted_idx.scalar_type() == torch::kInt32, "expanded_idx_to_permuted_idx must be int32");
    allreduce_fusion_params.expanded_idx_to_permuted_idx
        = static_cast<int32_t*>(expanded_idx_to_permuted_idx.data_ptr());
    allreduce_fusion_params.shared_expert_output
        = shared_expert_output.has_value() ? shared_expert_output.value().data_ptr() : nullptr;

    // output tensors
    torch::Tensor norm_out = torch::empty_like(residual);
    torch::Tensor residual_out = torch::empty_like(residual);

    allreduce_fusion_params.norm_out = norm_out.mutable_data_ptr();
    allreduce_fusion_params.residual_out = residual_out.mutable_data_ptr();

    tensorrt_llm::kernels::ar_fusion::moe::moefinalize_allreduce_fusion_op(allreduce_fusion_params);

    return {norm_out, residual_out};
}

std::vector<torch::Tensor> mnnvlFusionAllReduce(torch::Tensor& input, torch::optional<torch::Tensor> const& gamma,
    torch::optional<torch::Tensor> const& residual_in, torch::optional<double> epsilon, torch::Tensor& comm_buffer,
    torch::Tensor& buffer_flags, bool rmsnorm_fusion)
{
    auto* mcast_mem = tensorrt_llm::common::findMcastDevMemBuffer(comm_buffer.data_ptr());
    TORCH_CHECK(
        mcast_mem != nullptr, "[mnnvlFusionAllReduce] comm_buffer must be obtained from a mcastBuffer instance.");
    TORCH_CHECK(input.is_contiguous(), "[mnnvlFusionAllReduce] input must be contiguous");

    auto const eltsPerThread = sizeof(float4) / input.itemsize();
    auto const hiddenDim = input.size(1);
    auto const numTokens = input.size(0);
    TORCH_CHECK(hiddenDim % eltsPerThread == 0,
        "[mnnvlFusionAllReduce] Hidden dimension must be divisible by " + std::to_string(eltsPerThread) + ", got "
            + std::to_string(hiddenDim));

    auto const dtype = tensorrt_llm::runtime::TorchUtils::dataType(input.scalar_type());
    torch::Tensor output = torch::empty_like(input);
    torch::Tensor residualOut;

    auto allreduce_params = tensorrt_llm::kernels::mnnvl::AllReduceFusionParams();
    allreduce_params.nRanks = mcast_mem->getWorldSize();
    allreduce_params.rank = mcast_mem->getRank();
    allreduce_params.dType = dtype;
    allreduce_params.numTokens = numTokens;
    allreduce_params.tokenDim = hiddenDim;
    allreduce_params.bufferPtrsDev = reinterpret_cast<void**>(mcast_mem->getBufferPtrsDev());
    allreduce_params.bufferPtrLocal = comm_buffer.mutable_data_ptr();
    allreduce_params.multicastPtr = mcast_mem->getMulticastPtr();
    allreduce_params.bufferFlags = reinterpret_cast<uint32_t*>(buffer_flags.mutable_data_ptr());
    allreduce_params.input = input.const_data_ptr();
    allreduce_params.output = output.mutable_data_ptr();

    if (rmsnorm_fusion)
    {
        TORCH_CHECK(residual_in.has_value() && gamma.has_value() && epsilon.has_value(),
            "[mnnvlFusionAllReduce] residual_in, gamma, and epsilon must be provided for rmsnorm fusion");
        TORCH_CHECK(residual_in.value().is_contiguous(), "[mnnvlFusionAllReduce] residual_in must be contiguous");
        TORCH_CHECK(gamma.value().is_contiguous(), "[mnnvlFusionAllReduce] gamma must be contiguous");

        allreduce_params.residualIn = residual_in.value().const_data_ptr();
        allreduce_params.gamma = gamma.value().const_data_ptr();
        allreduce_params.epsilon = static_cast<float>(epsilon.value());
        allreduce_params.rmsNormFusion = true;

        residualOut = torch::empty_like(residual_in.value());
        allreduce_params.residualOut = residualOut.mutable_data_ptr();
    }
    else
    {
        allreduce_params.rmsNormFusion = false;
    }

    allreduce_params.stream = at::cuda::getCurrentCUDAStream(output.get_device());

    // Threshold to switch between one-shot and two-shot allreduce kernel
    // Empirical value, MSG size * World size
    constexpr size_t kOneShotSizeThreshold = 16 * 4 * 8192;

    if (numTokens * hiddenDim * allreduce_params.nRanks * input.itemsize() <= kOneShotSizeThreshold)
    {
        tensorrt_llm::kernels::mnnvl::oneshotAllreduceFusionOp(allreduce_params);
    }
    else
    {
        tensorrt_llm::kernels::mnnvl::twoshotAllreduceFusionOp(allreduce_params);
    }

    return {output, residualOut};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mnnvl_fusion_allreduce(Tensor input, Tensor? residual, Tensor? gamma, "
        "float? epsilon, Tensor(a!) comm_buffer, Tensor buffer_flags, bool rmsnorm_fusion) -> "
        "Tensor[]");
    m.def(
        "allreduce("
        "Tensor input,"
        "Tensor? residual,"
        "Tensor? norm_weight,"
        "Tensor? scale,"
        "Tensor? bias,"
        "Tensor? workspace,"
        "int[] group,"
        "int strategy,"
        "int op,"
        "float eps,"
        "bool trigger_completion_at_end) -> Tensor[]");
    m.def(
        "allreduce_pg("
        "Tensor input,"
        "Tensor? residual,"
        "Tensor? norm_weight,"
        "Tensor? scale,"
        "Tensor? bias,"
        "Tensor? workspace,"
        "int[] group,"
        "int rank,"
        "__torch__.torch.classes.c10d.ProcessGroup pg,"
        "int strategy,"
        "int op,"
        "float eps,"
        "bool trigger_completion_at_end) -> Tensor[]");
    m.def(
        "moe_allreduce("
        "Tensor residual,"
        "Tensor norm_weight,"
        "Tensor device_num_experts,"
        "Tensor scale_input,"
        "Tensor active_experts_token_input,"
        "Tensor token_input,"
        "Tensor workspace,"
        "int rank,"
        "int nranks,"
        "float eps) -> Tensor[]");
    m.def("initialize_static_lowprecision_buffers(Tensor workspace, int tp_size) -> Tensor[]");
    m.def(
        "moe_finalize_allreduce("
        "Tensor input,"
        "Tensor residual,"
        "Tensor norm_weight,"
        "Tensor expanded_idx_to_permuted_idx,"
        "Tensor? shared_expert_output,"
        "Tensor? expert_scale_factor,"
        "Tensor workspace,"
        "int rank,"
        "int nranks,"
        "float eps) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mnnvl_fusion_allreduce", &tensorrt_llm::torch_ext::mnnvlFusionAllReduce);
    m.impl("allreduce", &tensorrt_llm::torch_ext::allreduce_raw);
    m.impl("allreduce_pg", &tensorrt_llm::torch_ext::allreduce_pg);
    m.impl("moe_allreduce", &tensorrt_llm::torch_ext::moe_allreduce);
    m.impl("moe_finalize_allreduce", &tensorrt_llm::torch_ext::moe_finalize_allreduce);
}

TORCH_LIBRARY_IMPL(trtllm, CPU, m)
{
    m.impl("initialize_static_lowprecision_buffers",
        [](at::Tensor const& workspace, int64_t tp_size)
        {
            tensorrt_llm::kernels::initialize_static_lowprecision_buffers(
                reinterpret_cast<int64_t*>(workspace.data_ptr()), (int) tp_size);
            return std::vector<at::Tensor>{};
        });
}
