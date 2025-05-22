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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/customAllReduceUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/mcastDevMemUtils.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.h"
#include "tensorrt_llm/kernels/communicationKernels/customLowPrecisionAllReduceKernels.h"
#include "tensorrt_llm/kernels/communicationKernels/mnnvlTwoShotAllreduceKernels.h"
#include "tensorrt_llm/kernels/communicationKernels/moeAllReduceFusionKernels.h"
#include "tensorrt_llm/kernels/customAllReduceKernels.h"
#include "tensorrt_llm/kernels/internal_cutlass_kernels/include/fp4_gemm.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/kernels/userbuffers/ub_interface.h"
#include "tensorrt_llm/runtime/mcastDeviceMemory.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/thop/fp4Quantize.h"
#include "tensorrt_llm/thop/fp8Op.h"
#include "tensorrt_llm/thop/thUtils.h"
#include "tensorrt_llm/thop/userbuffersTensor.h"

#if ENABLE_MULTI_DEVICE
#include <ATen/cuda/EmptyTensor.h>
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE
#include <nvml.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <unordered_set>

// using namespace nvinfer1;
using tensorrt_llm::kernels::AllReduceFusionOp;
using tensorrt_llm::kernels::AllReduceStrategyType;
using tensorrt_llm::mpi::MpiTag;

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

class AllreduceOp
{
public:
    AllreduceOp(
        std::set<int> group, nvinfer1::DataType type, AllReduceStrategyType strategy, AllReduceFusionOp op, float eps)
        : mGroup(std::move(group))
        , mType(type)
        , mStrategy(strategy)
        , mOp(op)
        , mEps(eps)
    {
    }

    ~AllreduceOp() = default;

    std::vector<torch::Tensor> run(torch::Tensor const& input, torch::optional<torch::Tensor> const& residual,
        torch::optional<torch::Tensor> const& norm_weight, torch::optional<torch::Tensor> const& scale,
        torch::optional<torch::Tensor> const& bias, torch::optional<torch::Tensor> workspace) noexcept
    {
        size_t size = input.numel();
        size_t seq_len = input.size(0);

        // If strategy is set to UB, UB must be used as UB impl output is special and cannot be used
        // by others.
        AllReduceStrategyType runtime_strategy = getRuntimeStrategy(seq_len, size);

        // Log runtime strategy
        auto const rank = COMM_SESSION.getRank();
        logRunTimeStrategy(runtime_strategy, rank);

        // Dispatch to different allreduce implementations
        switch (runtime_strategy)
        {
        case AllReduceStrategyType::UB: return runUBAllReduce(input, residual, norm_weight, scale, bias);
        case AllReduceStrategyType::NCCL: return runNCCLAllReduce(input, residual, norm_weight, scale, bias);
        case AllReduceStrategyType::MIN_LATENCY:
        case AllReduceStrategyType::ONESHOT:
        case AllReduceStrategyType::TWOSHOT:
            return runFusionAllReduce(input, residual, norm_weight, scale, bias, workspace, runtime_strategy);
        case AllReduceStrategyType::LOWPRECISION:
            return runLowPrecisionAllReduce(input, residual, norm_weight, scale, bias);
        default: TORCH_CHECK(false, "Invalid runtime strategy"); return {};
        }
    }

    int initialize() noexcept
    {
        TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        mNcclComm = getComm(mGroup);
        if (mStrategy != AllReduceStrategyType::NCCL && mStrategy != AllReduceStrategyType::UB)
        {
            initGroupTopology();
        }

        TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        return 0;
    }

private:
    std::vector<torch::Tensor> runUBAllReduce(torch::Tensor const& input,
        torch::optional<torch::Tensor> const& residual, torch::optional<torch::Tensor> const& norm_weight,
        torch::optional<torch::Tensor> const& scale, torch::optional<torch::Tensor> const& bias) noexcept
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
        torch::optional<torch::Tensor> const& scale, torch::optional<torch::Tensor> const& bias) noexcept
    {

        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
        int size = input.numel();

        torch::Tensor reduce_output = torch::empty_like(input);
        NCCLCHECK(ncclAllReduce(input.data_ptr(), reduce_output.mutable_data_ptr(), size, (*getDtypeMap())[mType],
            ncclSum, *mNcclComm, stream));

        if (mOp == AllReduceFusionOp::NONE)
        {
            return {reduce_output};
        }

        // Treat any other patterns as fallback cases.
        return fallbackRunSubsequentOps(input, residual, norm_weight, scale, bias, reduce_output);
    }

    std::vector<torch::Tensor> runLowPrecisionAllReduce(torch::Tensor const& input,
        torch::optional<torch::Tensor> const& residual, torch::optional<torch::Tensor> const& norm_weight,
        torch::optional<torch::Tensor> const& scale, torch::optional<torch::Tensor> const& bias) noexcept
    {
#ifdef ENABLE_FP8
        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
        int size = input.numel();
        int hidden_size = input.size(-1);

        auto const tp_size = mGroup.size();
        auto const cur_rank = COMM_SESSION.getRank();
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
        torch::optional<torch::Tensor> workspace, AllReduceStrategyType strategy) noexcept
    {
        // Should handle only Lamport implementation
        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
        int size = input.numel();
        int hidden_size = input.size(-1);
        int seq_len = input.size(0);

        auto const tp_size = mGroup.size();
        auto const cur_rank = COMM_SESSION.getRank();
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

        // Determine if using oneshot or twoshot allreduce kernel
        if (strategy == AllReduceStrategyType::MIN_LATENCY)
        {
            allreduce_fusion_params.use_oneshot = seq_len <= tensorrt_llm::kernels::ar_fusion::kOneShotMaxToken;
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
            scale_out = at::detail::empty_cuda({tensorrt_llm::computeFP4SwizzledLayoutSFSize(m, k / sf_vec_size)},
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
        torch::Tensor& reduce_output) noexcept
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

        const int64_t sf_vecsize = 16;
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

    AllReduceStrategyType getRuntimeStrategy(size_t seq_len, size_t size) noexcept
    {
        static char* force_nccl_all_reduce_strategy_char = std::getenv("FORCE_NCCL_ALL_REDUCE_STRATEGY");
        bool force_nccl_all_reduce_strategy = (force_nccl_all_reduce_strategy_char != nullptr);
        AllReduceStrategyType runtime_strategy;
        if (mStrategy == AllReduceStrategyType::UB)
        {
            runtime_strategy = AllReduceStrategyType::UB;
        }
        else if (force_nccl_all_reduce_strategy || mStrategy == AllReduceStrategyType::NCCL)
        {
            runtime_strategy = AllReduceStrategyType::NCCL;
        }
        else
        {
            // This is for DEBUG and BENCHMARK purpose. It will overried the strategy if AUTO is set.
            static char* ifForBenchMark = std::getenv("OVERRIDE_HEURISTIC_ALLREDUCE_STRATEGY");
            if (ifForBenchMark != nullptr)
            {
                runtime_strategy = mStrategy;
            }
            else
            {
                runtime_strategy = selectImplementation(seq_len, size, mGroup.size(), mType);
            }
        }
        return runtime_strategy;
    }

    void logRunTimeStrategy(AllReduceStrategyType strategy, int rank) noexcept
    {
        switch (strategy)
        {
        case AllReduceStrategyType::NCCL:
        {
            TLLM_LOG_DEBUG("AllReducePlugin strategy for rank %d: NCCL", rank);
            break;
        }
        case AllReduceStrategyType::MIN_LATENCY:
        {
            TLLM_LOG_DEBUG("AllReducePlugin strategy for rank %d: MIN_LATENCY", rank);
            break;
        }
        case AllReduceStrategyType::UB:
        {
            TLLM_LOG_DEBUG("AllReducePlugin strategy for rank %d: UB", rank);
            break;
        }
        case AllReduceStrategyType::LOWPRECISION:
        {
            TLLM_LOG_DEBUG("AllReducePlugin strategy for rank %d: LOWPRECISION", rank);
            break;
        }
        default: break;
        }
    }

    bool Fusable() noexcept
    {
        return mOp != AllReduceFusionOp::NONE;
    }

    void initGroupTopology() noexcept
    {
        static std::map<std::set<int>, std::tuple<bool, bool>> cache;
        if (cache.find(mGroup) != cache.end())
        {
            auto [is_NVLINK_supported, is_P2P_supported] = cache[mGroup];
            mIsNVLINKSupported = is_NVLINK_supported;
            mIsP2PSupported = is_P2P_supported;
            return;
        }
        setGroupTopology();
        cache[mGroup] = {mIsNVLINKSupported, mIsP2PSupported};
    }

    void setGroupTopology() noexcept
    {
        auto const rank = COMM_SESSION.getRank();
        TLLM_LOG_INFO("Detecting local TP group for rank %d", rank);
        std::set<int> local_group = getLocalGroup(mGroup);
        if (mGroup.size() != local_group.size())
        {
            mIsP2PSupported = false;
            mIsNVLINKSupported = false;
            TLLM_LOG_INFO("Found inter-node TP group for rank %d", rank);
            return;
        }
        TLLM_LOG_INFO("TP group is intra-node for rank %d", rank);

        NvmlManager nvml_manager;
        std::unordered_set<int> visited_device;
        mIsP2PSupported = true;
        mIsNVLINKSupported = true;

        // Use cudaDeviceCanAccessPeer to determine whether p2p is supported,
        // and use nvml to determine whether there are nvlink links between ranks.
        for (int first_device_id : local_group)
        {
            for (int second_device_id : local_group)
            {
                if (first_device_id == second_device_id
                    || visited_device.find(second_device_id) != visited_device.end())
                {
                    continue;
                }

                int can_access_peer = 0;
                TLLM_CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, first_device_id, second_device_id));

                if (!can_access_peer)
                {
                    mIsP2PSupported = false;
                    mIsNVLINKSupported = false;

                    return;
                }

                nvmlDevice_t first_device;
                NVML_CHECK(nvmlDeviceGetHandleByIndex(first_device_id, &first_device));

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
                        NVML_CHECK(nvmlDeviceGetIndex(remote_device, &remote_device_id));

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
                        NVML_CHECK(nvmlDeviceGetHandleByIndex(second_device_id, &second_device));

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
                        NVML_CHECK(result);
                    }

                    if (is_NVLINK)
                    {
                        break;
                    }
                }

                mIsNVLINKSupported &= is_NVLINK;
            }
            visited_device.insert(first_device_id);
        }
    }

    bool ifFallbackToNCCL(size_t seq_len, size_t message_size_bytes, size_t max_workspace_size, bool is_auto) noexcept
    {
        // If messageSize is less than maxWorkspaceSize, use NCCL, regardless of the fusion type.
        if (message_size_bytes > max_workspace_size)
        {
            if (!is_auto)
            {
                TLLM_LOG_WARNING(
                    "Since messageSize is greater than maxWorkspaceSize, fallback to AllReduceStrategy: NCCL");
            }
            return true;
        }

        // If Peer to Peer is not supported, fallback to NCCL.
        if (!mIsP2PSupported)
        {
            if (!is_auto)
            {
                TLLM_LOG_WARNING("Since Peer to Peer not supported, fallback to AllReduceStrategy: NCCL");
            }
            return true;
        }

        // If NVLINK is not supported, fallback to NCCL.
        if (!mIsNVLINKSupported)
        {
            if (!is_auto)
            {
                TLLM_LOG_WARNING("Since NVLINK not supported, fallback to AllReduceStrategy: NCCL");
            }
            return true;
        }
        return false;
    }

    AllReduceStrategyType selectImplementation(
        size_t seq_len, size_t message_size, int world_size, nvinfer1::DataType type) noexcept
    {

        if (isUsingLowPrecision(message_size))
        {
            return AllReduceStrategyType::LOWPRECISION;
        }
        else
        {
            if (mStrategy == AllReduceStrategyType::LOWPRECISION)
            {
                mStrategy = AllReduceStrategyType::AUTO;
            }
        }

        // Check that heuristic is only applied when AUTO is set.
        // Use Auto select
        bool const is_auto = (mStrategy == AllReduceStrategyType::AUTO);
        auto const message_size_bytes = message_size * tensorrt_llm::common::getDTypeSize(type);
        auto const max_workspace_size
            = tensorrt_llm::utils::customAllReduceUtils::getMaxRequiredWorkspaceSize(world_size);

        if (ifFallbackToNCCL(seq_len, message_size_bytes, max_workspace_size, is_auto))
        {
            return AllReduceStrategyType::NCCL;
        }

        // This rule based heuristic only chooses between NCCL and MIN_LATENCY strategies.

        // Heurisitic will only be applied on NONE and RESIDUAL_RMS_NORM fusion types.
        // Because NCCL might be faster on some large messageSize cases.
        // Otherwise, MIN_LATENCY strategy will be directly returned due to more fusions it can support.
        // TODO: NCCL AllReduce + subsequent quantization ops (as fallback) can also support the fusion types.
        // This should be compared with MIN_LATENCY fused kernels to determine the best strategy.
        switch (mOp)
        {
        case AllReduceFusionOp::NONE:
        case AllReduceFusionOp::RESIDUAL_RMS_NORM: break;
        case AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_FP8:
        case AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_FP8:
        case AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4:
        case AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4: return AllReduceStrategyType::MIN_LATENCY;
        // Suppose NCCL has fallback implementations for all fusion types.
        default: return AllReduceStrategyType::NCCL;
        }

        // Check mOp to be supported by the heuristic.
        TORCH_CHECK(mOp == AllReduceFusionOp::NONE || mOp == AllReduceFusionOp::RESIDUAL_RMS_NORM,
            "Only NONE and RESIDUAL_RMS_NORM are supported for NCCL/MIN_LATENCY heuristic.");

        // Default to NCCL.
        AllReduceStrategyType strategy = AllReduceStrategyType::NCCL;

        // Currently we will not remove ONESHOT and TWOSHOT from the strategy list
        // But torch flow user should not use them, but use AUTO or MIN_LATENCY instead.
        // NOTICE: When a fusion type is not supported by the corresponding strategy but strategy is not AUTO,
        // user should guarantee the correctness of the fusion pattern dispatching.
        if (!is_auto)
        {
            if (mStrategy == AllReduceStrategyType::ONESHOT || mStrategy == AllReduceStrategyType::TWOSHOT)
            {
                strategy = AllReduceStrategyType::MIN_LATENCY;
            }
            else
            {
                strategy = mStrategy;
            }
        }
        else if (world_size <= 2)
        {
            strategy = AllReduceStrategyType::MIN_LATENCY;
        }
        else if (world_size <= 4)
        {
            if (message_size_bytes < 1 * 1000 * 1000)
            {
                strategy = AllReduceStrategyType::MIN_LATENCY;
            }
            else
            {
                strategy = AllReduceStrategyType::NCCL;
            }
        }
        else
        {
            if (message_size_bytes < 500 * 1000)
            {
                strategy = AllReduceStrategyType::MIN_LATENCY;
            }
            else
            {
                strategy = AllReduceStrategyType::NCCL;
            }
        }
        return strategy;
    }

    bool isUsingLowPrecision(size_t message_size) const noexcept
    {
        static char* force_low_precision_allreduce_strategy_char
            = std::getenv("FORCE_LOW_PRECISION_ALL_REDUCE_STRATEGY");
        bool force_low_precision = (force_low_precision_allreduce_strategy_char != nullptr)
            || (mStrategy == AllReduceStrategyType::LOWPRECISION);

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
    nvinfer1::DataType mType;
    AllReduceStrategyType mStrategy;
    AllReduceFusionOp mOp;
    float mEps;
    std::shared_ptr<ncclComm_t> mNcclComm;
};

} // namespace

#endif // ENABLE_MULTI_DEVICE

std::vector<torch::Tensor> allreduce(torch::Tensor const& input, torch::optional<torch::Tensor> const& residual,
    torch::optional<torch::Tensor> const& norm_weight, torch::optional<torch::Tensor> const& scale,
    torch::optional<torch::Tensor> const& bias, torch::optional<torch::Tensor> const& workspace,
    torch::List<int64_t> const& group_, int64_t const strategy_, int64_t const fusion_op_, double const eps_)
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
    return op.run(input, residual, norm_weight, scale, bias, workspace);
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

at::Tensor mnnvlTwoShotAllReduce(
    at::Tensor& output, at::Tensor& input, at::Tensor& comm_buffer, at::Tensor& buffer_flags, bool wait_for_results)
{
    auto* mcast_mem = tensorrt_llm::common::findMcastDevMemBuffer(comm_buffer.data_ptr());
    TORCH_CHECK(mcast_mem != nullptr, "two_shot_all_reduce: comm_buffer must be obtained from a mcastBuffer instance.");

    auto const dtype = tensorrt_llm::runtime::TorchUtils::dataType(input.scalar_type());

    auto allreduce_params = tensorrt_llm::kernels::mnnvl::AllReduceParams();
    allreduce_params.dtype = dtype;
    allreduce_params.output = output.data_ptr();
    allreduce_params.input = input.data_ptr();
    allreduce_params.buffer_flags = buffer_flags.data_ptr();
    allreduce_params.wait_for_results = wait_for_results;
    allreduce_params.stream = at::cuda::getCurrentCUDAStream(output.get_device());
    allreduce_params.nranks = mcast_mem->getWorldSize();
    allreduce_params.rank = mcast_mem->getRank();
    allreduce_params.buffer_M = comm_buffer.size(2);
    allreduce_params.num_tokens = input.size(0);
    allreduce_params.token_dim = input.size(1);
    allreduce_params.buffer_ptrs_dev = reinterpret_cast<void**>(mcast_mem->getBufferPtrsDev());
    allreduce_params.multicast_ptr = mcast_mem->getMulticastPtr();

    tensorrt_llm::kernels::mnnvl::twoshot_allreduce_op(allreduce_params);

    return output;
}

void twoShotRMSNorm(torch::Tensor& prenorm_output, torch::Tensor& normed_output, torch::Tensor const& input,
    torch::Tensor const& gamma, double epsilon, torch::Tensor const& residual, torch::Tensor& buffer_flags)
{
    auto const dtype = tensorrt_llm::runtime::TorchUtils::dataType(input.scalar_type());
    auto rmsnorm_params = tensorrt_llm::kernels::mnnvl::RMSNormParams();
    rmsnorm_params.dtype = dtype;
    rmsnorm_params.residual_output = prenorm_output.data_ptr();
    rmsnorm_params.output = normed_output.data_ptr();
    rmsnorm_params.input = input.data_ptr();
    rmsnorm_params.gamma = gamma.data_ptr();
    rmsnorm_params.epsilon = epsilon;
    rmsnorm_params.residual = residual.data_ptr();
    rmsnorm_params.buffer_flags = reinterpret_cast<uint32_t*>(buffer_flags.data_ptr());
    rmsnorm_params.batch = normed_output.size(0);
    rmsnorm_params.hidden_dim = normed_output.size(1);
    rmsnorm_params.stream = at::cuda::getCurrentCUDAStream(input.get_device());

    tensorrt_llm::kernels::mnnvl::twoshot_rmsnorm_op(rmsnorm_params);
}
} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mnnvl_twoshot_allreduce(Tensor(output!) output, Tensor(input!) input, Tensor(comm_buf!) comm_buffer, "
        "Tensor(buffer_flags!) buffer_flags, bool wait_for_result) -> Tensor");
    m.def(
        "mnnvl_twoshot_rmsnorm(Tensor prenorm_output, Tensor normed_output, Tensor input, Tensor gamma, "
        "float epsilon, Tensor residual, Tensor buffer_flags) -> ()");
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
        "float eps) -> Tensor[]");
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
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mnnvl_twoshot_allreduce", &torch_ext::mnnvlTwoShotAllReduce);
    m.impl("mnnvl_twoshot_rmsnorm", &torch_ext::twoShotRMSNorm);
    m.impl("allreduce", &torch_ext::allreduce);
    m.impl("moe_allreduce", &torch_ext::moe_allreduce);
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
