/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/pgUtils.h"

#include <NvInferRuntime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#endif // ENABLE_MULTI_DEVICE

#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <numeric>
#include <set>
#include <vector>

using tensorrt_llm::pg_utils::PgHelper;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
#if ENABLE_MULTI_DEVICE

namespace
{

class ReducescatterOp
{
public:
    ReducescatterOp(std::set<int> group)
        : mGroup(std::move(group))
    {
    }

    ~ReducescatterOp() = default;

    int initialize()
    {
        TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, -1);
        mNcclComm = getComm(mGroup);
        auto const rank = getCommWorldRank(mNcclComm);
        auto const rankIt = mGroup.find(rank);
        TLLM_CHECK_WITH_INFO(rankIt != mGroup.end(), "Global rank %d is not in the reduce-scatter group", rank);
        mGroupRank = static_cast<int>(std::distance(mGroup.begin(), rankIt));
        TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, rank);
        return 0;
    }

    std::vector<torch::Tensor> run_list(torch::TensorList input_list, torch::optional<torch::List<int64_t>> sizes)
    {
        TLLM_CHECK_WITH_INFO(mNcclComm.get() != nullptr, "mNcclComm should be initialized before used");
        bool use_nccl_reducescatter = !sizes.has_value()
            || std::all_of(sizes.value().begin(), sizes.value().end(),
                [&sizes](int64_t size) { return size == sizes.value()[0]; });
        int const groupRank = sizes.has_value() ? mGroupRank : 0;
        TLLM_CHECK_WITH_INFO(!sizes.has_value() || groupRank >= 0, "ReducescatterOp must be initialized before use");
        std::vector<torch::Tensor> output_list;
        std::vector<cudaStream_t> streams;
        bool const trackOperations = isNcclFaultToleranceEnabled();
        output_list.reserve(input_list.size());
        for (auto const& input : input_list)
        {
            std::vector<int64_t> outputShape = input.sizes().vec();
            if (sizes.has_value())
            {
                outputShape[0] = sizes.value()[groupRank];
            }
            else
            {
                outputShape[0] = outputShape[0] / mGroup.size();
            }
            output_list.push_back(torch::empty(outputShape, input.options()));
            if (trackOperations)
            {
                auto const stream = at::cuda::getCurrentCUDAStream(input.get_device());
                if (std::find(streams.begin(), streams.end(), stream) == streams.end())
                {
                    streams.push_back(stream);
                }
            }
        }

        auto commLease = acquireComm(mNcclComm);
        auto const comm = commLease.get();
        std::vector<uint64_t> watchdogTokens;
        if (trackOperations)
        {
            watchdogTokens.reserve(streams.size());
            for (auto const stream : streams)
            {
                watchdogTokens.push_back(commLease.begin(stream, "ncclReduceScatter"));
            }
        }
        commLease.groupStart("ncclGroupStart(reducescatter)");
        for (size_t index = 0; index < input_list.size(); ++index)
        {
            auto const& input = input_list[index];
            auto& output = output_list[index];
            auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
            auto type = tensorrt_llm::runtime::TorchUtils::dataType(input.scalar_type());
            if (use_nccl_reducescatter)
            {
                commLease.checkEnqueue(ncclReduceScatter(input.data_ptr(), output.mutable_data_ptr(), output.numel(),
                                           (*getDtypeMap())[type], ncclSum, comm, stream),
                    "ncclReduceScatter");
            }
            else
            {
                auto const& outputShape = output.sizes();
                size_t numel_base
                    = std::accumulate(outputShape.cbegin() + 1, outputShape.cend(), 1, std::multiplies<>{});
                int64_t split_offset = 0;
                for (int root = 0; root < static_cast<int>(mGroup.size()); ++root)
                {
                    auto split_size = sizes.value()[root];
                    commLease.checkEnqueue(
                        ncclReduce(
                            input.index({torch::indexing::Slice(split_offset, torch::indexing::None)}).data_ptr(),
                            output.mutable_data_ptr(), numel_base * split_size, (*getDtypeMap())[type], ncclSum, root,
                            comm, stream),
                        "ncclReduce(reducescatter)");
                    split_offset += split_size;
                }
            }
        }
        commLease.groupEnd("ncclGroupEnd(reducescatter)");
        if (trackOperations)
        {
            for (size_t index = 0; index < streams.size(); ++index)
            {
                commLease.track(watchdogTokens[index], streams[index]);
            }
        }
        return output_list;
    }

    torch::Tensor run(torch::Tensor const& input, torch::optional<torch::List<int64_t>> sizes)
    {
        return run_list({input}, sizes)[0];
    }

private:
    std::set<int> mGroup;
    std::shared_ptr<ncclComm_t> mNcclComm;
    int mGroupRank{-1};
};

class ReducescatterPgOp
{
public:
    ReducescatterPgOp(std::set<int> group, c10::intrusive_ptr<c10d::ProcessGroup> const& process_group_)
        : mGroup(std::move(group))
        , mProcessGroup(process_group_)
    {
    }

    ~ReducescatterPgOp() = default;

    int initialize() noexcept
    {
        TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, mProcessGroup->getRank());
        return 0;
    }

    std::pair<torch::Tensor, c10::intrusive_ptr<c10d::Work>> run(
        torch::Tensor input, torch::optional<torch::List<int64_t>> sizes, bool coalescing = false)
    {
        TLLM_CHECK_WITH_INFO(mProcessGroup.get() != nullptr, "mProcessGroup should be initialized before used");
        auto rank = mProcessGroup->getRank();
        std::vector<int64_t> outputShape = input.sizes().vec();
        if (sizes.has_value())
        {
            TLLM_CHECK(sizes.value().size() == mGroup.size());
            outputShape[0] = sizes.value()[rank];
        }
        else
        {
            outputShape[0] = outputShape[0] / mGroup.size();
        }
        auto output = torch::empty(outputShape, input.options());

        int64_t split_offset = 0;
        std::vector<torch::Tensor> inputTensors{};
        for (int root = 0; root < static_cast<int>(mGroup.size()); ++root)
        {
            auto split_size = sizes.has_value() ? sizes.value()[root] : outputShape[0];
            inputTensors.push_back(input.index({torch::indexing::Slice(split_offset, split_offset + split_size)}));
            split_offset += split_size;
        }
        std::vector<torch::Tensor> outputs{output};
        std::vector<std::vector<torch::Tensor>> inputs{inputTensors};
        auto work = mProcessGroup->reduce_scatter(outputs, inputs, {});

        if (!coalescing)
        {
            PGCHECK_THROW_WITH_INFO(work, "ProcessGroup: reduce_scatter");
            return {output, nullptr};
        }

        return {output, work};
    }

    std::vector<torch::Tensor> run_list(torch::TensorList input_list, torch::optional<torch::List<int64_t>> sizes)
    {
        std::vector<torch::Tensor> output_list;
        std::vector<c10::intrusive_ptr<c10d::Work>> work_list;
        output_list.reserve(input_list.size());
        work_list.reserve(input_list.size());
        mProcessGroup->startCoalescing(c10::DeviceType::CUDA);
        for (auto const& input : input_list)
        {
            auto [output, work] = run(input, sizes, true);
            output_list.push_back(output);
            work_list.push_back(work); // Hold work objects (input & output tensors) until endCoalescing wait finished
        }
        if (auto work = mProcessGroup->endCoalescing(c10::DeviceType::CUDA))
        {
            PGCHECK_THROW_WITH_INFO(work, "ProcessGroup: reduce_scatter, end coalescing");
        }
        return output_list;
    }

private:
    std::set<int> mGroup;
    c10::intrusive_ptr<c10d::ProcessGroup> mProcessGroup;
};

} // namespace

#endif // ENABLE_MULTI_DEVICE

extern torch::Tensor reducescatter(
    torch::Tensor input, torch::optional<torch::List<int64_t>> sizes, torch::List<int64_t> group_)
{
#if ENABLE_MULTI_DEVICE
    std::set<int> group;
    for (int64_t rank : group_)
    {
        group.insert(static_cast<int>(rank));
    }
    ReducescatterOp op(group);
    op.initialize();
    auto output = op.run(input, sizes);
    return output;
#else
    return input;
#endif // ENABLE_MULTI_DEVICE
}

extern torch::Tensor reducescatter_pg(torch::Tensor input, torch::optional<torch::List<int64_t>> sizes,
    torch::List<int64_t> group_, c10::intrusive_ptr<c10d::ProcessGroup> const& process_group_)
{
#if ENABLE_MULTI_DEVICE
    std::set<int> group;
    for (int64_t rank : group_)
    {
        group.insert(static_cast<int>(rank));
    }
    ReducescatterPgOp op(group, process_group_);
    op.initialize();
    auto [output, _] = op.run(input, sizes);
    return output;
#else
    return input;
#endif // ENABLE_MULTI_DEVICE
}

extern std::vector<torch::Tensor> reducescatter_list(
    torch::TensorList input_list, torch::optional<torch::List<int64_t>> sizes, torch::List<int64_t> group_)
{
#if ENABLE_MULTI_DEVICE
    std::set<int> group;
    for (int64_t rank : group_)
    {
        group.insert(static_cast<int>(rank));
    }
    ReducescatterOp op(group);
    op.initialize();
    auto output_list = op.run_list(input_list, sizes);
    return output_list;
#else
    return input_list.vec();
#endif // ENABLE_MULTI_DEVICE
}

extern std::vector<torch::Tensor> reducescatter_list_pg(torch::TensorList input_list,
    torch::optional<torch::List<int64_t>> sizes, torch::List<int64_t> group_,
    c10::intrusive_ptr<c10d::ProcessGroup> const& process_group_)
{
#if ENABLE_MULTI_DEVICE
    std::set<int> group;
    for (int64_t rank : group_)
    {
        group.insert(static_cast<int>(rank));
    }
    ReducescatterPgOp op(group, process_group_);
    op.initialize();
    auto output_list = op.run_list(input_list, sizes);
    return output_list;
#else
    return input_list.vec();
#endif // ENABLE_MULTI_DEVICE
}
} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("reducescatter(Tensor input, SymInt[]? sizes, int[] group) -> Tensor");
    m.def(
        "reducescatter_pg(Tensor input, SymInt[]? sizes, int[] group, __torch__.torch.classes.c10d.ProcessGroup "
        "process_group) -> Tensor");
    m.def("reducescatter_list(Tensor[] input_list, SymInt[]? sizes, int[] group) -> Tensor[]");
    m.def(
        "reducescatter_list_pg(Tensor[] input_list, SymInt[]? sizes, int[] group, "
        "__torch__.torch.classes.c10d.ProcessGroup process_group) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("reducescatter", &tensorrt_llm::torch_ext::reducescatter);
    m.impl("reducescatter_pg", &tensorrt_llm::torch_ext::reducescatter_pg);
    m.impl("reducescatter_list", &tensorrt_llm::torch_ext::reducescatter_list);
    m.impl("reducescatter_list_pg", &tensorrt_llm::torch_ext::reducescatter_list_pg);
}
