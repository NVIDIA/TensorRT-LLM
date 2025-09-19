/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cassert>
#include <set>
#include <vector>

using tensorrt_llm::pg_utils::PgHelper;

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
        TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, -1);
        return 0;
    }

    std::vector<torch::Tensor> run_list(torch::TensorList input_list, torch::optional<torch::List<int64_t>> sizes)
    {
        TLLM_CHECK_WITH_INFO(mNcclComm.get() != nullptr, "mNcclComm should be initialized before used");
        bool use_nccl_reducescatter = !sizes.has_value()
            || std::all_of(sizes.value().begin(), sizes.value().end(),
                [&sizes](int64_t size) { return size == sizes.value()[0]; });
        int groupRank = 0;
        if (sizes.has_value())
        {
            auto rank = COMM_SESSION.getRank();
            for (auto const& currentRank : mGroup)
            {
                if (rank == currentRank)
                    break;
                ++groupRank;
            }
            TLLM_CHECK(static_cast<size_t>(groupRank) < mGroup.size());
        }
        std::vector<torch::Tensor> output_list;
        output_list.reserve(input_list.size());
        ncclGroupStart();
        for (auto const& input : input_list)
        {
            auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
            auto type = tensorrt_llm::runtime::TorchUtils::dataType(input.scalar_type());
            std::vector<int64_t> outputShape = input.sizes().vec();
            if (sizes.has_value())
            {
                outputShape[0] = sizes.value()[groupRank];
            }
            else
            {
                outputShape[0] = outputShape[0] / mGroup.size();
            }
            auto output = torch::empty(outputShape, input.options());
            if (use_nccl_reducescatter)
            {
                ncclReduceScatter(input.data_ptr(), output.mutable_data_ptr(), output.numel(), (*getDtypeMap())[type],
                    ncclSum, *mNcclComm, stream);
            }
            else
            {
                size_t numel_base
                    = std::accumulate(outputShape.cbegin() + 1, outputShape.cend(), 1, std::multiplies<>{});
                int64_t split_offset = 0;
                for (int root = 0; root < static_cast<int>(mGroup.size()); ++root)
                {
                    auto split_size = sizes.value()[root];
                    ncclReduce(input.index({torch::indexing::Slice(split_offset, torch::indexing::None)}).data_ptr(),
                        output.mutable_data_ptr(), numel_base * split_size, (*getDtypeMap())[type], ncclSum, root,
                        *mNcclComm, stream);
                    split_offset += split_size;
                }
            }
            output_list.push_back(output);
        }
        NCCLCHECK_THROW(ncclGroupEnd());
        return output_list;
    }

    torch::Tensor run(torch::Tensor const& input, torch::optional<torch::List<int64_t>> sizes)
    {
        return run_list({input}, sizes)[0];
    }

private:
    std::set<int> mGroup;
    std::shared_ptr<ncclComm_t> mNcclComm;
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
    m.impl("reducescatter", &torch_ext::reducescatter);
    m.impl("reducescatter_pg", &torch_ext::reducescatter_pg);
    m.impl("reducescatter_list", &torch_ext::reducescatter_list);
    m.impl("reducescatter_list_pg", &torch_ext::reducescatter_list_pg);
}
