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

#include <NvInferRuntime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE

#include <cassert>
#include <set>
#include <vector>

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
        TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        mNcclComm = getComm(mGroup);
        TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        return 0;
    }

    torch::Tensor run(torch::Tensor const& input, torch::optional<torch::List<int64_t>> sizes)
    {
        TLLM_CHECK_WITH_INFO(mNcclComm.get() != nullptr, "mNcclComm should be initialized before used");
        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
        auto type = tensorrt_llm::runtime::TorchUtils::dataType(input.scalar_type());
        std::vector<int64_t> outputShape = input.sizes().vec();
        if (sizes.has_value())
        {
            auto rank = COMM_SESSION.getRank();
            int groupRank = 0;
            for (auto const& currentRank : mGroup)
            {
                if (rank == currentRank)
                    break;
                ++groupRank;
            }
            TLLM_CHECK(static_cast<size_t>(groupRank) < mGroup.size());
            outputShape[0] = sizes.value()[groupRank];
        }
        else
        {
            outputShape[0] = outputShape[0] / mGroup.size();
        }
        auto output = torch::empty(outputShape, input.options());
        bool use_nccl_reducescatter = !sizes.has_value()
            || std::all_of(sizes.value().begin(), sizes.value().end(),
                [&sizes](int64_t size) { return size == sizes.value()[0]; });
        if (use_nccl_reducescatter)
        {
            NCCLCHECK_THROW(ncclReduceScatter(input.data_ptr(), output.mutable_data_ptr(), output.numel(),
                (*getDtypeMap())[type], ncclSum, *mNcclComm, stream));
        }
        else
        {
            size_t numel_base = std::accumulate(outputShape.cbegin() + 1, outputShape.cend(), 1, std::multiplies<>{});
            int64_t split_offset = 0;
            ncclGroupStart();
            for (int root = 0; root < static_cast<int>(mGroup.size()); ++root)
            {
                auto split_size = sizes.value()[root];
                NCCLCHECK_THROW(
                    ncclReduce(input.index({torch::indexing::Slice(split_offset, torch::indexing::None)}).data_ptr(),
                        output.mutable_data_ptr(), numel_base * split_size, (*getDtypeMap())[type], ncclSum, root,
                        *mNcclComm, stream));
                split_offset += split_size;
            }
            ncclGroupEnd();
        }
        return output;
    }

    std::vector<torch::Tensor> run_list(
        torch::TensorList input_list, torch::optional<torch::List<int64_t>> sizes) noexcept
    {
        std::vector<torch::Tensor> output_list;
        output_list.reserve(input_list.size());
        ncclGroupStart();
        for (auto const& input : input_list)
        {
            auto output = run(input, sizes);
            output_list.push_back(output);
        }
        ncclGroupEnd();
        return output_list;
    }

private:
    std::set<int> mGroup;
    std::shared_ptr<ncclComm_t> mNcclComm;
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

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("reducescatter(Tensor input, SymInt[]? sizes, int[] group) -> Tensor");
    m.def("reducescatter_list(Tensor[] input_list, SymInt[]? sizes, int[] group) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("reducescatter", &torch_ext::reducescatter);
    m.impl("reducescatter_list", &torch_ext::reducescatter_list);
}
