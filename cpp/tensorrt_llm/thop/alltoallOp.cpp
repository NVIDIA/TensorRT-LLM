/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cassert>
#include <set>
#include <string>
#include <torch/extension.h>
#include <vector>
#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE

namespace torch_ext
{
#if ENABLE_MULTI_DEVICE

namespace
{

class AllToAllOp
{
public:
    AllToAllOp(std::set<int> group)
        : mGroup(std::move(group))
    {
    }

    ~AllToAllOp() = default;

    int initialize()
    {
        TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        mNcclComm = getComm(mGroup);
        TLLM_CHECK_WITH_INFO(mGroup.size() > 0, "group size should be greater than 0");
        TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        return 0;
    }

    void run(torch::TensorList output_list, torch::TensorList input_list, torch::optional<int64_t> num_lists)
    {
        TLLM_CHECK_WITH_INFO(mNcclComm.get() != nullptr, "mNcclComm should be initialized before used");
        auto num_lists_ = static_cast<int>(num_lists.value_or(1));
        auto num_ranks = static_cast<int>(mGroup.size());
        // note: ensures that input_list size > 0
        TLLM_CHECK_WITH_INFO(static_cast<int>(input_list.size()) == num_ranks * num_lists_,
            "input_list size should be equal to group size * num_lists");
        TLLM_CHECK_WITH_INFO(static_cast<int>(output_list.size()) == num_ranks * num_lists_,
            "output_list size should be equal to group size * num_lists");
        auto stream = at::cuda::getCurrentCUDAStream(input_list[0].get_device());
        ncclGroupStart();
        for (int il = 0; il < num_lists_; ++il)
        {
            auto off = il * num_ranks;
            auto type = tensorrt_llm::runtime::TorchUtils::dataType(input_list[off].scalar_type());
            auto nccl_type = (*getDtypeMap())[type];
            for (int r = 0; r < num_ranks; ++r)
            {
                auto const& input = input_list[off + r];
                auto& output = output_list[off + r];
                ncclSend(input.data_ptr(), input.numel(), nccl_type, r, *mNcclComm, stream);
                ncclRecv(output.mutable_data_ptr(), output.numel(), nccl_type, r, *mNcclComm, stream);
            }
        }
        NCCLCHECK_THROW(ncclGroupEnd());
    }

private:
    std::set<int> mGroup;
    std::shared_ptr<ncclComm_t> mNcclComm;
};

} // namespace

#endif // ENABLE_MULTI_DEVICE

void alltoall(torch::TensorList output_list, torch::TensorList input_list, torch::List<int64_t> group_,
    torch::optional<int64_t> num_lists)
{
#if ENABLE_MULTI_DEVICE
    std::set<int> group;
    for (int64_t rank : group_)
    {
        group.insert(static_cast<int>(rank));
    }
    AllToAllOp op(group);
    op.initialize();
    op.run(output_list, input_list, num_lists);
#endif // ENABLE_MULTI_DEVICE
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("alltoall(Tensor[] output_list, Tensor[] input_list, int[] group, int? num_lists) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("alltoall", &torch_ext::alltoall);
}
