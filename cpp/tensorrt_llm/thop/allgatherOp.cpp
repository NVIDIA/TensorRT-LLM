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

#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"

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

class AllgatherOp
{
public:
    AllgatherOp(std::set<int> group, nvinfer1::DataType type)
        : mGroup(std::move(group))
        , mType(type)
    {
    }

    ~AllgatherOp() = default;

    int initialize() noexcept
    {
        TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        mNcclComm = getComm(mGroup);
        TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        return 0;
    }

    torch::Tensor run(torch::Tensor input) noexcept
    {
        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
        std::vector<int64_t> outputShape = input.sizes().vec();
        outputShape.insert(outputShape.begin(), mGroup.size());
        auto output = torch::empty(outputShape, input.options());
        size_t size = input.numel();
        TLLM_CHECK_WITH_INFO(mNcclComm.get() != nullptr, "mNcclComm should be initialized before used");
        NCCLCHECK(ncclAllGather(
            input.data_ptr(), output.mutable_data_ptr(), size, (*getDtypeMap())[mType], *mNcclComm, stream));
        return output;
    }

private:
    std::set<int> mGroup;
    nvinfer1::DataType mType;
    std::shared_ptr<ncclComm_t> mNcclComm;
};

} // namespace

#endif // ENABLE_MULTI_DEVICE

torch::Tensor allgather(torch::Tensor input, torch::List<int64_t> group_)
{
#if ENABLE_MULTI_DEVICE
    auto const type = tensorrt_llm::runtime::TorchUtils::dataType(input.scalar_type());
    std::set<int> group;
    for (int64_t rank : group_)
    {
        group.insert(static_cast<int>(rank));
    }
    AllgatherOp op(group, type);
    op.initialize();
    auto output = op.run(input);
    return output;
#else
    return input;
#endif // ENABLE_MULTI_DEVICE
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("allgather(Tensor input, int[] group) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("allgather", &torch_ext::allgather);
}
