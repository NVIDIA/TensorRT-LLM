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

#include "alltoallOp.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "thUtils.h"

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

class AllToAllHelixOp
{
public:
    AllToAllHelixOp(std::set<int> group)
        : mGroup(std::move(group))
    {
    }

    ~AllToAllHelixOp() = default;

    int initialize()
    {
        TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        mNcclComm = getComm(mGroup);
        TLLM_CHECK_WITH_INFO(mGroup.size() > 0, "group size should be greater than 0");
        TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        return 0;
    }

    std::vector<torch::Tensor> run(torch::TensorList input_list, torch::optional<int64_t> num_lists)
    {
        TLLM_CHECK_WITH_INFO(mNcclComm.get() != nullptr, "mNcclComm should be initialized before used");
        auto num_lists_ = static_cast<int>(num_lists.value_or(1));
        auto num_ranks = static_cast<int>(mGroup.size());
        // note: ensures that input_list size > 0
        TLLM_CHECK_WITH_INFO(static_cast<int>(input_list.size()) == num_ranks * num_lists_,
            "input_list size should be equal to group size * num_lists");
        for (auto const& input : input_list)
        {
            TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
        }
        std::vector<torch::Tensor> output_list(static_cast<size_t>(num_lists_));
        auto stream = at::cuda::getCurrentCUDAStream(input_list[0].get_device());
        ncclGroupStart();
        for (int il = 0; il < num_lists_; ++il)
        {
            auto off = il * num_ranks;
            auto output_shape = input_list[off].sizes().vec();
            output_shape.insert(output_shape.begin(), num_ranks);
            auto output = torch::empty(output_shape, input_list[off].options());
            output_list[il] = output;
            auto type = tensorrt_llm::runtime::TorchUtils::dataType(input_list[off].scalar_type());
            auto nccl_type = (*getDtypeMap())[type];
            for (int r = 0; r < num_ranks; ++r)
            {
                auto const& input = input_list[off + r];
                ncclSend(input.data_ptr(), input.numel(), nccl_type, r, *mNcclComm, stream);
                ncclRecv(output[r].mutable_data_ptr(), output[r].numel(), nccl_type, r, *mNcclComm, stream);
            }
        }
        NCCLCHECK_THROW(ncclGroupEnd());
        return output_list;
    }

private:
    std::set<int> mGroup;
    std::shared_ptr<ncclComm_t> mNcclComm;
};

} // namespace

#endif // ENABLE_MULTI_DEVICE

std::vector<torch::Tensor> alltoall_helix(
    torch::TensorList input_list, torch::List<int64_t> group_, torch::optional<int64_t> num_lists)
{
#if ENABLE_MULTI_DEVICE
    std::set<int> group;
    for (int64_t rank : group_)
    {
        group.insert(static_cast<int>(rank));
    }
    AllToAllHelixOp op(group);
    op.initialize();
    return op.run(input_list, num_lists);
#else
    return {};
#endif // ENABLE_MULTI_DEVICE
}

/**
 * Helix All-to-All operation with two fields.
 *
 * Input tensors have shape [..., cp_size, kv_lora_rank] for field0 and [...,
 * cp_size, 2] for field1. The operation exchanges data along the cp_size
 * dimension across all ranks.
 *
 * @param field0 Field 0 tensor (half precision, shape [..., cp_size,
 * kv_lora_rank])
 * @param field1 Field 1 tensor (float32, shape [..., cp_size, 2])
 * @param workspace Workspace tensor (uint64, strided across ranks)
 * @param cp_rank Current context parallel rank
 * @param cp_size Total number of context parallel ranks
 * @return tuple of (field0_out, field1_out) with same shapes as inputs
 */
std::tuple<torch::Tensor, torch::Tensor> helixAllToAllNative(
    torch::Tensor field0, torch::Tensor field1, torch::Tensor workspace, int64_t cp_rank, int64_t cp_size)
{

    // Input validation
    CHECK_TH_CUDA(field0);
    CHECK_TH_CUDA(field1);
    CHECK_TH_CUDA(workspace);
    CHECK_CONTIGUOUS(field0);
    CHECK_CONTIGUOUS(field1);

    // Type checks
    TORCH_CHECK(field0.scalar_type() == at::ScalarType::Half || field0.scalar_type() == at::ScalarType::BFloat16,
        "field0 must be half or bfloat16");
    CHECK_TYPE(field1, at::ScalarType::Float);
    CHECK_TYPE(workspace, at::ScalarType::UInt64);

    // Shape validation
    TORCH_CHECK(field0.dim() >= 2, "field0 must have at least 2 dimensions");
    TORCH_CHECK(field1.dim() >= 2, "field1 must have at least 2 dimensions");
    TORCH_CHECK(field0.dim() == field1.dim(), "field0 and field1 must have same number of dimensions");

    // Get dimensions
    int kv_lora_rank = field0.size(-1);
    TORCH_CHECK(field0.size(-2) == cp_size && field1.size(-2) == cp_size,
        "field0/1 second-to-last dimension must equal cp_size");
    TORCH_CHECK(
        field1.size(-1) % 2 == 0 && field1.size(-1) >= 2, "field1 last dimension must be divisible by 2 (float2)");
    bool allowVariableField1 = field1.size(-1) > 2;

    // Check that leading dimensions match
    for (int i = 0; i < field0.dim() - 2; i++)
    {
        TORCH_CHECK(
            field0.size(i) == field1.size(i), "field0 and field1 must have matching dimensions except last two");
    }
    TORCH_CHECK(field0.size(-1) * field0.element_size() % 16 == 0, "field0 must be aligned to 16 bytes");

    TORCH_CHECK(workspace.dim() == 2, "workspace must be 2D (strided across ranks)");
    TORCH_CHECK(workspace.size(0) == cp_size, "workspace must have cp_size rows");

    // Calculate entry count (product of all dimensions before cp_size)
    // This is the number of entries to process per peer rank
    int entry_count = 1;
    for (int i = 0; i < field0.dim() - 2; i++)
    {
        entry_count *= field0.size(i);
    }

    // Reshape to 3D: [entry_count, cp_size, feature_dim]
    torch::Tensor field0_3d = field0.reshape({entry_count, cp_size, kv_lora_rank});
    torch::Tensor field1_3d = field1.reshape({entry_count, cp_size, field1.size(-1)});

    // Allocate output tensors (same shape as input)
    torch::Tensor field0_out = torch::empty_like(field0);
    torch::Tensor field1_out = torch::empty_like(field1);

    torch::Tensor field0_out_3d = field0_out.reshape({entry_count, cp_size, kv_lora_rank});
    torch::Tensor field1_out_3d = field1_out.reshape({entry_count, cp_size, field1.size(-1)});

    // Setup parameters
    tensorrt_llm::kernels::HelixAllToAllParams params;

    // Field 0 (variable size half)
    params.sendFields[0].dataPtr = reinterpret_cast<uint8_t*>(field0_3d.data_ptr());
    params.sendFields[0].elementCount = kv_lora_rank;
    params.sendFields[0].elementSize = field0.element_size();
    params.sendFields[0].stride = field0_3d.stride(1) * field0.element_size();

    params.recvFields[0].dataPtr = reinterpret_cast<uint8_t*>(field0_out_3d.data_ptr());
    params.recvFields[0].elementCount = kv_lora_rank;
    params.recvFields[0].elementSize = field0.element_size();
    params.recvFields[0].stride = field0_out_3d.stride(1) * field0.element_size();

    // Field 1 (single float2)
    params.sendFields[1].dataPtr = reinterpret_cast<uint8_t*>(field1_3d.data_ptr<float>());
    params.sendFields[1].elementCount = field1.size(-1);
    params.sendFields[1].elementSize = field1.element_size();
    params.sendFields[1].stride = field1_3d.stride(1) * field1.element_size();

    params.recvFields[1].dataPtr = reinterpret_cast<uint8_t*>(field1_out_3d.data_ptr<float>());
    params.recvFields[1].elementCount = field1.size(-1);
    params.recvFields[1].elementSize = field1.element_size();
    params.recvFields[1].stride = field1_out_3d.stride(1) * field1.element_size();

    // Entry count and workspace
    params.entryCount = entry_count;
    params.workspace = workspace.data_ptr<uint64_t>();
    params.workspaceStrideInU64 = workspace.stride(0);

    // CP info
    params.cpRank = cp_rank;
    params.cpSize = cp_size;
    params.channelCount = 0; // auto-compute
    params.maxChannelCount = tensorrt_llm::kernels::computeHelixMaxChannelCount(cp_size);

    // Launch kernel
    auto stream = at::cuda::getCurrentCUDAStream();
    tensorrt_llm::kernels::launchHelixAllToAll(params, allowVariableField1, stream);

    return std::make_tuple(field0_out, field1_out);
}

/**
 * Get workspace size per rank in bytes.
 * Use dummy tensor argument to allow using torch.ops
 */
int64_t getHelixWorkspaceSizePerRank(torch::Tensor __dummy__, int64_t cp_size)
{
    return tensorrt_llm::kernels::computeHelixWorkspaceSizePerRank(cp_size);
}

/**
 * Initialize workspace for helix all-to-all
 */
void initializeHelixWorkspace(torch::Tensor workspace, int64_t cp_rank, int64_t cp_size)
{
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, at::ScalarType::UInt64);
    TORCH_CHECK(workspace.dim() == 2, "workspace must be 2D");
    TORCH_CHECK(workspace.size(0) == cp_size, "workspace must have cp_size rows");
    TORCH_CHECK(cp_rank >= 0 && cp_rank < cp_size, "cp_rank must be in [0, cp_size)");

    auto stream = at::cuda::getCurrentCUDAStream();
    uint64_t* global_workspace_ptr = workspace.data_ptr<uint64_t>();
    uint64_t* local_workspace_ptr = workspace[cp_rank].data_ptr<uint64_t>();
    TORCH_CHECK(local_workspace_ptr == global_workspace_ptr + cp_rank * workspace.stride(0),
        "local_workspace_ptr must be at the correct offset in the global "
        "workspace");
    tensorrt_llm::kernels::initializeHelixWorkspace(local_workspace_ptr, cp_size, stream);
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("alltoall_helix(Tensor[] input_list, int[] group, int? num_lists) -> Tensor[]");
    m.def(
        "helixAllToAllNative(Tensor field0, Tensor field1, Tensor workspace, int "
        "cp_rank, int cp_size) -> (Tensor, Tensor)");
    m.def("getHelixWorkspaceSizePerRank(Tensor __dummy__, int cp_size) -> int");
    m.def(
        "initializeHelixWorkspace(Tensor workspace, int cp_rank, int cp_size) "
        "-> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("alltoall_helix", &torch_ext::alltoall_helix);
    m.impl("helixAllToAllNative", &torch_ext::helixAllToAllNative);
    m.impl("getHelixWorkspaceSizePerRank", &torch_ext::getHelixWorkspaceSizePerRank);
    m.impl("initializeHelixWorkspace", &torch_ext::initializeHelixWorkspace);
}
