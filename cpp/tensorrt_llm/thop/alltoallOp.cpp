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
#include "tensorrt_llm/kernels/helixAllToAll.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <vector>

TRTLLM_NAMESPACE_BEGIN

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
 * Input tensors have shape [..., cp_size, kv_lora_rank] for partial_o and [...,
 * cp_size, 2] for softmax_stats. The operation exchanges data along the cp_size
 * dimension across all ranks.
 *
 * @param partial_o Field 0 tensor (half precision, shape [..., cp_size,
 * kv_lora_rank])
 * @param softmax_stats Field 1 tensor (float32, shape [..., cp_size, 2])
 * @param workspace Workspace tensor (uint64, strided across ranks)
 * @param cp_rank Current context parallel rank
 * @param cp_size Total number of context parallel ranks
 * @return tuple of (partial_o_out, softmax_stats_out) with same shapes as inputs
 */
std::tuple<torch::Tensor, torch::Tensor> alltoall_helix_native(
    torch::Tensor partial_o, torch::Tensor softmax_stats, torch::Tensor workspace, int64_t cp_rank, int64_t cp_size)
{

    // Input validation
    CHECK_TH_CUDA(partial_o);
    CHECK_TH_CUDA(softmax_stats);
    CHECK_TH_CUDA(workspace);
    CHECK_CONTIGUOUS(partial_o);
    CHECK_CONTIGUOUS(softmax_stats);

    // Type checks
    TORCH_CHECK(partial_o.scalar_type() == at::ScalarType::Half || partial_o.scalar_type() == at::ScalarType::BFloat16,
        "partial_o must be half or bfloat16");
    CHECK_TYPE(softmax_stats, at::ScalarType::Float);
    CHECK_TYPE(workspace, at::ScalarType::UInt64);

    // Shape validation
    TORCH_CHECK(partial_o.dim() >= 2, "partial_o must have at least 2 dimensions");
    TORCH_CHECK(softmax_stats.dim() >= 2, "softmax_stats must have at least 2 dimensions");
    TORCH_CHECK(
        partial_o.dim() == softmax_stats.dim(), "partial_o and softmax_stats must have same number of dimensions");

    // Get dimensions
    int kv_lora_rank = partial_o.size(-1);
    TORCH_CHECK(partial_o.size(-2) == cp_size && softmax_stats.size(-2) == cp_size,
        "partial_o/softmax_stats second-to-last dimension must equal cp_size");
    TORCH_CHECK(softmax_stats.size(-1) % 2 == 0 && softmax_stats.size(-1) >= 2,
        "softmax_stats last dimension must be divisible by 2 (float2)");
    bool allowVariableField1 = softmax_stats.size(-1) > 2;

    // Check that leading dimensions match
    for (int i = 0; i < partial_o.dim() - 2; i++)
    {
        TORCH_CHECK(partial_o.size(i) == softmax_stats.size(i),
            "partial_o and softmax_stats must have matching dimensions except last two");
    }
    TORCH_CHECK(partial_o.size(-1) * partial_o.element_size() % 16 == 0, "partial_o must be aligned to 16 bytes");

    TORCH_CHECK(workspace.dim() == 2, "workspace must be 2D (strided across ranks)");
    TORCH_CHECK(workspace.size(0) == cp_size, "workspace must have cp_size rows");

    // Calculate entry count (product of all dimensions before cp_size)
    // This is the number of entries to process per peer rank
    int entry_count = 1;
    for (int i = 0; i < partial_o.dim() - 2; i++)
    {
        entry_count *= partial_o.size(i);
    }

    // Reshape to 3D: [entry_count, cp_size, feature_dim]
    torch::Tensor partial_o_3d = partial_o.reshape({entry_count, cp_size, kv_lora_rank});
    torch::Tensor softmax_stats_3d = softmax_stats.reshape({entry_count, cp_size, softmax_stats.size(-1)});

    // Allocate output tensors (same shape as input)
    torch::Tensor partial_o_out = torch::empty_like(partial_o);
    torch::Tensor softmax_stats_out = torch::empty_like(softmax_stats);

    torch::Tensor partial_o_out_3d = partial_o_out.reshape({entry_count, cp_size, kv_lora_rank});
    torch::Tensor softmax_stats_out_3d = softmax_stats_out.reshape({entry_count, cp_size, softmax_stats.size(-1)});

    // Setup parameters
    tensorrt_llm::kernels::HelixAllToAllParams params;

    // Field 0 (variable size half)
    params.sendFields[0].dataPtr = reinterpret_cast<uint8_t*>(partial_o_3d.data_ptr());
    params.sendFields[0].elementCount = kv_lora_rank;
    params.sendFields[0].elementSize = partial_o.element_size();
    params.sendFields[0].stride = partial_o_3d.stride(1) * partial_o.element_size();

    params.recvFields[0].dataPtr = reinterpret_cast<uint8_t*>(partial_o_out_3d.data_ptr());
    params.recvFields[0].elementCount = kv_lora_rank;
    params.recvFields[0].elementSize = partial_o.element_size();
    params.recvFields[0].stride = partial_o_out_3d.stride(1) * partial_o.element_size();

    // Field 1 (single float2)
    params.sendFields[1].dataPtr = reinterpret_cast<uint8_t*>(softmax_stats_3d.data_ptr<float>());
    params.sendFields[1].elementCount = softmax_stats.size(-1);
    params.sendFields[1].elementSize = softmax_stats.element_size();
    params.sendFields[1].stride = softmax_stats_3d.stride(1) * softmax_stats.element_size();

    params.recvFields[1].dataPtr = reinterpret_cast<uint8_t*>(softmax_stats_out_3d.data_ptr<float>());
    params.recvFields[1].elementCount = softmax_stats.size(-1);
    params.recvFields[1].elementSize = softmax_stats.element_size();
    params.recvFields[1].stride = softmax_stats_out_3d.stride(1) * softmax_stats.element_size();

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

    return std::make_tuple(partial_o_out, softmax_stats_out);
}

/**
 * Initialize workspace for helix all-to-all
 */
void initialize_helix_workspace(torch::Tensor workspace, int64_t cp_rank, int64_t cp_size)
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

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("alltoall_helix(Tensor[] input_list, int[] group, int? num_lists) -> Tensor[]");
    m.def(
        "alltoall_helix_native(Tensor partial_o, Tensor softmax_stats, Tensor(a!) workspace, int "
        "cp_rank, int cp_size) -> (Tensor, Tensor)");
    m.def(
        "initialize_helix_workspace(Tensor(a!) workspace, int cp_rank, int cp_size) "
        "-> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("alltoall_helix", &tensorrt_llm::torch_ext::alltoall_helix);
    m.impl("alltoall_helix_native", &tensorrt_llm::torch_ext::alltoall_helix_native);
    m.impl("initialize_helix_workspace", &tensorrt_llm::torch_ext::initialize_helix_workspace);
}
