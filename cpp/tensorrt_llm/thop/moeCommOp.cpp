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
#include "tensorrt_llm/kernels/fusedMoeCommKernels.h"
#include "tensorrt_llm/kernels/moePrepareKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <vector>

namespace torch_ext
{

void setMoeCommFieldInfo(tensorrt_llm::kernels::MoeCommFieldInfo& fieldInfo, torch::Tensor const& tensor)
{
    TORCH_CHECK(tensor.dim() == 2, "tensor must be a 2D tensor");
    int eltSize = tensor.dtype().itemsize();
    fieldInfo.fillFieldInfo(static_cast<uint8_t*>(tensor.data_ptr()), eltSize, tensor.size(1), tensor.stride(0),
        convert_torch_dtype(tensor.scalar_type()));
}

c10::List<torch::Tensor> moeCommOp(c10::List<torch::Tensor> inputs, torch::Tensor sendRankCumSum,
    torch::Tensor sendIndiceTensor, torch::Tensor recvRankCumSum, torch::Tensor recvIndiceTensor,
    torch::Tensor allWorkspaces, int64_t outputAllocationCount, int64_t epRank, int64_t epSize,
    std::optional<c10::List<bool>> needZeroOutput = std::nullopt, c10::optional<bool> useLowPrecision = std::nullopt)
{
    CHECK_INPUT(sendRankCumSum, torch::kInt32);
    CHECK_INPUT(sendIndiceTensor, torch::kInt32);
    CHECK_INPUT(recvRankCumSum, torch::kInt32);
    CHECK_INPUT(recvIndiceTensor, torch::kInt32);

    TORCH_CHECK(sendRankCumSum.dim() == 1, "sendRankCumSum must be a 1D tensor");
    TORCH_CHECK(sendIndiceTensor.dim() == 1, "sendIndices must be a 1D tensor");
    TORCH_CHECK(recvRankCumSum.dim() == 1, "recvRankCumSum must be a 1D tensor");
    TORCH_CHECK(recvIndiceTensor.dim() == 1, "recvIndices must be a 1D tensor");
    TORCH_CHECK(allWorkspaces.dim() == 2, "allWorkspaces must be a 2D tensor");

    TORCH_CHECK(sendRankCumSum.size(0) == epSize, "sendRankCumSum must have epSize elements");
    TORCH_CHECK(recvRankCumSum.size(0) == epSize, "recvRankCumSum must have epSize elements");
    TORCH_CHECK(allWorkspaces.size(0) == epSize, "allWorkspaces must have epSize elements");

    TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");
    TORCH_CHECK(!needZeroOutput.has_value() || needZeroOutput.value().size() == inputs.size(),
        "needZeroOutput should have same length as inputs");
    c10::List<torch::Tensor> outputs;

    tensorrt_llm::kernels::MoeEpWorldInfo epWorldInfo = {static_cast<int>(epSize), static_cast<int>(epRank)};
    tensorrt_llm::kernels::FusedMoeWorldInfo worldInfo = {epWorldInfo};

    tensorrt_llm::kernels::SendRecvIndices sendIndices, recvIndices;
    sendIndices.rankCountCumSum = sendRankCumSum.data_ptr<int>();
    sendIndices.rankLocalIndices = sendIndiceTensor.data_ptr<int>();

    recvIndices.rankCountCumSum = recvRankCumSum.data_ptr<int>();
    recvIndices.rankLocalIndices = recvIndiceTensor.data_ptr<int>();

    int fieldCount = inputs.size();
    TORCH_CHECK(fieldCount <= tensorrt_llm::kernels::MOE_COMM_FIELD_MAX_COUNT, "Number of fields (", fieldCount,
        ") exceeds maximum allowed (", tensorrt_llm::kernels::MOE_COMM_FIELD_MAX_COUNT, ")");
    tensorrt_llm::kernels::FusedMoeFieldInfo sendFieldInfo, recvFieldInfo;
    sendFieldInfo.isBasicInterleaved = false;
    recvFieldInfo.isBasicInterleaved = false;
    sendFieldInfo.fieldCount = fieldCount;
    recvFieldInfo.fieldCount = fieldCount;
    sendFieldInfo.expertScales = nullptr;
    recvFieldInfo.expertScales = nullptr;
    sendFieldInfo.tokenSelectedSlots = nullptr;
    recvFieldInfo.tokenSelectedSlots = nullptr;

    for (int i = 0; i < fieldCount; i++)
    {
        torch::Tensor const& t = inputs[i];
        setMoeCommFieldInfo(sendFieldInfo.fieldsInfo[i], t);
        if (needZeroOutput.has_value() && needZeroOutput.value()[i])
        {
            outputs.push_back(torch::zeros({outputAllocationCount, t.size(1)}, t.options()));
        }
        else
        {
            outputs.push_back(torch::empty({outputAllocationCount, t.size(1)}, t.options()));
        }
        setMoeCommFieldInfo(recvFieldInfo.fieldsInfo[i], outputs[i]);
    }
    sendFieldInfo.fillFieldPlacementInfo(0, false);
    recvFieldInfo.fillFieldPlacementInfo(0, false);

    tensorrt_llm::kernels::FusedMoeCommKernelParam params;
    params.worldInfo = worldInfo;
    params.sendIndices = sendIndices;
    params.recvIndices = recvIndices;
    params.sendFieldInfo = sendFieldInfo;
    params.recvFieldInfo = recvFieldInfo;
    // Do not need expertParallelInfo for fused moe comm now

    bool useLowPrecisionVal = useLowPrecision.value_or(false);
    params.isLowPrecision = useLowPrecisionVal;
    params.sendFieldInfo.fillMetaInfo(
        &(params.sendCommMeta), params.expertParallelInfo.topK, false, false, useLowPrecisionVal);
    params.recvFieldInfo.fillMetaInfo(
        &(params.recvCommMeta), params.expertParallelInfo.topK, false, false, useLowPrecisionVal);

    tensorrt_llm::kernels::FusedMoeWorkspace fusedMoeWorkspace;
    tensorrt_llm::kernels::constructWorkspace(
        &fusedMoeWorkspace, allWorkspaces.data_ptr<uint64_t>(), allWorkspaces.stride(0), epSize);

    auto stream = at::cuda::getCurrentCUDAStream();

    tensorrt_llm::kernels::moeAllToAll(params, fusedMoeWorkspace, stream);

    return outputs;
}

int64_t getWorkspaceSizePerRank(int64_t epSize)
{
    int epSize32 = static_cast<int>(epSize);
    return tensorrt_llm::kernels::getFusedMoeCommWorkspaceSize(epSize32);
}

void setMaxUsableSmCount(int64_t maxSmCount)
{
    tensorrt_llm::kernels::setMaxUsableSmCount(maxSmCount);
}

int64_t getPrepareWorkspaceSizePerRank(int64_t epSize)
{
    int epSize32 = static_cast<int>(epSize);
    return tensorrt_llm::kernels::moe_prepare::getMoePrepareWorkspaceSize(epSize32);
}

void initializeMoeWorkspace(torch::Tensor allWorkspaces, int64_t epRank, int64_t epSize)
{
    TORCH_CHECK(allWorkspaces.dim() == 2, "allWorkspaces must be a 2D tensor");
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");

    tensorrt_llm::kernels::MoeEpWorldInfo epWorldInfo = {static_cast<int>(epSize), static_cast<int>(epRank)};
    tensorrt_llm::kernels::FusedMoeWorldInfo worldInfo = {epWorldInfo};

    tensorrt_llm::kernels::FusedMoeWorkspace fusedMoeWorkspace;
    tensorrt_llm::kernels::constructWorkspace(
        &fusedMoeWorkspace, allWorkspaces.data_ptr<uint64_t>(), allWorkspaces.stride(0), epSize);

    tensorrt_llm::kernels::initializeFusedMoeLocalWorkspace(&fusedMoeWorkspace, worldInfo);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>>
moePrepareOp(torch::Tensor expertsIds, c10::optional<torch::Tensor> expertsStatics, torch::Tensor allWorkspaces,
    int64_t maxTokenCountPerRank, int64_t epRank, int64_t epSize, int64_t expertCount, int64_t slotCount, int64_t topK)
{
    CHECK_INPUT(expertsIds, torch::kInt32);
    TORCH_CHECK(expertCount % 4 == 0, "expertCount must be divisible by 4");
    TORCH_CHECK(slotCount % 4 == 0, "slotCount must be divisible by 4");
    TORCH_CHECK(expertCount + 1 <= 512, "expertCount + 1 is larger than 512");

    int64_t maxSendRanksPerToken = std::max(epSize, topK);
    int64_t tokenCount = expertsIds.size(0);

    torch::Tensor preparedLocalExpertIds
        = torch::empty({maxTokenCountPerRank * epSize, topK}, expertsIds.options().dtype(torch::kInt32));

    torch::Tensor sendRankCountCumSum = torch::empty({epSize}, expertsIds.options().dtype(torch::kInt32));
    torch::Tensor RecvRankCountCumSum = torch::empty({epSize}, expertsIds.options().dtype(torch::kInt32));

    torch::Tensor gatherRecvRankIndices
        = torch::empty({maxTokenCountPerRank * epSize}, expertsIds.options().dtype(torch::kInt32));
    torch::Tensor recvRankIndices
        = torch::empty({maxTokenCountPerRank * epSize}, expertsIds.options().dtype(torch::kInt32));

    torch::Tensor gatherBackwardRecvRankIndices
        = torch::empty({maxTokenCountPerRank * maxSendRanksPerToken}, expertsIds.options().dtype(torch::kInt32));
    torch::Tensor backwardRecvRankIndices
        = torch::empty({maxTokenCountPerRank * maxSendRanksPerToken}, expertsIds.options().dtype(torch::kInt32));

    torch::Tensor gatherSendRankIndices
        = torch::empty({maxTokenCountPerRank * maxSendRanksPerToken}, expertsIds.options().dtype(torch::kInt32));
    torch::Tensor sendRankIndices
        = torch::empty({maxTokenCountPerRank * maxSendRanksPerToken}, expertsIds.options().dtype(torch::kInt32));

    int* localExpertStaticsPtr = nullptr;
    int* gatheredExpertStaticsPtr = nullptr;
    c10::optional<torch::Tensor> gatheredExpertStatics;
    if (expertsStatics.has_value())
    {
        localExpertStaticsPtr = expertsStatics.value().data_ptr<int>();
        gatheredExpertStatics = torch::empty({epSize, expertCount}, expertsIds.options().dtype(torch::kInt32));
        gatheredExpertStaticsPtr = gatheredExpertStatics.value().data_ptr<int>();
    }

    tensorrt_llm::kernels::moe_prepare::MoeCommWorkspace workspace;
    workspace.workspacePtr = allWorkspaces.data_ptr<uint64_t>();
    workspace.rankStrideInU64 = allWorkspaces.stride(0);

    auto stream = at::cuda::getCurrentCUDAStream();

    tensorrt_llm::kernels::moe_prepare::computeCountAndIndice(expertsIds.data_ptr<int>(),
        sendRankCountCumSum.data_ptr<int>(), RecvRankCountCumSum.data_ptr<int>(), sendRankIndices.data_ptr<int>(),
        backwardRecvRankIndices.data_ptr<int>(), recvRankIndices.data_ptr<int>(), localExpertStaticsPtr,
        gatheredExpertStaticsPtr, workspace, tokenCount, maxTokenCountPerRank, topK, slotCount, expertCount, epRank,
        epSize, stream);

    tensorrt_llm::kernels::moe_prepare::computeCumsum(
        sendRankCountCumSum.data_ptr<int>(), RecvRankCountCumSum.data_ptr<int>(), epRank, epSize, stream);

    tensorrt_llm::kernels::moe_prepare::moveIndice(sendRankCountCumSum.data_ptr<int>(),
        RecvRankCountCumSum.data_ptr<int>(), sendRankIndices.data_ptr<int>(), gatherSendRankIndices.data_ptr<int>(),
        backwardRecvRankIndices.data_ptr<int>(), gatherBackwardRecvRankIndices.data_ptr<int>(),
        recvRankIndices.data_ptr<int>(), gatherRecvRankIndices.data_ptr<int>(), epRank, epSize, maxTokenCountPerRank,
        stream);

    return std::make_tuple(sendRankCountCumSum, gatherSendRankIndices, RecvRankCountCumSum, gatherRecvRankIndices,
        gatherBackwardRecvRankIndices, gatheredExpertStatics);
}

void memsetExpertIds(torch::Tensor expertsIds, torch::Tensor recvRankCountCumSum, int64_t maxTokenCountPerRank,
    int64_t topK, int64_t slotCount, int64_t epSize)
{
    CHECK_INPUT(expertsIds, torch::kInt32);
    TORCH_CHECK(expertsIds.dim() == 2, "expertsIds must be a 1D tensor");
    TORCH_CHECK(
        expertsIds.size(0) == maxTokenCountPerRank * epSize, "expertsIds must have maxTokenCountPerRank * epSize rows");
    TORCH_CHECK(expertsIds.size(1) == topK, "expertsIds must have topK columns");

    CHECK_INPUT(recvRankCountCumSum, torch::kInt32);
    TORCH_CHECK(recvRankCountCumSum.dim() == 1, "recvRankCountCumSum must be a 1D tensor");
    TORCH_CHECK(recvRankCountCumSum.size(0) == epSize, "recvRankCountCumSum must have epSize elements");

    auto stream = at::cuda::getCurrentCUDAStream();

    tensorrt_llm::kernels::moe_prepare::memsetExpertIds(expertsIds.data_ptr<int>(), recvRankCountCumSum.data_ptr<int>(),
        static_cast<int>(maxTokenCountPerRank), static_cast<int>(topK), static_cast<int>(slotCount),
        static_cast<int>(epSize), stream);
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "moe_comm(Tensor[] inputs, Tensor send_rank_cum_sum, Tensor send_indices, Tensor "
        "recv_rank_cum_sum, Tensor recv_indices, Tensor all_workspaces, int output_allocation_count, int ep_rank, int "
        "ep_size, bool[]? need_zero_output=None, bool? use_low_precision=None) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("moe_comm", &torch_ext::moeCommOp);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("moe_initialize_workspace(Tensor(a!) all_workspaces, int ep_rank, int ep_size) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("moe_initialize_workspace", &torch_ext::initializeMoeWorkspace);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("get_moe_commworkspace_size_per_rank(int ep_size) -> int");
}

TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("get_moe_commworkspace_size_per_rank", &torch_ext::getWorkspaceSizePerRank);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("set_moe_max_usable_sm_count(int max_sm_count) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("set_moe_max_usable_sm_count", &torch_ext::setMaxUsableSmCount);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mnnvl_moe_alltoallv_prepare_without_allgather(Tensor experts_ids, Tensor? experts_statics, "
        "Tensor allWorkspace, int max_token_count_per_rank, int ep_rank, int ep_size, int expert_count, int "
        "slot_count, int top_k) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor?)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mnnvl_moe_alltoallv_prepare_without_allgather", &torch_ext::moePrepareOp);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "memset_expert_ids(Tensor(a!) experts_ids, Tensor recv_rank_count_cumsum, int max_token_count_per_rank, int "
        "top_k, "
        "int slot_count, int ep_size) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("memset_expert_ids", &torch_ext::memsetExpertIds);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("get_moe_prepare_workspace_size_per_rank(int ep_size) -> int");
}

TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("get_moe_prepare_workspace_size_per_rank", &torch_ext::getPrepareWorkspaceSizePerRank);
}
