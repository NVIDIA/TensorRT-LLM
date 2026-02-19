/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/thop/moeAlltoAllMeta.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace moe_comm
{

static constexpr size_t CACHELINE_ALIGNMENT = 128;

// TODO: Is Alignment necessary?
// Helper function to align offset to specified byte boundary
inline size_t alignOffset(size_t offset, size_t alignment)
{
    return (offset + alignment - 1) & ~(alignment - 1);
}

// Calculate auxiliary data offsets
MoeA2ADataOffsets calculateOffsets(int epSize, int maxNumTokens, int eplbStatsNumExperts)
{
    // TODO: Use lambdas to encapsulate offset and alignment for each entry, which is less error prone and easier to
    // read.
    constexpr size_t SIZEOF_INT32 = 4;

    MoeA2ADataOffsets offsets;
    size_t offset = 0;

    // flag_val
    offsets[FLAG_VAL_OFFSET_INDEX] = offset;
    offset += SIZEOF_INT32;

    // local_token_counter
    offsets[LOCAL_TOKEN_COUNTER_OFFSET_INDEX] = offset;
    offset += SIZEOF_INT32;

    // send_counters
    offsets[SEND_COUNTERS_OFFSET_INDEX] = offset;
    offset += epSize * SIZEOF_INT32;

    // recv_counters
    offsets[RECV_COUNTERS_OFFSET_INDEX] = offset;
    offset += epSize * SIZEOF_INT32;

    // dispatch completion flags
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets[DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX] = offset;
    offset += epSize * SIZEOF_INT32;

    // combine completion flags
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets[COMBINE_COMPLETION_FLAGS_OFFSET_INDEX] = offset;
    offset += epSize * SIZEOF_INT32;

    // topk_target_ranks: [maxNumTokens, kMaxTopK]
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets[TOPK_TARGET_RANKS_OFFSET_INDEX] = offset;
    offset += static_cast<size_t>(maxNumTokens) * static_cast<size_t>(tensorrt_llm::kernels::moe_comm::kMaxTopK)
        * SIZEOF_INT32;

    // topk_send_indices: [maxNumTokens, kMaxTopK]
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets[TOPK_SEND_INDICES_OFFSET_INDEX] = offset;
    offset += static_cast<size_t>(maxNumTokens) * static_cast<size_t>(tensorrt_llm::kernels::moe_comm::kMaxTopK)
        * SIZEOF_INT32;

    // eplb gathered stats: [epSize, eplbStatsNumExperts]
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets[EPLB_GATHERED_STATS_OFFSET_INDEX] = offset;
    offset += static_cast<size_t>(epSize) * static_cast<size_t>(eplbStatsNumExperts) * SIZEOF_INT32;

    // payload data
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets[PAYLOAD_DATA_OFFSET_INDEX] = offset;

    return offsets;
}

// Initialize auxiliary data in workspace
// This function sets up the initial values for flag_val and completion_flags
//
// Inputs:
//   - workspace: [ep_size, size_per_rank] unified virtual memory workspace
//   - epRank: Current expert parallel rank
//   - epSize: Total expert parallel size
//   - maxNumTokens: Maximum number of tokens supported
//   - eplbStatsNumExperts: (Optional) Number of experts used for EPLB stats
//
// Returns:
//   - metainfo: Tensor containing offsets for auxiliary data
torch::Tensor moeA2AInitializeOp(torch::Tensor const& workspace, int64_t epRank, int64_t epSize, int64_t maxNumTokens,
    torch::optional<int64_t> eplbStatsNumExperts)
{
    // Validate inputs
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    TORCH_CHECK(workspace.dim() == 2, "workspace must be a 2D tensor of shape [epSize, sizePerRank]");
    TORCH_CHECK(workspace.size(0) == epSize, "workspace first dimension must equal epSize");
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");

    // Initialize workspace to zero
    workspace[epRank].zero_();

    int64_t eplbStatsNumExpertsValue = eplbStatsNumExperts.value_or(0);
    TORCH_CHECK(eplbStatsNumExpertsValue >= 0, "eplbStatsNumExperts must be positive if not None.");

    // Calculate auxiliary data offsets
    MoeA2ADataOffsets offsets = calculateOffsets(epSize, maxNumTokens, static_cast<int>(eplbStatsNumExpertsValue));

    // Return metainfo as a tensor containing offsets
    torch::Tensor metainfo = torch::empty(
        {static_cast<int64_t>(NUM_METAINFO_FIELDS)}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    for (int i = 0; i < static_cast<int>(NUM_METAINFO_FIELDS); i++)
    {
        metainfo[i] = static_cast<int64_t>(offsets[i]);
    }

    // Synchronize among ranks
    cudaDeviceSynchronize();
    tensorrt_llm::mpi::MpiComm::world().barrier();

    return metainfo;
}

// MoE All-to-All Dispatch Operation
// This operation dispatches tokens and their associated payloads to different expert ranks.
//
// Inputs:
//   - tokenSelectedExperts: [local_num_tokens, top_k] tensor of expert indices
//   - inputPayloads: List of tensors with shape [local_num_tokens, ...] containing data to dispatch
//   - workspace: [ep_size, size_per_rank] unified virtual memory workspace where size_per_rank is large enough to store
//   all the auxiliary data and recv payloads.
//   - metainfo: [NUM_METAINFO_FIELDS] tensor containing offsets for auxiliary data
//   - runtimeMaxTokensPerRank: Maximum of the number of tokens of each DP rank's local batch. This is a dynamic value
//   during runtime.
//   - maxNumTokens: Maximum number of tokens that could be supported. This is a static value that is setup during
//   initialization.
//   - epRank: Current expert parallel rank
//   - epSize: Total expert parallel size
//   - topK: Number of experts selected per token
//   - numExperts: Total number of routing slots (tokenSelectedExperts values are in [0, numExperts))
//   - eplbStatsNumExperts: Number of experts used for EPLB stats (may be <= numExperts)
//   - eplbLocalStats: [eplbStatsNumExperts] tensor containing local statistics for EPLB.
//
// Return values:
//   - recvTensors: Vector of receive buffers (one tensor per payload), each [ep_size, runtimeMaxTokensPerRank,
//   elements_per_token]
//   - combinePayloadOffset: Offset into workspace for the combine payload region, to be used by the combine operation
//   - eplbGatheredStats: (Optional) [ep_size, eplbStatsNumExperts] tensor containing gathered statistics for EPLB, or
//   an empty tensor if eplbLocalStats is None.
//
// Note: token_selected_experts is used for routing but is NOT automatically included as a payload.
//       If you want to dispatch token_selected_experts, include it explicitly in inputPayloads.
std::tuple<std::vector<torch::Tensor>, int64_t, torch::Tensor> moeA2ADispatchOp(
    torch::Tensor const& tokenSelectedExperts, std::vector<torch::Tensor> const& inputPayloads,
    torch::Tensor const& workspace, torch::Tensor const& metainfo, int64_t runtimeMaxTokensPerRank, int64_t epRank,
    int64_t epSize, int64_t topK, int64_t numExperts, torch::optional<torch::Tensor> eplbLocalStats)
{
    using tensorrt_llm::kernels::moe_comm::PayloadDescriptor;
    using tensorrt_llm::kernels::moe_comm::MoeA2ADispatchParams;
    using tensorrt_llm::kernels::moe_comm::moe_a2a_dispatch_launch;
    using tensorrt_llm::kernels::moe_comm::kMaxTopK;
    using tensorrt_llm::kernels::moe_comm::kMaxPayloads;

    // Validate inputs
    CHECK_INPUT(tokenSelectedExperts, torch::kInt32);
    TORCH_CHECK(tokenSelectedExperts.dim() == 2, "tokenSelectedExperts must be a 2D tensor");
    TORCH_CHECK(tokenSelectedExperts.size(1) == topK, "tokenSelectedExperts must have topK columns");

    CHECK_CPU(metainfo);
    CHECK_TYPE(metainfo, torch::kInt64);
    TORCH_CHECK(metainfo.dim() == 1, "metainfo must be a 1D tensor");
    TORCH_CHECK(metainfo.size(0) == static_cast<int64_t>(NUM_METAINFO_FIELDS),
        "metainfo must have NUM_METAINFO_FIELDS elements");
    MoeA2ADataOffsets const& offsets = *reinterpret_cast<MoeA2ADataOffsets const*>(metainfo.data_ptr<int64_t>());

    int64_t localNumTokens = tokenSelectedExperts.size(0);
    TORCH_CHECK(runtimeMaxTokensPerRank > 0, "runtimeMaxTokensPerRank must be positive");
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");
    TORCH_CHECK(topK > 0 && topK <= kMaxTopK, "topK must be in the range (0, kMaxTopK]");
    TORCH_CHECK(!inputPayloads.empty(), "inputPayloads must not be empty");
    TORCH_CHECK(inputPayloads.size() <= kMaxPayloads, "Too many input payloads");
    TORCH_CHECK(numExperts >= epSize, "numExperts must be greater than or equal to epSize");
    TORCH_CHECK(numExperts % epSize == 0, "numExperts must be divisible by epSize for contiguous partitioning");
    bool enableEplb = eplbLocalStats.has_value();
    int64_t eplbStatsNumExperts = 0;
    if (enableEplb)
    {
        TORCH_CHECK(eplbLocalStats.has_value(), "enable_eplb requires eplb_local_stats");
        torch::Tensor const& eplbLocalStatsTensor = eplbLocalStats.value();
        eplbStatsNumExperts = eplbLocalStatsTensor.size(0);
        TORCH_CHECK(eplbStatsNumExperts > 0, "eplb_local_stats must not be empty");
        TORCH_CHECK(eplbStatsNumExperts <= numExperts, "eplb_local_stats size must be <= numExperts (slots)");
        CHECK_INPUT(eplbLocalStatsTensor, torch::kInt32);
        TORCH_CHECK(eplbLocalStatsTensor.is_contiguous(), "eplb_local_stats must be contiguous");
        TORCH_CHECK(eplbLocalStatsTensor.dim() == 1, "eplb_local_stats must be a 1D tensor");
    }

    // All input payloads must have the same first dimension (localNumTokens)
    for (auto const& payload : inputPayloads)
    {
        TORCH_CHECK(payload.dim() >= 1, "All payloads must have at least 1 dimension");
        TORCH_CHECK(payload.size(0) == localNumTokens,
            "All payloads must have the same first dimension as tokenSelectedExperts");
        TORCH_CHECK(payload.is_contiguous(), "All payloads must be contiguous");
    }

    // Record the cacheline aligned start offset for each payload's recv buffer.
    // 1. We assume the base workspace ptr of each rank is aligned (checked in this OP)
    // 2. offsets[PAYLOAD_DATA_OFFSET_INDEX] is aligned (ensured in calculateOffsets)
    // 3. We align the currentOffset during update.
    // In this way, it is guaranteed that the recv buffer is (over-)aligned, sufficient for 128bit vectorized ld/st.

    std::vector<int> payloadElementSizes;
    std::vector<int> payloadElementsPerToken;
    std::vector<size_t> payloadRecvBufferOffsets;

    // Start offset for the first payload
    size_t currentOffset = static_cast<size_t>(offsets[PAYLOAD_DATA_OFFSET_INDEX]);
    for (auto const& payload : inputPayloads)
    {
        CHECK_CONTIGUOUS(payload);
        CHECK_TH_CUDA(payload);
        TORCH_CHECK(payload.dim() == 2, "payload must be a 2D tensor");
        TORCH_CHECK(
            payload.size(0) == localNumTokens, "payload must have the same first dimension as tokenSelectedExperts");
        // Unlike recv buffer for payloads, payload itself is not allocated by us and we cannot control its alignment.
        // We only make sure the payload start offset is 16-byte aligned, while the actual vectorized ld/st width is
        // dynamically determined based on bytes per token of this payload.
        TORCH_CHECK(reinterpret_cast<uintptr_t>(payload.data_ptr()) % 16 == 0, "payload must be 16-byte aligned");

        int elementsPerToken = static_cast<int>(payload.size(1));
        int elementSize = static_cast<int>(payload.dtype().itemsize());
        // Each payload buffer stores data from ALL ranks
        int64_t bytesPerPayload = epSize * runtimeMaxTokensPerRank * elementsPerToken * elementSize;

        payloadElementSizes.push_back(elementSize);
        payloadElementsPerToken.push_back(elementsPerToken);

        payloadRecvBufferOffsets.push_back(currentOffset);

        // Update offset and align to cacheline boundary for the next payload recv buffer.
        currentOffset += bytesPerPayload;
        currentOffset = alignOffset(currentOffset, CACHELINE_ALIGNMENT);
    }

    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    // Don't check contiguous - MnnvlMemory creates strided tensors for multi-GPU
    TORCH_CHECK(workspace.dim() == 2, "workspace must be a 2D tensor of shape [epSize, sizePerRank]");
    TORCH_CHECK(workspace.size(0) == epSize, "workspace first dimension must equal epSize");

    // Validate workspace size - must include space for auxiliary data + payloads
    int64_t sizePerRank = workspace.size(1);
    int64_t requiredSize = static_cast<int64_t>(currentOffset);
    TORCH_CHECK(sizePerRank >= requiredSize,
        "Workspace size per rank insufficient for dispatch. "
        "Need at least ",
        requiredSize, " bytes (", offsets[PAYLOAD_DATA_OFFSET_INDEX], " for auxiliary data + payloads), but got ",
        sizePerRank);

    // Get base workspace pointer
    uint8_t* workspacePtr = workspace.data_ptr<uint8_t>();
    uint8_t* rankWorkSpacePtr = workspacePtr + epRank * workspace.stride(0);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(rankWorkSpacePtr) % CACHELINE_ALIGNMENT == 0,
        "rankWorkSpacePtr must be %d-byte aligned", CACHELINE_ALIGNMENT);

    // Setup payload descriptors for source data
    int num_payloads = static_cast<int>(inputPayloads.size());
    std::vector<PayloadDescriptor> payloadDescriptors(num_payloads);
    for (int i = 0; i < num_payloads; i++)
    {
        payloadDescriptors[i].src_data = inputPayloads[i].data_ptr();
        payloadDescriptors[i].element_size = payloadElementSizes[i];
        payloadDescriptors[i].elements_per_token = payloadElementsPerToken[i];
    }

    // Setup dispatch parameters
    MoeA2ADispatchParams params{};
    params.one_block_per_token
        = tensorrt_llm::common::getEnvMoeA2AOneBlockPerToken(); // TODO: Decide this based on the workload
    params.ep_size = static_cast<int>(epSize);
    params.ep_rank = static_cast<int>(epRank);
    params.num_experts = static_cast<int>(numExperts);
    params.local_num_tokens = static_cast<int>(localNumTokens);
    params.max_tokens_per_rank = static_cast<int>(runtimeMaxTokensPerRank);
    params.top_k = static_cast<int>(topK);
    params.enable_eplb = enableEplb;
    params.eplb_stats_num_experts = static_cast<int>(eplbStatsNumExperts);

    params.token_selected_experts = tokenSelectedExperts.data_ptr<int32_t>();

    params.num_payloads = num_payloads;
    std::copy(payloadDescriptors.begin(), payloadDescriptors.end(), &params.payloads[0]);

    params.flag_val = reinterpret_cast<uint32_t*>(rankWorkSpacePtr + offsets[FLAG_VAL_OFFSET_INDEX]);
    params.local_token_counter = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[LOCAL_TOKEN_COUNTER_OFFSET_INDEX]);
    params.send_counters = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[SEND_COUNTERS_OFFSET_INDEX]);
    params.topk_target_ranks = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[TOPK_TARGET_RANKS_OFFSET_INDEX]);
    params.topk_send_indices = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[TOPK_SEND_INDICES_OFFSET_INDEX]);

    for (int target_rank = 0; target_rank < epSize; target_rank++)
    {
        uint8_t* targetWorkSpacePtr = workspacePtr + (target_rank * workspace.stride(0));

        params.recv_counters[target_rank]
            = reinterpret_cast<int*>(targetWorkSpacePtr + offsets[RECV_COUNTERS_OFFSET_INDEX]);
        params.completion_flags[target_rank]
            = reinterpret_cast<uint32_t*>(targetWorkSpacePtr + offsets[DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX]);
        if (enableEplb)
        {
            params.eplb_gathered_stats[target_rank]
                = reinterpret_cast<int*>(targetWorkSpacePtr + offsets[EPLB_GATHERED_STATS_OFFSET_INDEX]);
        }
        else
        {
            params.eplb_gathered_stats[target_rank] = nullptr;
        }

        for (int payload_idx = 0; payload_idx < num_payloads; payload_idx++)
        {
            // Store pointer for current payload using pre-calculated aligned offset
            params.recv_buffers[target_rank][payload_idx] = targetWorkSpacePtr + payloadRecvBufferOffsets[payload_idx];
        }
    }

    if (enableEplb)
    {
        params.eplb_local_stats = eplbLocalStats.value().data_ptr<int32_t>();
    }
    else
    {
        params.eplb_local_stats = nullptr;
    }

    params.stream = at::cuda::getCurrentCUDAStream();

    // Prepare for dispatch (zero counters/indices and increment flag_val)
    moe_a2a_prepare_dispatch_launch(params);

    // Launch the dispatch kernel
    moe_a2a_dispatch_launch(params);
    cudaError_t result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "moe_a2a_dispatch kernel launch failed: ", cudaGetErrorString(result));

    // Create tensor views for the current rank's receive buffers only
    std::vector<torch::Tensor> recvTensors;
    for (int payload_idx = 0; payload_idx < num_payloads; payload_idx++)
    {
        auto const& payload = inputPayloads[payload_idx];
        // Create tensor view for this payload using pre-calculated aligned offset
        auto recvTensor = torch::from_blob(rankWorkSpacePtr + payloadRecvBufferOffsets[payload_idx],
            {epSize, runtimeMaxTokensPerRank, payloadElementsPerToken[payload_idx]}, payload.options());
        recvTensors.push_back(recvTensor);
    }

    // Compute aligned offset after dispatch payloads for combine payload region
    int64_t combinePayloadOffset = static_cast<int64_t>(alignOffset(currentOffset, CACHELINE_ALIGNMENT));

    torch::Tensor eplbGatheredStats;
    if (enableEplb)
    {
        int* gatheredStatsPtr = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[EPLB_GATHERED_STATS_OFFSET_INDEX]);
        auto statsOptions = workspace.options().dtype(torch::kInt32);
        eplbGatheredStats = torch::from_blob(
            gatheredStatsPtr, {static_cast<int64_t>(epSize), static_cast<int64_t>(eplbStatsNumExperts)}, statsOptions);
    }
    else
    {
        eplbGatheredStats = torch::empty({0}, workspace.options().dtype(torch::kInt32));
    }

    return std::make_tuple(std::move(recvTensors), combinePayloadOffset, std::move(eplbGatheredStats));
}

// MoE All-to-All Combine Operation
// Combine the per-rank expert outputs into the originating tokens' buffers on the local rank.
//
// Two payload modes are supported:
//   1) External payload tensor: 'payload' is a tensor with shape [ep_size, max_tokens_per_rank, elements_per_token]
//      that is NOT backed by the shared workspace. In this mode, the op stages the current rank's
//      slice into the workspace region at 'payloadRegionOffset' via the prepare kernel.
//   2) Workspace-backed payload tensor: 'payload' is a view into the shared workspace. Set
//      payloadInWorkspace=true to skip staging. The op will read directly from the workspace region
//      at 'combinePayloadOffset'.
// In both cases, the combine kernel reads from the workspace at 'combinePayloadOffset'.
torch::Tensor moeA2ACombineOp(torch::Tensor const& payload, int64_t localNumTokens, torch::Tensor const& workspace,
    torch::Tensor const& metainfo, int64_t runtimeMaxTokensPerRank, int64_t epRank, int64_t epSize, int64_t topK,
    int64_t combinePayloadOffset, bool payloadInWorkspace)
{
    using tensorrt_llm::kernels::moe_comm::MoeA2ACombineParams;
    using tensorrt_llm::kernels::moe_comm::moe_a2a_combine_launch;
    using tensorrt_llm::kernels::moe_comm::kMaxTopK;

    // Validate inputs
    CHECK_TH_CUDA(payload);
    CHECK_CONTIGUOUS(payload);
    TORCH_CHECK(payload.dim() == 3, "payload must be a 3D tensor [ep_size, max_tokens_per_rank, elements_per_token]");
    TORCH_CHECK(payload.size(0) == epSize, "payload first dimension must equal epSize");
    TORCH_CHECK(
        payload.size(1) == runtimeMaxTokensPerRank, "payload second dimension must equal runtimeMaxTokensPerRank");
    // We only make sure the payload start offset is 16-byte aligned, while the actual vectorized ld/st width is
    // dynamically determined based on bytes per token of this payload.
    TORCH_CHECK(reinterpret_cast<uintptr_t>(payload.data_ptr()) % 16 == 0, "payload must be 16-byte aligned");
    int64_t elementsPerToken = payload.size(2);
    TORCH_CHECK(elementsPerToken > 0, "elementsPerToken must be positive");
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");
    TORCH_CHECK(topK > 0 && topK <= kMaxTopK, "topK must be in the range (0, kMaxTopK]");

    // Map torch dtype to nvinfer1::DataType
    nvinfer1::DataType nvDtype = nvinfer1::DataType::kFLOAT;
    auto scalarType = payload.scalar_type();
    if (scalarType == at::kHalf)
    {
        nvDtype = nvinfer1::DataType::kHALF;
    }
    else if (scalarType == at::kBFloat16)
    {
        nvDtype = nvinfer1::DataType::kBF16;
    }
    else if (scalarType == at::kFloat)
    {
        nvDtype = nvinfer1::DataType::kFLOAT;
    }
    else
    {
        TORCH_CHECK(false, "Unsupported data type for payload");
    }

    CHECK_CPU(metainfo);
    CHECK_TYPE(metainfo, torch::kInt64);
    TORCH_CHECK(metainfo.dim() == 1, "metainfo must be a 1D tensor");
    TORCH_CHECK(metainfo.size(0) == static_cast<int64_t>(NUM_METAINFO_FIELDS),
        "metainfo must have NUM_METAINFO_FIELDS elements");
    MoeA2ADataOffsets const& offsets = *reinterpret_cast<MoeA2ADataOffsets const*>(metainfo.data_ptr<int64_t>());

    // Validate workspace and set synchronization pointers
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    TORCH_CHECK(workspace.dim() == 2 && workspace.size(0) == epSize, "workspace must be [ep_size, size_per_rank]");
    uint8_t* workspacePtr = workspace.data_ptr<uint8_t>();
    int64_t sizePerRank = workspace.size(1);
    uint8_t* rankWorkSpacePtr = workspacePtr + epRank * workspace.stride(0);

    // If user claims payload is in workspace, ensure payload tensor matches combinePayloadOffset
    if (payloadInWorkspace)
    {
        TORCH_CHECK(payload.data_ptr() == rankWorkSpacePtr + combinePayloadOffset,
            "payload_in_workspace is true but 'payload' dataptr does not match combinePayloadOffset");
    }

    int64_t payloadSize = payload.numel() * payload.element_size();
    TORCH_CHECK(combinePayloadOffset >= 0 && combinePayloadOffset + payloadSize <= sizePerRank,
        "Workspace size per rank insufficient for combine. "
        "Need at least ",
        combinePayloadOffset + payloadSize, " bytes (", combinePayloadOffset, " for offset + ", payloadSize,
        " for payload), but got ", sizePerRank);

    // Create output tensor (local on current rank), no need for initialization
    // Typically, newly allocated GPU torch tensors are at least 16-byte aligned.
    torch::Tensor output = torch::empty({localNumTokens, elementsPerToken}, payload.options());

    // Setup combine parameters
    MoeA2ACombineParams params{};
    params.one_block_per_token
        = tensorrt_llm::common::getEnvMoeA2AOneBlockPerToken(); // TODO: Decide this based on the workload
    params.ep_size = static_cast<int>(epSize);
    params.ep_rank = static_cast<int>(epRank);
    params.local_num_tokens = static_cast<int>(localNumTokens);
    params.max_tokens_per_rank = static_cast<int>(runtimeMaxTokensPerRank);
    params.top_k = static_cast<int>(topK);
    // If payload is not in workspace, stage it into current rank's region at prepare phase
    if (!payloadInWorkspace)
    {
        params.prepare_payload = payload.data_ptr();
    }
    params.output_data = output.data_ptr();
    params.elements_per_token = static_cast<int>(elementsPerToken);
    params.dtype = nvDtype;

    params.flag_val = reinterpret_cast<uint32_t*>(rankWorkSpacePtr + offsets[FLAG_VAL_OFFSET_INDEX]);
    params.topk_target_ranks = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[TOPK_TARGET_RANKS_OFFSET_INDEX]);
    params.topk_send_indices = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[TOPK_SEND_INDICES_OFFSET_INDEX]);
    params.recv_counters = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[RECV_COUNTERS_OFFSET_INDEX]);

    for (int target_rank = 0; target_rank < epSize; target_rank++)
    {
        uint8_t* target_workspace_ptr = workspacePtr + target_rank * workspace.stride(0);
        params.completion_flags[target_rank]
            = reinterpret_cast<uint32_t*>(target_workspace_ptr + offsets[COMBINE_COMPLETION_FLAGS_OFFSET_INDEX]);
        params.recv_buffers[target_rank] = target_workspace_ptr + combinePayloadOffset;
    }

    params.stream = at::cuda::getCurrentCUDAStream();

    moe_a2a_prepare_combine_launch(params);

    // Launch the combine kernel
    moe_a2a_combine_launch(params);
    cudaError_t result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "moe_a2a_combine kernel launch failed: ", cudaGetErrorString(result));

    return output;
}

// Op: moe_a2a_sanitize_expert_ids
void moeA2ASanitizeExpertIdsOp(torch::Tensor& expert_ids, torch::Tensor& workspace, torch::Tensor const& metainfo,
    int64_t epRank, int64_t invalid_expert_id)
{
    CHECK_INPUT(expert_ids, torch::kInt32);
    TORCH_CHECK(expert_ids.dim() == 3, "expert_ids must be [ep_size, runtime_max_tokens_per_rank, top_k]");

    int ep_size = static_cast<int>(expert_ids.size(0));
    int runtime_max_tokens_per_rank = static_cast<int>(expert_ids.size(1));
    int top_k = static_cast<int>(expert_ids.size(2));

    CHECK_CPU(metainfo);
    CHECK_TYPE(metainfo, torch::kInt64);
    TORCH_CHECK(metainfo.dim() == 1, "metainfo must be a 1D tensor");
    TORCH_CHECK(metainfo.size(0) == static_cast<int64_t>(NUM_METAINFO_FIELDS),
        "metainfo must have NUM_METAINFO_FIELDS elements");
    MoeA2ADataOffsets const& offsets = *reinterpret_cast<MoeA2ADataOffsets const*>(metainfo.data_ptr<int64_t>());

    uint8_t* rankWorkSpacePtr = workspace.data_ptr<uint8_t>() + epRank * workspace.stride(0);
    int* recv_counters = reinterpret_cast<int*>(rankWorkSpacePtr + offsets[RECV_COUNTERS_OFFSET_INDEX]);

    tensorrt_llm::kernels::moe_comm::moe_a2a_sanitize_expert_ids_launch(expert_ids.data_ptr<int32_t>(), recv_counters,
        static_cast<int32_t>(invalid_expert_id), ep_size, runtime_max_tokens_per_rank, top_k,
        at::cuda::getCurrentCUDAStream());
}

// Return a workspace-backed tensor for combine payload region using from_blob
torch::Tensor moeA2AGetCombinePayloadTensorOp(torch::Tensor const& workspace, int64_t epRank, int64_t epSize,
    int64_t runtimeMaxTokensPerRank, int64_t combinePayloadOffset, c10::ScalarType outDtype, int64_t hiddenSize)
{
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    TORCH_CHECK(workspace.dim() == 2, "workspace must be [ep_size, size_per_rank_bytes]");
    TORCH_CHECK(epRank >= 0 && epRank < workspace.size(0), "epRank out of range");
    TORCH_CHECK(epSize == workspace.size(0), "epSize mismatch with workspace");
    TORCH_CHECK(runtimeMaxTokensPerRank > 0, "runtimeMaxTokensPerRank must be positive");
    TORCH_CHECK(hiddenSize > 0, "hidden must be positive");

    int64_t sizePerRank = workspace.size(1); // bytes
    int64_t elementSize = static_cast<int64_t>(c10::elementSize(outDtype));
    int64_t bytesNeeded = epSize * runtimeMaxTokensPerRank * hiddenSize * elementSize;
    TORCH_CHECK(combinePayloadOffset >= 0, "combine_payload_offset must be non-negative");
    TORCH_CHECK(combinePayloadOffset + bytesNeeded <= sizePerRank,
        "workspace does not have enough space for combine payload tensor. combine payload offset=",
        combinePayloadOffset, ", payload size needed=", bytesNeeded, ", workspace size per rank=", sizePerRank);

    uint8_t* base = workspace.data_ptr<uint8_t>();
    uint8_t* rankBase = base + epRank * workspace.stride(0);
    uint8_t* dataPtr = rankBase + combinePayloadOffset;

    auto options = workspace.options().dtype(outDtype);
    torch::Tensor t = torch::from_blob(dataPtr, {epSize * runtimeMaxTokensPerRank, hiddenSize}, options);
    return t;
}

// Return the size of auxiliary data in workspace
int64_t moeA2AGetAuxDataSizeOp(int64_t epSize, int64_t maxNumTokens, torch::optional<int64_t> eplbStatsNumExperts)
{
    int64_t eplbStatsNumExpertsValue = eplbStatsNumExperts.value_or(0);
    TORCH_CHECK(eplbStatsNumExpertsValue >= 0, "eplbStatsNumExperts must be positive if not None.");
    MoeA2ADataOffsets offsets = calculateOffsets(
        static_cast<int>(epSize), static_cast<int>(maxNumTokens), static_cast<int>(eplbStatsNumExpertsValue));
    return static_cast<int64_t>(offsets[PAYLOAD_DATA_OFFSET_INDEX]);
}

} // namespace moe_comm

} // namespace torch_ext

TRTLLM_NAMESPACE_END

// PyTorch bindings
TORCH_LIBRARY_FRAGMENT(trtllm, module)
{
    // Note that we returns recv_tensors as a list of views into workspace, we need to upcast its alias
    // group to wildcard (a!->*). See
    // https://github.com/pytorch/pytorch/blob/b1eb6dede556136f9fdcee28415b0358d58ad877/aten/src/ATen/native/README.md#annotations
    module.def(
        "moe_a2a_dispatch(Tensor token_selected_experts, Tensor[] input_payloads, "
        "Tensor(a!->*) workspace, Tensor metainfo, int runtime_max_tokens_per_rank, "
        "int ep_rank, int ep_size, int top_k, int num_experts, "
        "Tensor? eplb_local_stats=None) -> (Tensor(a!)[], int, Tensor(a!))");
    module.def(
        "moe_a2a_combine(Tensor(a) payload, int local_num_tokens,"
        "Tensor(a!) workspace, Tensor metainfo, int runtime_max_tokens_per_rank, "
        "int ep_rank, int ep_size, int top_k, int combine_payload_offset, "
        "bool payload_in_workspace) -> Tensor");
    module.def(
        "moe_a2a_initialize(Tensor(a!) workspace, int ep_rank, int ep_size, int max_num_tokens_per_rank, "
        "int? eplb_stats_num_experts=None) -> Tensor");
    module.def(
        "moe_a2a_sanitize_expert_ids(Tensor(a!) expert_ids, Tensor(a!) workspace, Tensor metainfo, int ep_rank, int "
        "invalid_expert_id) -> ()");
    module.def(
        "moe_a2a_get_combine_payload_tensor(Tensor(a) workspace, int ep_rank, int ep_size, int "
        "runtime_max_tokens_per_rank, "
        "int combine_payload_offset, ScalarType out_dtype, int hidden_size) -> Tensor(a)");
    module.def("moe_a2a_get_aux_data_size(int ep_size, int max_num_tokens, int? eplb_stats_num_experts=None) -> int",
        &tensorrt_llm::torch_ext::moe_comm::moeA2AGetAuxDataSizeOp);
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, module)
{
    module.impl("moe_a2a_dispatch", &tensorrt_llm::torch_ext::moe_comm::moeA2ADispatchOp);
    module.impl("moe_a2a_combine", &tensorrt_llm::torch_ext::moe_comm::moeA2ACombineOp);
    module.impl("moe_a2a_initialize", &tensorrt_llm::torch_ext::moe_comm::moeA2AInitializeOp);
    module.impl("moe_a2a_sanitize_expert_ids", &tensorrt_llm::torch_ext::moe_comm::moeA2ASanitizeExpertIdsOp);
    module.impl(
        "moe_a2a_get_combine_payload_tensor", &tensorrt_llm::torch_ext::moe_comm::moeA2AGetCombinePayloadTensorOp);
}
