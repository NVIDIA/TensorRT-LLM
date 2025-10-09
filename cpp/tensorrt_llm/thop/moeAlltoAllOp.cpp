/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/thop/thUtils.h"
#include "tensorrt_llm/common/envUtils.h"

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

namespace torch_ext
{

// Enum for indexing into moe_a2a_metainfo tensor
enum MoeA2AMetaInfoIndex {
    FLAG_VAL_OFFSET_INDEX = 0,
    LOCAL_TOKEN_COUNTER_OFFSET_INDEX = 1,
    SEND_COUNTERS_OFFSET_INDEX = 2,
    RECV_COUNTERS_OFFSET_INDEX = 3,
    // Dispatch completion flags offset
    DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX = 4,
    // Combine completion flags offset
    COMBINE_COMPLETION_FLAGS_OFFSET_INDEX = 5,
    PAYLOAD_DATA_OFFSET_INDEX = 6,
    NUM_METAINFO_FIELDS = 7
};

namespace
{

// Helper function to align offset to specified byte boundary
inline size_t alignOffset(size_t offset, size_t alignment)
{
    return (offset + alignment - 1) & ~(alignment - 1);
}

// Structure to hold auxiliary data offsets
struct MoeA2ADataOffsets
{
    size_t flag_val_offset;
    size_t local_token_counter_offset;
    size_t send_counters_offset;
    size_t recv_counters_offset;
    size_t dispatch_completion_flags_offset;
    size_t combine_completion_flags_offset;
    size_t topk_target_ranks_offset;
    size_t topk_send_indices_offset;
    size_t payload_data_offset;
};

// Calculate auxiliary data offsets
MoeA2ADataOffsets calculateOffsets(int epSize, int maxNumTokensPerRank)
{
    constexpr size_t SIZEOF_INT32 = 4;
    constexpr size_t CACHELINE_ALIGNMENT     = 128;
    
    MoeA2ADataOffsets offsets{};
    size_t offset = 0;

    // flag_val
    offsets.flag_val_offset = offset;
    offset += SIZEOF_INT32;
    
    // local_token_counter
    offsets.local_token_counter_offset = offset;
    offset += SIZEOF_INT32;
    
    // send_counters
    offsets.send_counters_offset = offset;
    offset += epSize * SIZEOF_INT32;
    
    // recv_counters
    offsets.recv_counters_offset = offset;
    offset += epSize * SIZEOF_INT32;
    
    // dispatch completion flags
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets.dispatch_completion_flags_offset = offset;
    offset += epSize * SIZEOF_INT32;

    // combine completion flags
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets.combine_completion_flags_offset = offset;
    offset += epSize * SIZEOF_INT32;
    

    // topk_target_ranks: [maxNumTokensPerRank, kMaxTopK]
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets.topk_target_ranks_offset = offset;
    offset += static_cast<size_t>(maxNumTokensPerRank)
        * static_cast<size_t>(tensorrt_llm::kernels::moe_a2a::kMaxTopK) * SIZEOF_INT32;

    // topk_send_indices: [maxNumTokensPerRank, kMaxTopK]
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets.topk_send_indices_offset = offset;
    offset += static_cast<size_t>(maxNumTokensPerRank)
        * static_cast<size_t>(tensorrt_llm::kernels::moe_a2a::kMaxTopK) * SIZEOF_INT32;

    // payload data
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets.payload_data_offset = offset;
    
    return offsets;
}

// MoE All-to-All Dispatch Operation
// This operation dispatches tokens and their associated payloads to different expert ranks.
//
// Inputs:
//   - tokenSelectedExperts: [local_num_tokens, top_k] tensor of expert indices
//   - inputPayloads: List of tensors with shape [local_num_tokens, ...] containing data to dispatch
//   - workspace: [ep_size, size_per_rank] unified virtual memory workspace where
//                size_per_rank = sum(ep_size * max_tokens_per_rank * elements_per_token * element_size) for all
//                payloads
//   - maxTokensPerRank: Maximum number of tokens that can be received per rank
//   - epRank: Current expert parallel rank
//   - epSize: Total expert parallel size
//   - topK: Number of experts selected per token
//   - numExperts: Total number of experts (must be divisible by epSize)
//
// Returns:
//   - recvBuffers: List of receive buffers, one for each payload
//   - sendCounters: [ep_size] tensor tracking tokens sent to each rank (local)
//   - recvCounters: [ep_size] tensor tracking tokens received from each rank (all ranks)
//   - topkTargetRanks: [local_num_tokens, top_k] compact routing - target EP rank per k
//   - topkSendIndices: [local_num_tokens, top_k] compact routing - dst slot per k
//
// Note: token_selected_experts is used for routing but is NOT automatically included as a payload.
//       If you want to dispatch token_selected_experts, include it explicitly in inputPayloads.
std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t> moeA2ADispatchOp(
    torch::Tensor const& tokenSelectedExperts, std::vector<torch::Tensor> const& inputPayloads,
    torch::Tensor const& workspace, int64_t maxTokensPerRank, int64_t epRank, int64_t epSize, int64_t topK, int64_t numExperts)
{
    using tensorrt_llm::kernels::moe_a2a::PayloadDescriptor;
    using tensorrt_llm::kernels::moe_a2a::MoeA2ADispatchParams;
    using tensorrt_llm::kernels::moe_a2a::moe_a2a_dispatch_launch;
    using tensorrt_llm::kernels::moe_a2a::kMaxTopK;
    using tensorrt_llm::kernels::moe_a2a::kMaxPayloads;

    // Validate inputs
    CHECK_INPUT(tokenSelectedExperts, torch::kInt32);
    TORCH_CHECK(tokenSelectedExperts.dim() == 2, "tokenSelectedExperts must be a 2D tensor");
    TORCH_CHECK(tokenSelectedExperts.size(1) == topK, "tokenSelectedExperts must have topK columns");

    int64_t localNumTokens = tokenSelectedExperts.size(0);
    TORCH_CHECK(localNumTokens > 0, "localNumTokens must be positive");
    TORCH_CHECK(maxTokensPerRank > 0, "maxTokensPerRank must be positive");
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");
    TORCH_CHECK(topK > 0 && topK <= kMaxTopK, "topK must be in the range (0, kMaxTopK]");
    TORCH_CHECK(!inputPayloads.empty(), "inputPayloads must not be empty");
    TORCH_CHECK(inputPayloads.size() <= kMaxPayloads, "Too many input payloads");
    TORCH_CHECK(numExperts >= epSize, "numExperts must be greater than or equal to epSize");
    TORCH_CHECK(numExperts % epSize == 0, "numExperts must be divisible by epSize for contiguous partitioning");

    // All input payloads must have the same first dimension (localNumTokens)
    for (auto const& payload : inputPayloads)
    {
        TORCH_CHECK(payload.dim() >= 1, "All payloads must have at least 1 dimension");
        TORCH_CHECK(payload.size(0) == localNumTokens,
            "All payloads must have the same first dimension as tokenSelectedExperts");
        TORCH_CHECK(payload.is_contiguous(), "All payloads must be contiguous");
    }

    // Validate workspace - unified virtual memory [epSize, sizePerRank]
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    // Don't check contiguous - MnnvlMemory creates strided tensors for multi-GPU
    TORCH_CHECK(workspace.dim() == 2, "workspace must be a 2D tensor of shape [epSize, sizePerRank]");
    TORCH_CHECK(workspace.size(0) == epSize, "workspace first dimension must equal epSize");

    // Calculate buffer sizes for all payloads
    // Each payload buffer needs space for data from ALL ranks: epSize * maxTokensPerRank * elementsPerToken
    int64_t totalBytesNeeded = 0;
    std::vector<int64_t> payloadByteSizes;
    std::vector<int> payloadElementSizes;
    std::vector<int> payloadElementsPerToken;

    for (auto const& payload : inputPayloads)
    {
        CHECK_CONTIGUOUS(payload);
        CHECK_TH_CUDA(payload);
        TORCH_CHECK(payload.dim() == 2, "payload must be a 2D tensor");
        TORCH_CHECK(
            payload.size(0) == localNumTokens, "payload must have the same first dimension as tokenSelectedExperts");

        int elementsPerToken = static_cast<int>(payload.size(1));
        int elementSize = static_cast<int>(payload.dtype().itemsize());
        // Each payload buffer stores data from ALL ranks
        int64_t bytesPerPayload = epSize * maxTokensPerRank * elementsPerToken * elementSize;

        payloadByteSizes.push_back(bytesPerPayload);
        payloadElementSizes.push_back(elementSize);
        payloadElementsPerToken.push_back(elementsPerToken);
        totalBytesNeeded += bytesPerPayload;
    }

    // Calculate auxiliary data offsets
    MoeA2ADataOffsets offsets = calculateOffsets(epSize, maxTokensPerRank);
    
    // Validate workspace size - must include space for auxiliary data + payloads
    int64_t sizePerRank = workspace.size(1);  // size in bytes since dtype is uint8
    int64_t requiredSize = offsets.payload_data_offset + totalBytesNeeded;
    TORCH_CHECK(sizePerRank >= requiredSize,
        "Workspace size per rank insufficient. "
        "Need at least ", requiredSize, " bytes (", offsets.payload_data_offset, " for auxiliary data + ", 
        totalBytesNeeded, " for payloads), but got ", sizePerRank);

    // Setup receive buffer pointers from unified workspace
    std::vector<PayloadDescriptor> payloadDescriptors;

    // Get base workspace pointer
    uint8_t* workspace_ptr = workspace.data_ptr<uint8_t>();

    // Setup payload descriptors for source data
    for (int i = 0; i < static_cast<int>(inputPayloads.size()); i++)
    {
        PayloadDescriptor desc{};
        desc.src_data = inputPayloads[i].data_ptr();
        desc.element_size = payloadElementSizes[i];
        desc.elements_per_token = payloadElementsPerToken[i];
        payloadDescriptors.push_back(desc);
    }


    // Create tensors for return values (these are views into workspace)
    auto options = tokenSelectedExperts.options().dtype(torch::kInt32);
    uint8_t* rank_workspace = workspace_ptr + epRank * workspace.stride(0);
    
    // Create send_counters tensor - view into workspace
    // Initialized to 0 in prepare dispatch kernel
    torch::Tensor sendCounters = torch::from_blob(
        rank_workspace + offsets.send_counters_offset,
        {epSize},
        options
    );
    
    // Create recv_counters tensor - view into workspace
    // No need for initialization
    torch::Tensor recvCounters = torch::from_blob(
        rank_workspace + offsets.recv_counters_offset,
        {epSize},
        options
    );
    
    // Create local_token_counter - view into workspace
    // Initialized to 0 in prepare dispatch kernel
    torch::Tensor localTokenCounter = torch::from_blob(
        rank_workspace + offsets.local_token_counter_offset,
        {1},
        options
    );
    

    // Allocate compact Top-K routing tensors [localNumTokens, topK]
    torch::Tensor topkTargetRanks = torch::empty({localNumTokens, topK}, options);
    torch::Tensor topkSendIndices = torch::empty({localNumTokens, topK}, options);

    // Setup dispatch parameters
    MoeA2ADispatchParams params{};
    params.one_block_per_token = tensorrt_llm::common::getEnvMoeA2AOneBlockPerToken();  // TODO: Decide this based on the workload
    params.token_selected_experts = tokenSelectedExperts.data_ptr<int32_t>();
    params.num_payloads = static_cast<int32_t>(payloadDescriptors.size());
    std::copy(payloadDescriptors.begin(), payloadDescriptors.end(), &params.payloads[0]);
    params.flag_val = reinterpret_cast<uint32_t*>(rank_workspace + offsets.flag_val_offset);
    
    // Calculate and store recv buffer pointers directly in params
    for (int target_rank = 0; target_rank < epSize; target_rank++)
    {
        // Each rank gets workspace[target_rank] - calculate base pointer
        uint8_t* target_workspace = workspace_ptr + (target_rank * workspace.stride(0));

        params.recv_counters[target_rank] = reinterpret_cast<int*>(target_workspace + offsets.recv_counters_offset);
        params.completion_flags[target_rank] = reinterpret_cast<uint32_t*>(target_workspace + offsets.dispatch_completion_flags_offset);

        int64_t offset = offsets.payload_data_offset;  // Start after auxiliary data

        for (int payload_idx = 0; payload_idx < static_cast<int>(inputPayloads.size()); payload_idx++)
        {
            // Store buffer pointer for kernel
            params.recv_buffers[target_rank][payload_idx] = target_workspace + offset;

            // Update offset for next payload
            offset += payloadByteSizes[payload_idx];
        }
    }
    params.max_tokens_per_rank = static_cast<int>(maxTokensPerRank);
    params.send_counters = sendCounters.data_ptr<int>();
    params.local_token_counter = localTokenCounter.data_ptr<int>();
    params.topk_target_ranks = topkTargetRanks.data_ptr<int>();
    params.topk_send_indices = topkSendIndices.data_ptr<int>();
    params.local_num_tokens = static_cast<int>(localNumTokens);
    params.ep_size = static_cast<int>(epSize);
    params.ep_rank = static_cast<int>(epRank);
    params.top_k = static_cast<int>(topK);
    params.num_experts_per_rank = static_cast<int>(numExperts) / static_cast<int>(epSize);
    params.stream = at::cuda::getCurrentCUDAStream();

    // Prepare for dispatch (zero counters/indices and increment flag_val)
    moe_a2a_prepare_dispatch_launch(params);

    // Launch the dispatch kernel
    moe_a2a_dispatch_launch(params);
    cudaError_t result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "moe_a2a_dispatch kernel launch failed: ", cudaGetErrorString(result));

    // Create tensor views for the current rank's receive buffers only
    std::vector<torch::Tensor> recvBuffers;
    auto* current_rank_workspace = workspace_ptr + (epRank * workspace.stride(0));
    int64_t offset = offsets.payload_data_offset;

    for (int payload_idx = 0; payload_idx < static_cast<int>(inputPayloads.size()); payload_idx++)
    {
        auto const& payload = inputPayloads[payload_idx];

        // Create tensor view for this payload - contains data from ALL ranks
        auto recvBuffer = torch::from_blob(current_rank_workspace + offset,
            {epSize, maxTokensPerRank, payloadElementsPerToken[payload_idx]}, payload.options());
        recvBuffers.push_back(recvBuffer);

        // Update offset for next payload
        offset += payloadByteSizes[payload_idx];
    }
    // Compute aligned offset after dispatch payloads for combine payload region
    constexpr size_t CACHELINE_ALIGNMENT = 128;
    int64_t combinePayloadOffset = static_cast<int64_t>(alignOffset(static_cast<size_t>(offset), CACHELINE_ALIGNMENT));

    return std::make_tuple(std::move(recvBuffers), sendCounters, recvCounters, topkTargetRanks, topkSendIndices, combinePayloadOffset);
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
torch::Tensor moeA2ACombineOp(torch::Tensor const& topkTargetRanks, torch::Tensor const& topkSendIndices,
    torch::Tensor const& recvCounters, torch::Tensor const& payload, torch::Tensor const& workspace,
    int64_t maxTokensPerRank, int64_t epRank, int64_t epSize, int64_t topK, int64_t combinePayloadOffset, bool payloadInWorkspace)
{
    using tensorrt_llm::kernels::moe_a2a::MoeA2ACombineParams;
    using tensorrt_llm::kernels::moe_a2a::moe_a2a_combine_launch;
    using tensorrt_llm::kernels::moe_a2a::kMaxTopK;

    // Validate inputs
    CHECK_INPUT(topkTargetRanks, torch::kInt32);
    CHECK_INPUT(topkSendIndices, torch::kInt32);
    CHECK_INPUT(recvCounters, torch::kInt32);
    TORCH_CHECK(topkTargetRanks.dim() == 2 && topkSendIndices.dim() == 2,
        "topkTargetRanks/topkSendIndices must be 2D [local_num_tokens, top_k]");
    TORCH_CHECK(topkTargetRanks.size(0) == topkSendIndices.size(0),
        "topkTargetRanks and topkSendIndices must have the same first dimension, which is local_num_tokens");
        int64_t localNumTokens = topkTargetRanks.size(0);
    TORCH_CHECK(topkTargetRanks.size(1) == topK && topkSendIndices.size(1) == topK,
        "topk tensors second dim must equal topK");
    

    CHECK_TH_CUDA(payload);
    CHECK_CONTIGUOUS(payload);
    TORCH_CHECK(payload.dim() == 3, "payload must be a 3D tensor [ep_size, max_tokens_per_rank, elements_per_token]");
    TORCH_CHECK(payload.size(0) == epSize, "payload first dimension must equal epSize");
    TORCH_CHECK(payload.size(1) == maxTokensPerRank, "payload second dimension must equal maxTokensPerRank");
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

    
    // Create output tensor (local on current rank), no need for initialization
    torch::Tensor output = torch::empty({localNumTokens, elementsPerToken}, payload.options());

    // Setup combine parameters
    MoeA2ACombineParams params{};
    params.one_block_per_token = tensorrt_llm::common::getEnvMoeA2AOneBlockPerToken();  // TODO: Decide this based on the workload
    params.ep_size = static_cast<int>(epSize);
    params.ep_rank = static_cast<int>(epRank);
    params.local_num_tokens = static_cast<int>(localNumTokens);
    params.max_tokens_per_rank = static_cast<int>(maxTokensPerRank);
    params.top_k = static_cast<int>(topK);
    params.topk_target_ranks = topkTargetRanks.data_ptr<int>();
    params.topk_send_indices = topkSendIndices.data_ptr<int>();
    params.output_data = output.data_ptr();
    params.elements_per_token = static_cast<int>(elementsPerToken);
    params.dtype = nvDtype;
    params.recv_counters = recvCounters.data_ptr<int>();
    params.stream = at::cuda::getCurrentCUDAStream();

    MoeA2ADataOffsets offsets = calculateOffsets(static_cast<int>(epSize), static_cast<int>(maxTokensPerRank));
    
    // Validate workspace and set synchronization pointers
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    TORCH_CHECK(workspace.dim() == 2 && workspace.size(0) == epSize, "workspace must be [ep_size, size_per_rank]");
    uint8_t* workspace_ptr = workspace.data_ptr<uint8_t>();
    int64_t sizePerRank = workspace.size(1);
    uint8_t* workspace_currank_base = workspace_ptr + epRank * workspace.stride(0);

    // If user claims payload is in workspace, ensure payload tensor matches combinePayloadOffset
    if (payloadInWorkspace)
    {
        TORCH_CHECK(payload.data_ptr() == workspace_currank_base + combinePayloadOffset,
            "payload_in_workspace is true but 'payload' dataptr does not match combinePayloadOffset");
    }

    int64_t payloadSize = payload.numel() * payload.element_size();
    TORCH_CHECK(combinePayloadOffset >= 0 && combinePayloadOffset + payloadSize <= sizePerRank,
        "workspace does not contain enough space for the payload region for combine. combine payload offset=", combinePayloadOffset,
        ", payload size needed=", payloadSize,
        ", workspace size per rank=", sizePerRank);

    for (int src_rank = 0; src_rank < epSize; src_rank++)
    {
        params.recv_buffers[src_rank] = workspace_ptr + src_rank * workspace.stride(0) + combinePayloadOffset;
    }

    // completion flags for all ranks (combine)
    for (int rank = 0; rank < epSize; rank++)
    {
        uint8_t* rank_base = workspace_ptr + rank * workspace.stride(0);
        params.completion_flags[rank] = reinterpret_cast<uint32_t*>(rank_base + offsets.combine_completion_flags_offset);
    }
    params.flag_val = reinterpret_cast<uint32_t*>(workspace_ptr + epRank * workspace.stride(0) + offsets.flag_val_offset);


    // If payload is not already in workspace, stage it into current rank's region
    if (!payloadInWorkspace)
    {
        params.prepare_payload = payload.data_ptr();
    }

    moe_a2a_prepare_combine_launch(params);

    
    // Launch the combine kernel
    moe_a2a_combine_launch(params);
    cudaError_t result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "moe_a2a_combine kernel launch failed: ", cudaGetErrorString(result));

    return output;
}

// Initialize auxiliary data in workspace
// This function sets up the initial values for flag_val and completion_flags
// 
// Inputs:
//   - workspace: [ep_size, size_per_rank] unified virtual memory workspace
//   - epRank: Current expert parallel rank
//   - epSize: Total expert parallel size
//
// Returns:
//   - Auxiliary data size (payload_data_offset) in bytes
//
// The function initializes:
//   - flag_val to 1 (on current rank)
//   - completion_flags to 0 (on all ranks)
torch::Tensor moeA2AInitializeOp(
    torch::Tensor const& workspace, int64_t epRank, int64_t epSize, int64_t maxNumTokensPerRank)
{
    // Validate inputs
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    TORCH_CHECK(workspace.dim() == 2, "workspace must be a 2D tensor of shape [epSize, sizePerRank]");
    TORCH_CHECK(workspace.size(0) == epSize, "workspace first dimension must equal epSize");
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");

    // Initialize workspace to zero
    workspace[epRank].zero_();

    // Calculate auxiliary data offsets
    MoeA2ADataOffsets offsets = calculateOffsets(epSize, maxNumTokensPerRank);
    
    // Return moe_a2a_metainfo as a tensor containing offsets
    torch::Tensor moe_a2a_metainfo = torch::zeros({NUM_METAINFO_FIELDS}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    moe_a2a_metainfo[FLAG_VAL_OFFSET_INDEX] = static_cast<int64_t>(offsets.flag_val_offset);
    moe_a2a_metainfo[LOCAL_TOKEN_COUNTER_OFFSET_INDEX] = static_cast<int64_t>(offsets.local_token_counter_offset);
    moe_a2a_metainfo[SEND_COUNTERS_OFFSET_INDEX] = static_cast<int64_t>(offsets.send_counters_offset);
    moe_a2a_metainfo[RECV_COUNTERS_OFFSET_INDEX] = static_cast<int64_t>(offsets.recv_counters_offset);
    moe_a2a_metainfo[DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX] = static_cast<int64_t>(offsets.dispatch_completion_flags_offset);
    moe_a2a_metainfo[COMBINE_COMPLETION_FLAGS_OFFSET_INDEX] = static_cast<int64_t>(offsets.combine_completion_flags_offset);
    moe_a2a_metainfo[PAYLOAD_DATA_OFFSET_INDEX] = static_cast<int64_t>(offsets.payload_data_offset);

    // Memset workspace to 0 and synchronize among ranks
    workspace[epRank].zero_();
    cudaDeviceSynchronize();
    tensorrt_llm::mpi::MpiComm::world().barrier();
    
    return moe_a2a_metainfo;
}

// Op: moe_a2a_sanitize_expert_ids
void moeA2ASanitizeExpertIdsOp(torch::Tensor expert_ids, torch::Tensor recv_counters, int64_t invalid_expert_id)
{
    CHECK_INPUT(expert_ids, torch::kInt32);
    CHECK_INPUT(recv_counters, torch::kInt32);
    TORCH_CHECK(expert_ids.dim() == 3, "expert_ids must be [ep_size, max_tokens_per_rank, top_k]");
    TORCH_CHECK(recv_counters.dim() == 1, "recv_counters must be [ep_size]");
    TORCH_CHECK(expert_ids.size(0) == recv_counters.size(0), "expert_ids and recv_counters must have the same ep_size");

    int ep_size = static_cast<int>(expert_ids.size(0));
    int max_tokens_per_rank = static_cast<int>(expert_ids.size(1));
    int top_k = static_cast<int>(expert_ids.size(2));

    tensorrt_llm::kernels::moe_a2a::moe_a2a_sanitize_expert_ids_launch(
        expert_ids.data_ptr<int32_t>(),
        recv_counters.data_ptr<int32_t>(),
        static_cast<int32_t>(invalid_expert_id),
        ep_size,
        max_tokens_per_rank,
        top_k,
        at::cuda::getCurrentCUDAStream());
}


// Return a workspace-backed tensor for combine payload region using from_blob
torch::Tensor moeA2AGetCombinePayloadTensorOp(
    torch::Tensor const& workspace,
    int64_t epRank,
    int64_t epSize,
    int64_t maxTokensPerRank,
    int64_t combinePayloadOffset,
    c10::ScalarType outDtype,
    int64_t hiddenSize)
{
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    TORCH_CHECK(workspace.dim() == 2, "workspace must be [ep_size, size_per_rank_bytes]");
    TORCH_CHECK(epRank >= 0 && epRank < workspace.size(0), "epRank out of range");
    TORCH_CHECK(epSize == workspace.size(0), "epSize mismatch with workspace");
    TORCH_CHECK(maxTokensPerRank > 0, "maxTokensPerRank must be positive");
    TORCH_CHECK(hiddenSize > 0, "hidden must be positive");

    int64_t sizePerRank = workspace.size(1); // bytes
    int64_t elementSize = static_cast<int64_t>(c10::elementSize(outDtype));
    int64_t bytesNeeded = epSize * maxTokensPerRank * hiddenSize * elementSize;
    TORCH_CHECK(combinePayloadOffset >= 0, "combine_payload_offset must be non-negative");
    TORCH_CHECK(
        combinePayloadOffset + bytesNeeded <= sizePerRank,
        "workspace does not have enough space for combine payload tensor. combine payload offset=", combinePayloadOffset,
        ", payload size needed=", bytesNeeded,
        ", workspace size per rank=", sizePerRank
    );

    uint8_t* base = workspace.data_ptr<uint8_t>();
    uint8_t* rankBase = base + epRank * workspace.stride(0);
    uint8_t* dataPtr = rankBase + combinePayloadOffset;

    auto options = workspace.options().dtype(outDtype);
    torch::Tensor t = torch::from_blob(
        dataPtr,
        {epSize * maxTokensPerRank, hiddenSize},
        options);
    return t;
}

} // anonymous namespace

} // namespace torch_ext

// PyTorch bindings
TORCH_LIBRARY_FRAGMENT(trtllm, module)
{
    // Export metainfo index constants
    module.def("MOE_A2A_FLAG_VAL_OFFSET_INDEX() -> int", []() -> int64_t { return static_cast<int64_t>(torch_ext::FLAG_VAL_OFFSET_INDEX); });
    module.def("MOE_A2A_LOCAL_TOKEN_COUNTER_OFFSET_INDEX() -> int", []() -> int64_t { return static_cast<int64_t>(torch_ext::LOCAL_TOKEN_COUNTER_OFFSET_INDEX); });
    module.def("MOE_A2A_SEND_COUNTERS_OFFSET_INDEX() -> int", []() -> int64_t { return static_cast<int64_t>(torch_ext::SEND_COUNTERS_OFFSET_INDEX); });
    module.def("MOE_A2A_RECV_COUNTERS_OFFSET_INDEX() -> int", []() -> int64_t { return static_cast<int64_t>(torch_ext::RECV_COUNTERS_OFFSET_INDEX); });
    // Backward-compat: legacy COMPLETION_FLAGS returns dispatch flags
    module.def("MOE_A2A_COMPLETION_FLAGS_OFFSET_INDEX() -> int", []() -> int64_t { return static_cast<int64_t>(torch_ext::DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX); });
    // New explicit exports
    module.def("MOE_A2A_DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX() -> int", []() -> int64_t { return static_cast<int64_t>(torch_ext::DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX); });
    module.def("MOE_A2A_COMBINE_COMPLETION_FLAGS_OFFSET_INDEX() -> int", []() -> int64_t { return static_cast<int64_t>(torch_ext::COMBINE_COMPLETION_FLAGS_OFFSET_INDEX); });
    module.def("MOE_A2A_PAYLOAD_DATA_OFFSET_INDEX() -> int", []() -> int64_t { return static_cast<int64_t>(torch_ext::PAYLOAD_DATA_OFFSET_INDEX); });
    
    module.def(
        "moe_a2a_dispatch(Tensor token_selected_experts, Tensor[] input_payloads, Tensor workspace, int "
        "max_tokens_per_rank, "
        "int ep_rank, int ep_size, int top_k, int num_experts) -> (Tensor[], Tensor, Tensor, Tensor, Tensor, int)");
    module.def(
        "moe_a2a_combine(Tensor topk_target_ranks, Tensor topk_send_indices, Tensor recv_counters, Tensor payload, "
        "Tensor workspace, int max_tokens_per_rank, int ep_rank, int ep_size, int top_k, int combine_payload_offset, bool payload_in_workspace) -> Tensor");
    module.def(
        "moe_a2a_initialize(Tensor workspace, int ep_rank, int ep_size, int max_num_tokens_per_rank) -> Tensor");
    module.def(
        "moe_a2a_sanitize_expert_ids(Tensor expert_ids, Tensor recv_counters, int invalid_expert_id) -> ()");
    module.def(
        "moe_a2a_get_combine_payload_tensor(Tensor workspace, int ep_rank, int ep_size, int max_tokens_per_rank, int combine_payload_offset, ScalarType out_dtype, int hidden) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, module)
{
    module.impl("moe_a2a_dispatch", &torch_ext::moeA2ADispatchOp);
    module.impl("moe_a2a_combine", &torch_ext::moeA2ACombineOp);
    module.impl("moe_a2a_initialize", &torch_ext::moeA2AInitializeOp);
    module.impl("moe_a2a_sanitize_expert_ids", &torch_ext::moeA2ASanitizeExpertIdsOp);
    module.impl("moe_a2a_get_combine_payload_tensor", &torch_ext::moeA2AGetCombinePayloadTensorOp);
}
