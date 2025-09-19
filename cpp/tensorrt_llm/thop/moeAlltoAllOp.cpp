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
    COMPLETION_FLAGS_OFFSET_INDEX = 4,
    SEND_INDICES_OFFSET_INDEX = 5,
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
    size_t completion_flags_offset;
    size_t sendIndices_offset;
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
    
    // completion_flags
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets.completion_flags_offset = offset;
    offset += epSize * SIZEOF_INT32;
    
    // send_indices
    offset = alignOffset(offset, CACHELINE_ALIGNMENT);
    offsets.sendIndices_offset = offset;
    offset += epSize * maxNumTokensPerRank * SIZEOF_INT32;
    
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
//   - recvCounters: [ep_size] tensor tracking tokens received from each rank
//
// Note: token_selected_experts is used for routing but is NOT automatically included as a payload.
//       If you want to dispatch token_selected_experts, include it explicitly in inputPayloads.
std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor> moeA2ADispatchOp(
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
    
    // Create send_indices tensor - view into workspace
    // Initialized to -1 in prepare dispatch kernel
    torch::Tensor sendIndices = torch::from_blob(
        rank_workspace + offsets.sendIndices_offset,
        {localNumTokens, epSize},
        options
    );

    // Setup dispatch parameters
    MoeA2ADispatchParams params{};
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
        params.completion_flags[target_rank] = reinterpret_cast<uint32_t*>(target_workspace + offsets.completion_flags_offset);

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
    params.send_indices = sendIndices.data_ptr<int>();
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

    return std::make_tuple(std::move(recvBuffers), sendCounters, sendIndices, recvCounters);
}

// MoE All-to-All Combine Operation
// This operation pulls back and sums tokens that were dispatched to different expert ranks.
// Each rank acts as a sender in dispatch stage, pulling back the processed results of tokens it originally sent.
//
// Inputs:
//   - sendIndices: [local_num_tokens, ep_size] tensor from dispatch showing where tokens were sent
//   - payload: [ep_size, max_tokens_per_rank, elements_per_token] MoE-processed results (NOT on workspace)
//              This contains the expert outputs for all tokens that were dispatched to this rank
//   - workspace: [ep_size, size_per_rank] unified virtual memory workspace for synchronization
//   - maxTokensPerRank: Maximum number of tokens that can be received per rank
//   - epRank: Current expert parallel rank
//   - epSize: Total expert parallel size
//   - topK: Number of experts selected per token
//
// Returns:
//   - output: [local_num_tokens, elements_per_token] combined/summed results
//
// Note: The payload is the receive buffer from MoE processing. It needs to be copied to
// workspace before the combine kernel can gather results across ranks.
torch::Tensor moeA2ACombineOp(torch::Tensor const& sendIndices, torch::Tensor const& payload,
    torch::Tensor const& workspace, int64_t maxTokensPerRank, int64_t epRank, int64_t epSize, int64_t topK)
{
    using tensorrt_llm::kernels::moe_a2a::MoeA2ACombineParams;
    using tensorrt_llm::kernels::moe_a2a::moe_a2a_combine_launch;
    using tensorrt_llm::kernels::moe_a2a::kMaxTopK;

    // Validate inputs
    CHECK_INPUT(sendIndices, torch::kInt32);
    TORCH_CHECK(sendIndices.dim() == 2, "sendIndices must be a 2D tensor");
    TORCH_CHECK(sendIndices.size(1) == epSize, "sendIndices must have epSize columns");

    int64_t localNumTokens = sendIndices.size(0);
    TORCH_CHECK(localNumTokens > 0, "localNumTokens must be positive");

    CHECK_TH_CUDA(payload);
    TORCH_CHECK(payload.dim() == 3, "payload must be a 3D tensor [ep_size, max_tokens_per_rank, elements_per_token]");
    TORCH_CHECK(payload.size(0) == epSize, "payload first dimension must equal epSize");
    TORCH_CHECK(payload.size(1) == maxTokensPerRank, "payload second dimension must equal maxTokensPerRank");

    int64_t elementsPerToken = payload.size(2);
    TORCH_CHECK(elementsPerToken > 0, "elementsPerToken must be positive");
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");
    TORCH_CHECK(topK > 0 && topK <= kMaxTopK, "topK must be in the range (0, kMaxTopK]");

    // Validate workspace
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    TORCH_CHECK(workspace.dim() == 2, "workspace must be a 2D tensor of shape [epSize, sizePerRank]");
    TORCH_CHECK(workspace.size(0) == epSize, "workspace first dimension must equal epSize");

    // Map torch dtype to nvinfer1::DataType
    nvinfer1::DataType nvDtype;
    if (payload.dtype() == torch::kFloat16)
    {
        nvDtype = nvinfer1::DataType::kHALF;
    }
    else if (payload.dtype() == torch::kBFloat16)
    {
        nvDtype = nvinfer1::DataType::kBF16;
    }
    else if (payload.dtype() == torch::kFloat32)
    {
        nvDtype = nvinfer1::DataType::kFLOAT;
    }
    else
    {
        TORCH_CHECK(false, "Unsupported data type for payload");
    }

    // Calculate auxiliary data offsets
    MoeA2ADataOffsets offsets = calculateOffsets(epSize, maxTokensPerRank);
    
    // Calculate workspace requirements
    int elementSize = static_cast<int>(payload.dtype().itemsize());
    int64_t bytesPerPayload = epSize * maxTokensPerRank * elementsPerToken * elementSize;
    int64_t sizePerRank = workspace.size(1);
    int64_t requiredSize = offsets.payload_data_offset + bytesPerPayload;
    
    TORCH_CHECK(sizePerRank >= requiredSize, 
        "Workspace size per rank insufficient. Need at least ", requiredSize,
        " bytes (", offsets.payload_data_offset, " for auxiliary data + ", bytesPerPayload, 
        " for payload), but got ", sizePerRank);

    // Get workspace base pointer
    auto* workspace_ptr = workspace.data_ptr<uint8_t>();

    // Prepare for combine: zero output and copy payload to workspace recv buffer
    auto* current_rank_workspace = workspace_ptr + (epRank * workspace.stride(0));
    
    // Create output tensor (local on current rank), no need for initialization
    torch::Tensor output = torch::empty({localNumTokens, elementsPerToken}, payload.options());

    // Setup combine parameters
    MoeA2ACombineParams params{};
    params.ep_size = static_cast<int>(epSize);
    params.ep_rank = static_cast<int>(epRank);
    params.local_num_tokens = static_cast<int>(localNumTokens);
    params.max_tokens_per_rank = static_cast<int>(maxTokensPerRank);
    params.top_k = static_cast<int>(topK);
    params.send_indices = sendIndices.data_ptr<int>();
    params.output_data = output.data_ptr();
    params.elements_per_token = static_cast<int>(elementsPerToken);
    params.dtype = nvDtype;
    params.recv_counters = reinterpret_cast<int const*>(current_rank_workspace + offsets.recv_counters_offset);
    params.stream = at::cuda::getCurrentCUDAStream();

    // Set up completion flag pointers for all ranks (reuse from dispatch)
    for (int rank = 0; rank < epSize; rank++)
    {
        uint8_t* rank_base = workspace_ptr + rank * workspace.stride(0);
        params.completion_flags[rank] = reinterpret_cast<uint32_t*>(rank_base + offsets.completion_flags_offset);
    }
    params.flag_val = reinterpret_cast<uint32_t*>(current_rank_workspace + offsets.flag_val_offset);
    
    // Calculate and store recv buffer pointers
    for (int target_rank = 0; target_rank < epSize; target_rank++)
    {
        // Each rank gets workspace[target_rank] - calculate base pointer
        auto* target_workspace = workspace_ptr + (target_rank * workspace.stride(0));
        
        // Store receive buffer pointer for this rank (at payload_data_offset)
        params.recv_buffers[target_rank] = target_workspace + offsets.payload_data_offset;
    }

    // Prepare combine: zero output and stage payload into current rank's workspace slot
    params.prepare_payload = payload.data_ptr();
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
    moe_a2a_metainfo[COMPLETION_FLAGS_OFFSET_INDEX] = static_cast<int64_t>(offsets.completion_flags_offset);
    moe_a2a_metainfo[SEND_INDICES_OFFSET_INDEX] = static_cast<int64_t>(offsets.sendIndices_offset);
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
    module.def("MOE_A2A_COMPLETION_FLAGS_OFFSET_INDEX() -> int", []() -> int64_t { return static_cast<int64_t>(torch_ext::COMPLETION_FLAGS_OFFSET_INDEX); });
    module.def("MOE_A2A_SEND_INDICES_OFFSET_INDEX() -> int", []() -> int64_t { return static_cast<int64_t>(torch_ext::SEND_INDICES_OFFSET_INDEX); });
    module.def("MOE_A2A_PAYLOAD_DATA_OFFSET_INDEX() -> int", []() -> int64_t { return static_cast<int64_t>(torch_ext::PAYLOAD_DATA_OFFSET_INDEX); });
    
    module.def(
        "moe_a2a_dispatch(Tensor token_selected_experts, Tensor[] input_payloads, Tensor workspace, int "
        "max_tokens_per_rank, "
        "int ep_rank, int ep_size, int top_k, int num_experts) -> (Tensor[], Tensor, Tensor, Tensor)");
    module.def(
        "moe_a2a_combine(Tensor send_indices, Tensor payload, Tensor workspace, "
        "int max_tokens_per_rank, int ep_rank, int ep_size, int top_k) -> Tensor");
    module.def(
        "moe_a2a_initialize(Tensor workspace, int ep_rank, int ep_size, int max_num_tokens_per_rank) -> Tensor");
    module.def(
        "moe_a2a_sanitize_expert_ids(Tensor expert_ids, Tensor recv_counters, int invalid_expert_id) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, module)
{
    module.impl("moe_a2a_dispatch", &torch_ext::moeA2ADispatchOp);
    module.impl("moe_a2a_combine", &torch_ext::moeA2ACombineOp);
    module.impl("moe_a2a_initialize", &torch_ext::moeA2AInitializeOp);
    module.impl("moe_a2a_sanitize_expert_ids", &torch_ext::moeA2ASanitizeExpertIdsOp);
}
