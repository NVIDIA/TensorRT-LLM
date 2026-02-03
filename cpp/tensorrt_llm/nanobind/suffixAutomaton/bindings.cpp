/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 *
 * Adapted from Baseten's sa_spec library (Apache-2.0)
 * https://github.com/basetenlabs/sa_spec
 */

#include "bindings.h"
#include "tensorrt_llm/nanobind/common/customCasters.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// Forward declarations - we don't include the suffix automaton headers directly
// because they contain macros that redefine cudaStream_t to int for non-CUDA compilers.
// These functions are implemented in the CUDA-compiled suffixAutomatonKernels.cu
// Note: Must use the _v1 inline namespace to match the TRTLLM_NAMESPACE_BEGIN macro
namespace tensorrt_llm::_v1::kernels::speculative_decoding::suffix_automaton
{
// Forward declaration of the opaque SuffixAutomaton type
struct SuffixAutomaton;

struct SuffixAutomatonExtendParams
{
    int batchSize{0};
    int draftLength{0};
    SuffixAutomaton* slots{nullptr};
    int const* batchIndices{nullptr};
    int* matchLenOut{nullptr};
    int* draftTokensOut{nullptr};
    int const* acceptedTokensIn{nullptr};
    int const* acceptedLensIn{nullptr};
};

void invokeSuffixAutomatonExtend(SuffixAutomatonExtendParams const& params, cudaStream_t stream);
size_t getSuffixAutomatonStateSize();
size_t getSuffixAutomatonMaxSlots();
size_t getSuffixAutomatonMaxSeqLen();

// Functions for building automatons - these are implemented in the .cu file
// We'll call them via host-side wrapper functions
void buildAutomatonFromTokens(SuffixAutomaton* sa, int const* tokens, int numTokens);
void initAutomaton(SuffixAutomaton* sa);
} // namespace tensorrt_llm::_v1::kernels::speculative_decoding::suffix_automaton

namespace sa = tensorrt_llm::_v1::kernels::speculative_decoding::suffix_automaton;

namespace tensorrt_llm::nanobind::suffix_automaton
{

void initBindings(nb::module_& m)
{
    // Export configuration constants
    m.attr("MAX_SEQUENCE_LENGTH") = nb::int_(sa::getSuffixAutomatonMaxSeqLen());
    m.attr("MAX_SLOTS") = nb::int_(sa::getSuffixAutomatonMaxSlots());
    m.attr("STATE_SIZE_BYTES") = nb::int_(sa::getSuffixAutomatonStateSize());

    // Export the extend function that invokes the CUDA kernel
    m.def(
        "invoke_extend",
        [](int batchSize, int draftLength, at::Tensor slots, at::Tensor batchIndices, at::Tensor matchLenOut,
            at::Tensor draftTokensOut, at::Tensor acceptedTokensIn, at::Tensor acceptedLensIn)
        {
            // Validate inputs
            TORCH_CHECK(slots.is_cuda(), "slots must be a CUDA tensor");
            TORCH_CHECK(batchIndices.is_cuda(), "batchIndices must be a CUDA tensor");
            TORCH_CHECK(matchLenOut.is_cuda(), "matchLenOut must be a CUDA tensor");
            TORCH_CHECK(draftTokensOut.is_cuda(), "draftTokensOut must be a CUDA tensor");
            TORCH_CHECK(acceptedTokensIn.is_cuda(), "acceptedTokensIn must be a CUDA tensor");
            TORCH_CHECK(acceptedLensIn.is_cuda(), "acceptedLensIn must be a CUDA tensor");

            sa::SuffixAutomatonExtendParams params;
            params.batchSize = batchSize;
            params.draftLength = draftLength;
            params.slots = reinterpret_cast<sa::SuffixAutomaton*>(slots.data_ptr());
            params.batchIndices = batchIndices.data_ptr<int>();
            params.matchLenOut = matchLenOut.data_ptr<int>();
            params.draftTokensOut = draftTokensOut.data_ptr<int>();
            params.acceptedTokensIn = static_cast<int const*>(acceptedTokensIn.data_ptr<int>());
            params.acceptedLensIn = static_cast<int const*>(acceptedLensIn.data_ptr<int>());

            cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
            sa::invokeSuffixAutomatonExtend(params, stream);
        },
        nb::arg("batch_size"), nb::arg("draft_length"), nb::arg("slots"), nb::arg("batch_indices"),
        nb::arg("match_len_out"), nb::arg("draft_tokens_out"), nb::arg("accepted_tokens_in"),
        nb::arg("accepted_lens_in"), "Invoke suffix automaton extend CUDA kernel");

    // Helper function to allocate workspace for suffix automaton states
    m.def(
        "allocate_workspace",
        [](int maxSlots)
        {
            size_t stateSize = sa::getSuffixAutomatonStateSize();
            size_t totalSize = static_cast<size_t>(maxSlots) * stateSize;

            auto options = at::TensorOptions().dtype(at::kByte).device(at::kCUDA);
            at::Tensor workspace = at::zeros({static_cast<int64_t>(totalSize)}, options);

            return workspace;
        },
        nb::arg("max_slots"), "Allocate GPU workspace for suffix automaton states");

    // Helper function to get workspace size
    m.def(
        "get_workspace_size",
        [](int maxSlots)
        {
            size_t stateSize = sa::getSuffixAutomatonStateSize();
            return static_cast<size_t>(maxSlots) * stateSize;
        },
        nb::arg("max_slots"), "Get required workspace size in bytes for suffix automaton states");

    // Build a suffix automaton on the host from context tokens
    // Returns a CPU tensor containing the serialized SuffixAutomaton state
    m.def(
        "build_automaton_host",
        [](std::vector<int> const& tokens)
        {
            size_t stateSize = sa::getSuffixAutomatonStateSize();

            // Allocate pinned memory for the state
            auto options = at::TensorOptions().dtype(at::kByte).device(at::kCPU).pinned_memory(true);
            at::Tensor hostState = at::zeros({static_cast<int64_t>(stateSize)}, options);

            // Get pointer to the SuffixAutomaton struct and build it
            sa::SuffixAutomaton* sa_ptr = reinterpret_cast<sa::SuffixAutomaton*>(hostState.data_ptr());
            sa::initAutomaton(sa_ptr);
            sa::buildAutomatonFromTokens(sa_ptr, tokens.data(), static_cast<int>(tokens.size()));

            return hostState;
        },
        nb::arg("tokens"), "Build a suffix automaton on host from context tokens. Returns pinned CPU tensor.");

    // Copy a host-built suffix automaton state to a GPU slot
    m.def(
        "copy_state_to_slot",
        [](at::Tensor hostState, at::Tensor gpuSlots, int slotIndex)
        {
            TORCH_CHECK(!hostState.is_cuda(), "hostState must be a CPU tensor");
            TORCH_CHECK(gpuSlots.is_cuda(), "gpuSlots must be a CUDA tensor");

            size_t stateSize = sa::getSuffixAutomatonStateSize();
            TORCH_CHECK(hostState.numel() >= static_cast<int64_t>(stateSize),
                "hostState is too small for a SuffixAutomaton state");

            size_t offset = static_cast<size_t>(slotIndex) * stateSize;
            TORCH_CHECK(offset + stateSize <= static_cast<size_t>(gpuSlots.numel()),
                "slotIndex is out of bounds for gpuSlots");

            // Async copy from host to device
            cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
            cudaMemcpyAsync(static_cast<uint8_t*>(gpuSlots.data_ptr()) + offset, hostState.data_ptr(), stateSize,
                cudaMemcpyHostToDevice, stream);
        },
        nb::arg("host_state"), nb::arg("gpu_slots"), nb::arg("slot_index"),
        "Copy a host-built suffix automaton state to a GPU slot (async)");

    // Clear a suffix automaton state at a GPU slot (reset for reuse)
    m.def(
        "clear_slot",
        [](at::Tensor gpuSlots, int slotIndex)
        {
            TORCH_CHECK(gpuSlots.is_cuda(), "gpuSlots must be a CUDA tensor");

            size_t stateSize = sa::getSuffixAutomatonStateSize();
            size_t offset = static_cast<size_t>(slotIndex) * stateSize;
            TORCH_CHECK(offset + stateSize <= static_cast<size_t>(gpuSlots.numel()),
                "slotIndex is out of bounds for gpuSlots");

            // Zero out the slot
            cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
            cudaMemsetAsync(static_cast<uint8_t*>(gpuSlots.data_ptr()) + offset, 0, stateSize, stream);
        },
        nb::arg("gpu_slots"), nb::arg("slot_index"), "Clear a suffix automaton state at a GPU slot");
}

} // namespace tensorrt_llm::nanobind::suffix_automaton
