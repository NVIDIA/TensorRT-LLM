/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Include only the params header (structs + function declarations).
// We cannot include the full suffixAutomatonKernels.h because it transitively
// includes suffixAutomaton.h → saCudaCallable.h, which redefines cudaStream_t
// to int when __CUDACC__ is not defined (this file is compiled as C++, not CUDA).
#include "tensorrt_llm/kernels/speculativeDecoding/suffixAutomaton/suffixAutomatonParams.h"

namespace sa = tensorrt_llm::kernels::speculative_decoding::suffix_automaton;

namespace tensorrt_llm::nanobind::suffix_automaton
{

void initBindings(nb::module_& m)
{
    // Export the state size function (replaces static STATE_SIZE_BYTES constant)
    m.def(
        "get_state_size",
        [](int maxSeqLen)
        {
            TORCH_CHECK(maxSeqLen > 0, "maxSeqLen must be positive");
            return sa::getSuffixAutomatonStateSize(static_cast<size_t>(maxSeqLen));
        },
        nb::arg("max_seq_len"), "Get state size in bytes for given max sequence length");

    // Export the extend function that invokes the CUDA kernel
    m.def(
        "invoke_extend",
        [](int batchSize, int draftLength, int maxSlots, int maxSeqLen, at::Tensor slots, at::Tensor batchIndices,
            at::Tensor matchLenOut, at::Tensor draftTokensOut, at::Tensor acceptedTokensIn, at::Tensor acceptedLensIn)
        {
            // Validate inputs
            TORCH_CHECK(slots.is_cuda(), "slots must be a CUDA tensor");
            TORCH_CHECK(batchIndices.is_cuda(), "batchIndices must be a CUDA tensor");
            TORCH_CHECK(matchLenOut.is_cuda(), "matchLenOut must be a CUDA tensor");
            TORCH_CHECK(draftTokensOut.is_cuda(), "draftTokensOut must be a CUDA tensor");
            TORCH_CHECK(acceptedTokensIn.is_cuda(), "acceptedTokensIn must be a CUDA tensor");
            TORCH_CHECK(acceptedLensIn.is_cuda(), "acceptedLensIn must be a CUDA tensor");
            TORCH_CHECK(maxSeqLen > 0, "maxSeqLen must be positive");

            sa::SuffixAutomatonExtendParams params;
            params.batchSize = batchSize;
            params.draftLength = draftLength;
            params.maxSlots = maxSlots;
            params.maxSeqLen = maxSeqLen;
            params.slots = slots.data_ptr();
            params.batchIndices = batchIndices.data_ptr<int>();
            params.matchLenOut = matchLenOut.data_ptr<int>();
            params.draftTokensOut = draftTokensOut.data_ptr<int>();
            params.acceptedTokensIn = static_cast<int const*>(acceptedTokensIn.data_ptr<int>());
            params.acceptedLensIn = static_cast<int const*>(acceptedLensIn.data_ptr<int>());

            cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
            sa::invokeSuffixAutomatonExtend(params, stream);
        },
        nb::arg("batch_size"), nb::arg("draft_length"), nb::arg("max_slots"), nb::arg("max_seq_len"), nb::arg("slots"),
        nb::arg("batch_indices"), nb::arg("match_len_out"), nb::arg("draft_tokens_out"), nb::arg("accepted_tokens_in"),
        nb::arg("accepted_lens_in"), "Invoke suffix automaton extend CUDA kernel");

    // Export the extend function with ngram support that invokes the CUDA kernel
    m.def(
        "invoke_extend_ngram",
        [](int batchSize, int draftLength, int maxNgramSize, int maxSlots, int maxSeqLen, at::Tensor slots,
            at::Tensor batchIndices, at::Tensor matchLenOut, at::Tensor draftTokensOut, at::Tensor acceptedTokensIn,
            at::Tensor acceptedLensIn)
        {
            // Validate inputs
            TORCH_CHECK(slots.is_cuda(), "slots must be a CUDA tensor");
            TORCH_CHECK(batchIndices.is_cuda(), "batchIndices must be a CUDA tensor");
            TORCH_CHECK(matchLenOut.is_cuda(), "matchLenOut must be a CUDA tensor");
            TORCH_CHECK(draftTokensOut.is_cuda(), "draftTokensOut must be a CUDA tensor");
            TORCH_CHECK(acceptedTokensIn.is_cuda(), "acceptedTokensIn must be a CUDA tensor");
            TORCH_CHECK(acceptedLensIn.is_cuda(), "acceptedLensIn must be a CUDA tensor");
            TORCH_CHECK(maxSeqLen > 0, "maxSeqLen must be positive");

            sa::SuffixAutomatonExtendNgramParams params;
            params.batchSize = batchSize;
            params.draftLength = draftLength;
            params.maxNgramSize = maxNgramSize;
            params.maxSlots = maxSlots;
            params.maxSeqLen = maxSeqLen;
            params.slots = slots.data_ptr();
            params.batchIndices = batchIndices.data_ptr<int>();
            params.matchLenOut = matchLenOut.data_ptr<int>();
            params.draftTokensOut = draftTokensOut.data_ptr<int>();
            params.acceptedTokensIn = static_cast<int const*>(acceptedTokensIn.data_ptr<int>());
            params.acceptedLensIn = static_cast<int const*>(acceptedLensIn.data_ptr<int>());

            cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
            sa::invokeSuffixAutomatonExtendNgram(params, stream);
        },
        nb::arg("batch_size"), nb::arg("draft_length"), nb::arg("max_ngram_size"), nb::arg("max_slots"),
        nb::arg("max_seq_len"), nb::arg("slots"), nb::arg("batch_indices"), nb::arg("match_len_out"),
        nb::arg("draft_tokens_out"), nb::arg("accepted_tokens_in"), nb::arg("accepted_lens_in"),
        "Invoke suffix automaton extend CUDA kernel with ngram support. "
        "If max_ngram_size == -1, uses longest match. "
        "If max_ngram_size > 0, tries ngram sizes from max down to 1.");

    // Export the global search function (cross-request pattern sharing)
    m.def(
        "invoke_global_search",
        [](int batchSize, int draftLength, int maxNgramSize, int maxSlots, int maxSeqLen, at::Tensor slots,
            at::Tensor batchIndices, at::Tensor activeSlotMask, at::Tensor matchLenOut, at::Tensor matchSlotOut,
            at::Tensor draftTokensOut, at::Tensor acceptedTokensIn, at::Tensor acceptedLensIn)
        {
            TORCH_CHECK(slots.is_cuda(), "slots must be a CUDA tensor");
            TORCH_CHECK(batchIndices.is_cuda(), "batchIndices must be a CUDA tensor");
            TORCH_CHECK(activeSlotMask.is_cuda(), "activeSlotMask must be a CUDA tensor");
            TORCH_CHECK(matchLenOut.is_cuda(), "matchLenOut must be a CUDA tensor");
            TORCH_CHECK(matchSlotOut.is_cuda(), "matchSlotOut must be a CUDA tensor");
            TORCH_CHECK(draftTokensOut.is_cuda(), "draftTokensOut must be a CUDA tensor");
            TORCH_CHECK(acceptedTokensIn.is_cuda(), "acceptedTokensIn must be a CUDA tensor");
            TORCH_CHECK(acceptedLensIn.is_cuda(), "acceptedLensIn must be a CUDA tensor");
            TORCH_CHECK(maxSeqLen > 0, "maxSeqLen must be positive");

            sa::SuffixAutomatonGlobalSearchParams params;
            params.batchSize = batchSize;
            params.draftLength = draftLength;
            params.maxNgramSize = maxNgramSize;
            params.maxSlots = maxSlots;
            params.maxSeqLen = maxSeqLen;
            params.slots = slots.data_ptr();
            params.batchIndices = batchIndices.data_ptr<int>();
            params.activeSlotMask = static_cast<int const*>(activeSlotMask.data_ptr<int>());
            params.matchLenOut = matchLenOut.data_ptr<int>();
            params.matchSlotOut = matchSlotOut.data_ptr<int>();
            params.draftTokensOut = draftTokensOut.data_ptr<int>();
            params.acceptedTokensIn = static_cast<int const*>(acceptedTokensIn.data_ptr<int>());
            params.acceptedLensIn = static_cast<int const*>(acceptedLensIn.data_ptr<int>());

            cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
            sa::invokeSuffixAutomatonGlobalSearch(params, stream);
        },
        nb::arg("batch_size"), nb::arg("draft_length"), nb::arg("max_ngram_size"), nb::arg("max_slots"),
        nb::arg("max_seq_len"), nb::arg("slots"), nb::arg("batch_indices"), nb::arg("active_slot_mask"),
        nb::arg("match_len_out"), nb::arg("match_slot_out"), nb::arg("draft_tokens_out"), nb::arg("accepted_tokens_in"),
        nb::arg("accepted_lens_in"),
        "Invoke global search across all active SA states. "
        "Launches extend + search kernels on the same stream for cross-request pattern sharing.");

    // Helper function to allocate workspace for suffix automaton states
    m.def(
        "allocate_workspace",
        [](int maxSlots, int maxSeqLen)
        {
            TORCH_CHECK(maxSlots > 0, "maxSlots must be positive");
            TORCH_CHECK(maxSeqLen > 0, "maxSeqLen must be positive");

            size_t stateSize = sa::getSuffixAutomatonStateSize(static_cast<size_t>(maxSeqLen));
            size_t totalSize = static_cast<size_t>(maxSlots) * stateSize;

            auto options = at::TensorOptions().dtype(at::kByte).device(at::kCUDA);
            at::Tensor workspace = at::zeros({static_cast<int64_t>(totalSize)}, options);

            return workspace;
        },
        nb::arg("max_slots"), nb::arg("max_seq_len"), "Allocate GPU workspace for suffix automaton states");

    // Helper function to get workspace size
    m.def(
        "get_workspace_size",
        [](int maxSlots, int maxSeqLen)
        {
            TORCH_CHECK(maxSlots > 0, "maxSlots must be positive");
            TORCH_CHECK(maxSeqLen > 0, "maxSeqLen must be positive");

            size_t stateSize = sa::getSuffixAutomatonStateSize(static_cast<size_t>(maxSeqLen));
            return static_cast<size_t>(maxSlots) * stateSize;
        },
        nb::arg("max_slots"), nb::arg("max_seq_len"),
        "Get required workspace size in bytes for suffix automaton states");

    // Build a suffix automaton on the host from context tokens
    // Returns a CPU tensor containing the serialized SuffixAutomaton state
    m.def(
        "build_automaton_host",
        [](std::vector<int> const& tokens, int maxSeqLen)
        {
            TORCH_CHECK(maxSeqLen > 0, "maxSeqLen must be positive");

            size_t stateSize = sa::getSuffixAutomatonStateSize(static_cast<size_t>(maxSeqLen));

            // Allocate pinned memory for the state
            auto options = at::TensorOptions().dtype(at::kByte).device(at::kCPU).pinned_memory(true);
            at::Tensor hostState = at::zeros({static_cast<int64_t>(stateSize)}, options);

            // Initialize and build the automaton
            void* memory = hostState.data_ptr();
            sa::initAutomaton(memory, static_cast<size_t>(maxSeqLen));

            sa::SuffixAutomaton* sa_ptr = reinterpret_cast<sa::SuffixAutomaton*>(memory);
            sa::buildAutomatonFromTokens(sa_ptr, tokens.data(), static_cast<int>(tokens.size()));

            return hostState;
        },
        nb::arg("tokens"), nb::arg("max_seq_len"),
        "Build a suffix automaton on host from context tokens. Returns pinned CPU tensor.");

    // Copy a host-built suffix automaton state to a GPU slot
    m.def(
        "copy_state_to_slot",
        [](at::Tensor hostState, at::Tensor gpuSlots, int slotIndex, int maxSeqLen)
        {
            TORCH_CHECK(!hostState.is_cuda(), "hostState must be a CPU tensor");
            TORCH_CHECK(gpuSlots.is_cuda(), "gpuSlots must be a CUDA tensor");
            TORCH_CHECK(maxSeqLen > 0, "maxSeqLen must be positive");

            size_t stateSize = sa::getSuffixAutomatonStateSize(static_cast<size_t>(maxSeqLen));
            TORCH_CHECK(hostState.numel() >= static_cast<int64_t>(stateSize),
                "hostState is too small for a SuffixAutomaton state");

            size_t offset = static_cast<size_t>(slotIndex) * stateSize;
            TORCH_CHECK(
                offset + stateSize <= static_cast<size_t>(gpuSlots.numel()), "slotIndex is out of bounds for gpuSlots");

            // Calculate GPU destination address
            void* gpuDst = static_cast<uint8_t*>(gpuSlots.data_ptr()) + offset;
            void* hostSrc = hostState.data_ptr();

            // Relocate pointers from host to GPU addresses
            sa::SuffixAutomaton* sa_ptr = reinterpret_cast<sa::SuffixAutomaton*>(hostSrc);
            sa::relocateAutomaton(sa_ptr, hostSrc, gpuDst);

            // Async copy from host to device
            cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
            cudaMemcpyAsync(gpuDst, hostSrc, stateSize, cudaMemcpyHostToDevice, stream);
        },
        nb::arg("host_state"), nb::arg("gpu_slots"), nb::arg("slot_index"), nb::arg("max_seq_len"),
        "Copy a host-built suffix automaton state to a GPU slot (async).\n\n"
        "WARNING: This function is SINGLE-USE per host_state tensor. It mutates host_state\n"
        "in-place via relocateAutomaton(), which rebases internal pointers from the host\n"
        "address to the GPU destination address. After this call, the host tensor's internal\n"
        "pointer graph is relative to the GPU destination, making it corrupted for any\n"
        "subsequent use (e.g., copying to a different slot). If you need to copy the same\n"
        "automaton state to multiple GPU slots, you must call build_automaton_host() to\n"
        "create a fresh host state for each destination slot.");

    // Clear a suffix automaton state at a GPU slot (reset for reuse)
    m.def(
        "clear_slot",
        [](at::Tensor gpuSlots, int slotIndex, int maxSeqLen)
        {
            TORCH_CHECK(gpuSlots.is_cuda(), "gpuSlots must be a CUDA tensor");
            TORCH_CHECK(maxSeqLen > 0, "maxSeqLen must be positive");

            size_t stateSize = sa::getSuffixAutomatonStateSize(static_cast<size_t>(maxSeqLen));
            size_t offset = static_cast<size_t>(slotIndex) * stateSize;
            TORCH_CHECK(
                offset + stateSize <= static_cast<size_t>(gpuSlots.numel()), "slotIndex is out of bounds for gpuSlots");

            // Zero out the slot
            cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
            cudaMemsetAsync(static_cast<uint8_t*>(gpuSlots.data_ptr()) + offset, 0, stateSize, stream);
        },
        nb::arg("gpu_slots"), nb::arg("slot_index"), nb::arg("max_seq_len"),
        "Clear a suffix automaton state at a GPU slot");
}

} // namespace tensorrt_llm::nanobind::suffix_automaton
