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
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "tensorrt_llm/kernels/speculativeDecoding/suffixAutomaton/suffixAutomatonKernels.h"

namespace py = pybind11;
namespace sa = tensorrt_llm::kernels::speculative_decoding::suffix_automaton;

namespace tensorrt_llm::pybind::suffix_automaton
{

void initBindings(pybind11::module_& m)
{
    // Export configuration constants
    m.attr("MAX_SEQUENCE_LENGTH") = py::int_(sa::getSuffixAutomatonMaxSeqLen());
    m.attr("MAX_SLOTS") = py::int_(sa::getSuffixAutomatonMaxSlots());
    m.attr("STATE_SIZE_BYTES") = py::int_(sa::getSuffixAutomatonStateSize());

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
            params.acceptedTokensIn = acceptedTokensIn.data_ptr<int const>();
            params.acceptedLensIn = acceptedLensIn.data_ptr<int const>();

            cudaStream_t stream = at::cuda::getCurrentCUDAStream();
            sa::invokeSuffixAutomatonExtend(params, stream);
        },
        py::arg("batch_size"), py::arg("draft_length"), py::arg("slots"), py::arg("batch_indices"),
        py::arg("match_len_out"), py::arg("draft_tokens_out"), py::arg("accepted_tokens_in"),
        py::arg("accepted_lens_in"), "Invoke suffix automaton extend CUDA kernel",
        py::call_guard<py::gil_scoped_release>());

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
        py::arg("max_slots"), "Allocate GPU workspace for suffix automaton states");

    // Helper function to get workspace size
    m.def(
        "get_workspace_size",
        [](int maxSlots)
        {
            size_t stateSize = sa::getSuffixAutomatonStateSize();
            return static_cast<size_t>(maxSlots) * stateSize;
        },
        py::arg("max_slots"), "Get required workspace size in bytes for suffix automaton states");
}

} // namespace tensorrt_llm::pybind::suffix_automaton
