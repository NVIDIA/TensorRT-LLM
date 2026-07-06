/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorrt_llm/nanobind/common/customCasters.h"
#include "tensorrt_llm/thop/compactPseudoKvAttentionOp.h"

#include <nanobind/nanobind.h>

#include <optional>

namespace nb = nanobind;

namespace tensorrt_llm::nanobind::thop
{
namespace
{

template <typename T>
T requiredKwarg(nb::kwargs const& kwargs, char const* name)
{
    TORCH_CHECK(kwargs.contains(name), name, " is required.");
    auto value = kwargs[name];
    TORCH_CHECK(!value.is_none(), name, " is required.");
    return nb::cast<T>(value);
}

template <typename T>
std::optional<T> optionalKwarg(nb::kwargs const& kwargs, char const* name)
{
    if (!kwargs.contains(name))
    {
        return std::nullopt;
    }
    auto value = kwargs[name];
    if (value.is_none())
    {
        return std::nullopt;
    }
    return nb::cast<T>(value);
}

void compactPseudoKvAttentionAbiBinding(nb::kwargs kwargs)
{
    auto q = requiredKwarg<torch::Tensor>(kwargs, "q");
    auto output = requiredKwarg<torch::Tensor>(kwargs, "output");
    auto compactPseudokvKey = optionalKwarg<torch::Tensor>(kwargs, "compact_pseudokv_key");
    auto compactPseudokvValue = optionalKwarg<torch::Tensor>(kwargs, "compact_pseudokv_value");
    auto compactPseudokvPositions = optionalKwarg<torch::Tensor>(kwargs, "compact_pseudokv_positions");
    auto compactPseudokvCausalMask = optionalKwarg<torch::Tensor>(kwargs, "compact_pseudokv_causal_mask");
    auto compactPseudokvSourceSeqLen = optionalKwarg<int64_t>(kwargs, "compact_pseudokv_source_seq_len");

    torch_ext::runCompactPseudoKvAttention(q, output, compactPseudokvKey, compactPseudokvValue,
        compactPseudokvPositions, compactPseudokvCausalMask, compactPseudokvSourceSeqLen);
}

void compactPseudoKvAttentionTypedAbiBinding(torch::Tensor const& q, torch::Tensor output,
    torch::Tensor const& compactPseudokvKey, torch::Tensor const& compactPseudokvValue,
    torch::Tensor const& compactPseudokvPositions, torch::Tensor const& compactPseudokvCausalMask,
    int64_t compactPseudokvSourceSeqLen)
{
    torch_ext::runCompactPseudoKvAttention(q, output, compactPseudokvKey, compactPseudokvValue,
        compactPseudokvPositions, compactPseudokvCausalMask, compactPseudokvSourceSeqLen);
}

} // namespace

void initCompactPseudoKvBindings(nb::module_& m)
{
    m.def("compact_pseudokv_attention", &torch_ext::compactPseudoKvAttention, nb::arg("q"), nb::arg("output"),
        nb::arg("compact_pseudokv_key"), nb::arg("compact_pseudokv_value"), nb::arg("compact_pseudokv_positions"),
        nb::arg("compact_pseudokv_causal_mask"), nb::arg("compact_pseudokv_source_seq_len"),
        "Compact pseudo-KV attention operation.", nb::call_guard<nb::gil_scoped_release>());
    m.def("attention", &compactPseudoKvAttentionAbiBinding,
        "attention(..., compact_pseudokv_key=None, compact_pseudokv_value=None, "
        "compact_pseudokv_positions=None, compact_pseudokv_causal_mask=None, "
        "compact_pseudokv_source_seq_len=None)");
    m.def("attention", &compactPseudoKvAttentionTypedAbiBinding, nb::arg("q"), nb::arg("output"),
        nb::arg("compact_pseudokv_key"), nb::arg("compact_pseudokv_value"), nb::arg("compact_pseudokv_positions"),
        nb::arg("compact_pseudokv_causal_mask"), nb::arg("compact_pseudokv_source_seq_len"),
        "typed_compact_pseudokv_attention_abi");
}

} // namespace tensorrt_llm::nanobind::thop
