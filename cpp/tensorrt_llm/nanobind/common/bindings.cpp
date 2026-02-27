/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "bindings.h"
#include "tensorrt_llm/common/hashUtils.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

namespace nb = nanobind;
namespace tc = tensorrt_llm::common;

namespace tensorrt_llm::nanobind::common
{

void initBindings(nb::module_& m)
{
    // Bind Hasher class for SHA-256 hashing
    nb::class_<tc::Hasher>(m, "Hasher")
        .def(nb::init<>(), "Create hasher with seed = 0")
        .def(nb::init<uint64_t>(), nb::arg("seed"), "Create hasher with explicit seed")
        .def(nb::init<std::optional<uint64_t>>(), nb::arg("seed"), "Create hasher with optional seed")
        .def(
            "__init__",
            [](tc::Hasher* self, nb::bytes data)
            {
                new (self) tc::Hasher();
                self->update(data.data(), data.size());
            },
            nb::arg("seed"), "Create hasher with bytes as initial data")
        .def(
            "update", [](tc::Hasher& self, uint64_t value) -> tc::Hasher& { return self.update(value); },
            nb::arg("value"), nb::rv_policy::reference_internal, "Update hash with a 64-bit integer")
        .def(
            "update",
            [](tc::Hasher& self, nb::bytes data) -> tc::Hasher& { return self.update(data.data(), data.size()); },
            nb::arg("data"), nb::rv_policy::reference_internal, "Update hash with bytes")
        .def(
            "update",
            [](tc::Hasher& self, nb::list data) -> tc::Hasher&
            {
                // Handle generic sequence (list) of int or bytes in C++ - avoids Python iteration overhead
                for (auto item : data)
                {
                    if (nb::isinstance<nb::int_>(item))
                    {
                        self.update(nb::cast<uint64_t>(item));
                    }
                    else if (nb::isinstance<nb::bytes>(item))
                    {
                        auto bytes = nb::cast<nb::bytes>(item);
                        self.update(bytes.data(), bytes.size());
                    }
                    else
                    {
                        throw std::runtime_error("Sequence items must be int or bytes");
                    }
                }
                return self;
            },
            nb::arg("data"), nb::rv_policy::reference_internal, "Update hash with sequence of int or bytes")
        .def_prop_ro(
            "digest",
            [](tc::Hasher const& self) -> nb::bytes
            {
                auto digestBytes = self.digestBytes();
                return nb::bytes(reinterpret_cast<char const*>(digestBytes.data()), digestBytes.size());
            },
            "Get the final hash as 32 bytes (SHA-256)")
        .def("digest_int", &tc::Hasher::digest, "Get the final hash as a 64-bit integer");
}

} // namespace tensorrt_llm::nanobind::common
