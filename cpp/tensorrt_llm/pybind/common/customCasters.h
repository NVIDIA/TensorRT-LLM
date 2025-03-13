/*
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#pragma once

#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/detail/descr.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/common/optionalRef.h"
#include "tensorrt_llm/runtime/request.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include <filesystem>
#include <pybind11/stl_bind.h>

// Pybind requires to have a central include in order for type casters to work.
// Opaque bindings add a type caster, so they have the same requirement.
// See the warning in https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html

// Opaque bindings
PYBIND11_MAKE_OPAQUE(tensorrt_llm::batch_manager::ReqIdsSet)
PYBIND11_MAKE_OPAQUE(tensorrt_llm::batch_manager::RequestVector)
PYBIND11_MAKE_OPAQUE(std::vector<tensorrt_llm::batch_manager::SlotDecoderBuffers>)
PYBIND11_MAKE_OPAQUE(std::vector<tensorrt_llm::runtime::decoder_batch::Request>)
PYBIND11_MAKE_OPAQUE(std::vector<tensorrt_llm::runtime::SamplingConfig>)

// Custom casters
namespace PYBIND11_NAMESPACE
{

namespace detail
{

template <typename T>
struct type_caster<tensorrt_llm::common::OptionalRef<T>>
{
    using value_conv = make_caster<T>;

    PYBIND11_TYPE_CASTER(tensorrt_llm::common::OptionalRef<T>, value_conv::name);

    bool load(handle src, bool convert)
    {
        if (src.is_none())
        {
            // If the Python object is None, create an empty OptionalRef
            value = tensorrt_llm::common::OptionalRef<T>();
            return true;
        }

        value_conv conv;
        if (!conv.load(src, convert))
            return false;

        // Create an OptionalRef with a reference to the converted value
        value = tensorrt_llm::common::OptionalRef<T>(conv);
        return true;
    }

    static handle cast(tensorrt_llm::common::OptionalRef<T> const& src, return_value_policy policy, handle parent)
    {
        if (!src.has_value())
            return none().release();

        return value_conv::cast(*src, policy, parent);
    }
};

template <typename T>
struct PathCaster
{

private:
    static PyObject* unicode_from_fs_native(std::string const& w)
    {
        return PyUnicode_DecodeFSDefaultAndSize(w.c_str(), ssize_t(w.size()));
    }

    static PyObject* unicode_from_fs_native(std::wstring const& w)
    {
        return PyUnicode_FromWideChar(w.c_str(), ssize_t(w.size()));
    }

public:
    static handle cast(T const& path, return_value_policy, handle)
    {
        if (auto py_str = unicode_from_fs_native(path.native()))
        {
            return module_::import("pathlib").attr("Path")(reinterpret_steal<object>(py_str)).release();
        }
        return nullptr;
    }

    bool load(handle handle, bool)
    {
        PyObject* native = nullptr;
        if constexpr (std::is_same_v<typename T::value_type, char>)
        {
            if (PyUnicode_FSConverter(handle.ptr(), &native) != 0)
            {
                if (auto* c_str = PyBytes_AsString(native))
                {
                    // AsString returns a pointer to the internal buffer, which
                    // must not be free'd.
                    value = c_str;
                }
            }
        }
        else if constexpr (std::is_same_v<typename T::value_type, wchar_t>)
        {
            if (PyUnicode_FSDecoder(handle.ptr(), &native) != 0)
            {
                if (auto* c_str = PyUnicode_AsWideCharString(native, nullptr))
                {
                    // AsWideCharString returns a new string that must be free'd.
                    value = c_str; // Copies the string.
                    PyMem_Free(c_str);
                }
            }
        }
        Py_XDECREF(native);
        if (PyErr_Occurred())
        {
            PyErr_Clear();
            return false;
        }
        return true;
    }

    PYBIND11_TYPE_CASTER(T, const_name("os.PathLike"));
};

template <>
struct type_caster<std::filesystem::path> : public PathCaster<std::filesystem::path>
{
};

} // namespace detail
} // namespace PYBIND11_NAMESPACE
