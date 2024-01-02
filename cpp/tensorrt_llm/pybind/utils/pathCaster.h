/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/detail/descr.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include <filesystem>

namespace PYBIND11_NAMESPACE
{

namespace detail
{

template <typename T>
struct PathCaster
{

private:
    static PyObject* unicode_from_fs_native(const std::string& w)
    {
        return PyUnicode_DecodeFSDefaultAndSize(w.c_str(), ssize_t(w.size()));
    }

    static PyObject* unicode_from_fs_native(const std::wstring& w)
    {
        return PyUnicode_FromWideChar(w.c_str(), ssize_t(w.size()));
    }

public:
    static handle cast(const T& path, return_value_policy, handle)
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
