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

#pragma once

#include "c10/util/intrusive_ptr.h"
#include <Python.h>

namespace tensorrt_llm::common
{

// Adapted from pybind11's example implementation:
// https://github.com/pybind/pybind11/blob/master/include/pybind11/conduit/pybind11_conduit_v1.h
// Copyright (c) 2024 The pybind Community.

inline void* get_raw_pointer_ephemeral(
    PyObject* py_obj, std::type_info const* cpp_type_info, std::string const& pybind11_abi)
{
    PyObject* cpp_type_info_capsule = PyCapsule_New(
        const_cast<void*>(static_cast<void const*>(cpp_type_info)), typeid(std::type_info).name(), nullptr);
    if (cpp_type_info_capsule == nullptr)
    {
        return nullptr;
    }
    PyObject* cpp_conduit = PyObject_CallMethod(
        py_obj, "_pybind11_conduit_v1_", "yOy", pybind11_abi.c_str(), cpp_type_info_capsule, "raw_pointer_ephemeral");
    Py_DECREF(cpp_type_info_capsule);
    if (cpp_conduit == nullptr)
    {
        return nullptr;
    }
    void* raw_ptr = PyCapsule_GetPointer(cpp_conduit, cpp_type_info->name());
    Py_DECREF(cpp_conduit);
    if (PyErr_Occurred())
    {
        return nullptr;
    }
    return raw_ptr;
}

template <typename T, typename E>
T* get_type_pointer_ephemeral(PyObject* py_obj, std::string pybind11_abi)
{
    void* raw_ptr = get_raw_pointer_ephemeral(py_obj, &typeid(T), pybind11_abi);
    if (raw_ptr == nullptr)
    {
        throw E();
    }
    return static_cast<T*>(raw_ptr);
}

template <typename T, typename E>
c10::intrusive_ptr<T> get_intrusive_ptr(PyObject* py_obj, std::string pybind11_abi)
{
    auto* const p = get_type_pointer_ephemeral<T, E>(py_obj, pybind11_abi);
    return c10::intrusive_ptr<T>::reclaim_copy(p);
}

} // namespace tensorrt_llm::common
