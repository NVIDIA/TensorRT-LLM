/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/utils/pgUtils.h"
#include <pybind11/pybind11.h>
#include <torch/csrc/distributed/c10d/HashStore.hpp>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <typeinfo>

namespace py = pybind11;

namespace pybind11_conduit_v1
{

inline void* get_raw_pointer_ephemeral(PyObject* py_obj, std::type_info const* cpp_type_info, std::string pybind11_abi)
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

template <typename T>
T* get_type_pointer_ephemeral(PyObject* py_obj, std::string pybind11_abi)
{
    void* raw_ptr = get_raw_pointer_ephemeral(py_obj, &typeid(T), pybind11_abi);
    if (raw_ptr == nullptr)
    {
        return nullptr;
    }
    return static_cast<T*>(raw_ptr);
}

} // namespace pybind11_conduit_v1

using namespace tensorrt_llm::pg_broker;

PYBIND11_MODULE(pg_utils_bindings, m)
{
    m.def("init_pg",
        [](py::object world_pg_obj, py::object local_pg_obj)
        {
            auto world_pg = torch::jit::toCustomClass<c10d::ProcessGroup>(world_pg_obj);
            auto local_pg = torch::jit::toCustomClass<c10d::ProcessGroup>(local_pg_obj);
            init_pg(world_pg, local_pg);
        });

    m.def("init_store",
        [](const py::object store_obj, std::string pybind11_abi)
        {
            auto* pStore = pybind11_conduit_v1::get_type_pointer_ephemeral<c10d::Store>(store_obj.ptr(), pybind11_abi);
            if (pStore == nullptr)
            {
                throw py::error_already_set();
            }
            init_store(c10::intrusive_ptr<c10d::Store>::reclaim_copy(pStore));
        });
}
