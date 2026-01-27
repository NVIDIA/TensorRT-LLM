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

#include "kvCacheManagerV2Utils.h"

#include "tensorrt_llm/batch_manager/kvCacheManagerV2Utils.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

void KVCacheManagerV2UtilsBindings::initBindings(py::module_& module)
{
    // Bind DiskAddress struct
    py::class_<DiskAddress>(module, "DiskAddress")
        .def(py::init<int, ssize_t>(), py::arg("fd"), py::arg("pos"))
        .def_readwrite("fd", &DiskAddress::fd)
        .def_readwrite("pos", &DiskAddress::pos);

    // Bind Task template instantiations
    py::class_<Task<DiskAddress, DiskAddress>>(module, "DiskToDiskTask")
        .def(py::init<DiskAddress, DiskAddress>(), py::arg("dst"), py::arg("src"))
        .def_readwrite("dst", &Task<DiskAddress, DiskAddress>::dst)
        .def_readwrite("src", &Task<DiskAddress, DiskAddress>::src);

    py::class_<Task<MemAddress, DiskAddress>>(module, "DiskToHostTask")
        .def(py::init<MemAddress, DiskAddress>(), py::arg("dst"), py::arg("src"))
        .def_readwrite("dst", &Task<MemAddress, DiskAddress>::dst)
        .def_readwrite("src", &Task<MemAddress, DiskAddress>::src);

    py::class_<Task<DiskAddress, MemAddress>>(module, "HostToDiskTask")
        .def(py::init<DiskAddress, MemAddress>(), py::arg("dst"), py::arg("src"))
        .def_readwrite("dst", &Task<DiskAddress, MemAddress>::dst)
        .def_readwrite("src", &Task<DiskAddress, MemAddress>::src);

    py::class_<Task<MemAddress, MemAddress>>(module, "MemToMemTask")
        .def(py::init<MemAddress, MemAddress>(), py::arg("dst"), py::arg("src"))
        .def_readwrite("dst", &Task<MemAddress, MemAddress>::dst)
        .def_readwrite("src", &Task<MemAddress, MemAddress>::src);

    // Bind copy functions
    module.def(
        "copy_disk_to_disk",
        [](std::vector<Task<DiskAddress, DiskAddress>> tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDiskToDisk(std::move(tasks), numBytes, reinterpret_cast<CUstream>(stream)); },
        py::arg("tasks"), py::arg("num_bytes"), py::arg("stream"), py::call_guard<py::gil_scoped_release>(),
        "Copy data from disk to disk using CUDA host function");

    module.def(
        "copy_disk_to_host",
        [](std::vector<Task<MemAddress, DiskAddress>> tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDiskToHost(std::move(tasks), numBytes, reinterpret_cast<CUstream>(stream)); },
        py::arg("tasks"), py::arg("num_bytes"), py::arg("stream"), py::call_guard<py::gil_scoped_release>(),
        "Copy data from disk to host using CUDA host function");

    module.def(
        "copy_host_to_disk",
        [](std::vector<Task<DiskAddress, MemAddress>> tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyHostToDisk(std::move(tasks), numBytes, reinterpret_cast<CUstream>(stream)); },
        py::arg("tasks"), py::arg("num_bytes"), py::arg("stream"), py::call_guard<py::gil_scoped_release>(),
        "Copy data from host to disk using CUDA host function");

    module.def(
        "copy_host_to_host",
        [](std::vector<Task<MemAddress, MemAddress>> tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyHostToHost(std::move(tasks), numBytes, reinterpret_cast<CUstream>(stream)); },
        py::arg("tasks"), py::arg("num_bytes"), py::arg("stream"), py::call_guard<py::gil_scoped_release>(),
        "Copy data from host to host using CUDA host function");

    module.def(
        "copy_host_to_device",
        [](std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyHostToDevice(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        py::arg("tasks"), py::arg("num_bytes"), py::arg("stream"), py::call_guard<py::gil_scoped_release>(),
        "Copy data from host to device using CUDA kernels");

    module.def(
        "copy_device_to_host",
        [](std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDeviceToHost(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        py::arg("tasks"), py::arg("num_bytes"), py::arg("stream"), py::call_guard<py::gil_scoped_release>(),
        "Copy data from device to host using CUDA kernels");

    module.def(
        "copy_device_to_device",
        [](std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDeviceToDevice(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        py::arg("tasks"), py::arg("num_bytes"), py::arg("stream"), py::call_guard<py::gil_scoped_release>(),
        "Copy data from device to device using CUDA kernels");
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
