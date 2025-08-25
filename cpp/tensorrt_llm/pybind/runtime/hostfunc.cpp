/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "hostfunc.h"

#include <cuda_runtime.h>
#include <memory>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace tensorrt_llm::pybind::runtime
{

struct HostFuncUserData
{
    py::function pyHostFunc;
    py::tuple pyArgs;
    py::dict pyKwargs;

    HostFuncUserData(py::function func, py::tuple args, py::dict kwargs)
        : pyHostFunc(std::move(func))
        , pyArgs(std::move(args))
        , pyKwargs(std::move(kwargs))
    {
    }
};

static void cudaHostFuncTrampoline(void* userData)
{
    auto* hostFuncUserData = static_cast<HostFuncUserData*>(userData);
    // Acquire the GIL since we are calling Python code from a CUDA stream.
    py::gil_scoped_acquire gil;
    try
    {
        hostFuncUserData->pyHostFunc(*hostFuncUserData->pyArgs, **hostFuncUserData->pyKwargs);
    }
    catch (py::error_already_set& e)
    {
        e.restore();
        PyErr_Print();
    }
}

uintptr_t launchHostFunc(uintptr_t streamPtr, py::function pyHostFunc, py::args pyArgs, py::kwargs pyKwargs)
{
    auto const stream = reinterpret_cast<cudaStream_t>(streamPtr);

    auto hostFuncUserData = std::make_unique<HostFuncUserData>(pyHostFunc, py::tuple(pyArgs), py::dict(pyKwargs));

    cudaError_t err = cudaLaunchHostFunc(stream, cudaHostFuncTrampoline, hostFuncUserData.get());
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to launch host function.");
    }

    // Take over the ownership of the user data from the unique_ptr;
    // the user data lifetime is managed manually from Python via freeHostFuncUserData.
    return reinterpret_cast<uintptr_t>(hostFuncUserData.release());
}

void freeHostFuncUserData(uintptr_t userDataPtr)
{
    // Create a unique_ptr to take over the ownership of the user data;
    // the user data is released when the unique_ptr is destroyed.
    auto hostFuncUserData = std::unique_ptr<HostFuncUserData>(reinterpret_cast<HostFuncUserData*>(userDataPtr));

    // Acquire the GIL to safely release the Python objects.
    py::gil_scoped_acquire gil;
}

void initHostFuncBindings(pybind11::module_& m)
{
    m.def("launch_hostfunc", &launchHostFunc, "Launch a Python host function to a CUDA stream", py::arg("stream_ptr"),
        py::arg("py_hostfunc"));
    m.def("free_hostfunc_user_data", &freeHostFuncUserData, "Free the user data for the Python host function",
        py::arg("user_data_ptr"));
}
} // namespace tensorrt_llm::pybind::runtime
