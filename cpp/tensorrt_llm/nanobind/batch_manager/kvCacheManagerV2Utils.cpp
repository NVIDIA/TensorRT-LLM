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
#include "tensorrt_llm/nanobind/common/customCasters.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/torchView.h"
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <torch/extension.h>

namespace tr = tensorrt_llm::runtime;
namespace nb = nanobind;

using SizeType32 = tensorrt_llm::runtime::SizeType32;

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

void checkTensorReadableByGpu(at::Tensor const& tensor, at::Device const& outputDevice, char const* name)
{
    void const* const inputPointer = tensor.const_data_ptr();
    cudaPointerAttributes attributes{};
    cudaError_t const status = cudaPointerGetAttributes(&attributes, inputPointer);
    TLLM_CHECK_WITH_INFO(status == cudaSuccess, "%s pointer %p is not readable by output device %s: %s", name,
        inputPointer, outputDevice.str().c_str(), cudaGetErrorString(status));
    TLLM_CHECK_WITH_INFO(attributes.devicePointer != nullptr, "%s pointer %p is not readable by output device %s", name,
        inputPointer, outputDevice.str().c_str());

    bool originalPointerIsReadable = static_cast<void const*>(attributes.devicePointer) == inputPointer;
    if (!originalPointerIsReadable && attributes.type == cudaMemoryTypeHost)
    {
        int canUseHostPointer = 0;
        cudaError_t const attributeStatus = cudaDeviceGetAttribute(
            &canUseHostPointer, cudaDevAttrCanUseHostPointerForRegisteredMem, outputDevice.index());
        TLLM_CHECK_WITH_INFO(attributeStatus == cudaSuccess,
            "Could not determine whether output device %s can read %s through its host pointer: %s",
            outputDevice.str().c_str(), name, cudaGetErrorString(attributeStatus));
        originalPointerIsReadable = canUseHostPointer != 0;
    }

    TLLM_CHECK_WITH_INFO(originalPointerIsReadable,
        "%s pointer %p is readable only through device alias %p on output device %s, but the kernel receives the "
        "original pointer",
        name, inputPointer, attributes.devicePointer, outputDevice.str().c_str());
}

} // namespace

std::optional<tensorrt_llm::runtime::ITensor::UniquePtr> from_torch(std::optional<at::Tensor> torchPtr)
{
    if (torchPtr)
    {
        return tr::TorchView::of(torchPtr.value());
    }
    return std::nullopt;
}

void KVCacheManagerV2UtilsBindings::initBindings(nb::module_& module)
{
    // Bind DiskAddress struct
    nb::class_<DiskAddress>(module, "DiskAddress")
        .def(nb::init<int, ssize_t>(), nb::arg("fd"), nb::arg("pos"))
        .def_rw("fd", &DiskAddress::fd)
        .def_rw("pos", &DiskAddress::pos);

    // Bind Task template instantiations
    nb::class_<Task<DiskAddress, DiskAddress>>(module, "DiskToDiskTask")
        .def(nb::init<DiskAddress, DiskAddress>(), nb::arg("dst"), nb::arg("src"))
        .def_rw("dst", &Task<DiskAddress, DiskAddress>::dst)
        .def_rw("src", &Task<DiskAddress, DiskAddress>::src);

    nb::class_<Task<MemAddress, DiskAddress>>(module, "DiskToHostTask")
        .def(nb::init<MemAddress, DiskAddress>(), nb::arg("dst"), nb::arg("src"))
        .def_rw("dst", &Task<MemAddress, DiskAddress>::dst)
        .def_rw("src", &Task<MemAddress, DiskAddress>::src);

    nb::class_<Task<DiskAddress, MemAddress>>(module, "HostToDiskTask")
        .def(nb::init<DiskAddress, MemAddress>(), nb::arg("dst"), nb::arg("src"))
        .def_rw("dst", &Task<DiskAddress, MemAddress>::dst)
        .def_rw("src", &Task<DiskAddress, MemAddress>::src);

    nb::class_<Task<MemAddress, MemAddress>>(module, "MemToMemTask")
        .def(nb::init<MemAddress, MemAddress>(), nb::arg("dst"), nb::arg("src"))
        .def_rw("dst", &Task<MemAddress, MemAddress>::dst)
        .def_rw("src", &Task<MemAddress, MemAddress>::src);

    nb::class_<IndexMapper>(module, "IndexMapper")
        .def(nb::init<SizeType32, SizeType32>(), nb::arg("max_batch_size"), nb::arg("max_beam_width"))
        .def("add_new_sequence", &IndexMapper::addNewSequence)
        .def("get_index", &IndexMapper::getIndex)
        .def("remove_sequence", &IndexMapper::removeSequence)
        .def("get_copy_index", &IndexMapper::getCopyIndex)
        .def("gather_k_block_offsets", &IndexMapper::gatherKBlockOffsets, nb::arg("source"), nb::arg("destination"),
            nb::arg("request_ids"), nb::arg("num_blocks"))
        .def("size", &IndexMapper::size)
        .def("num_free_slots", &IndexMapper::numFreeSlots);

    // Bind copy functions
    module.def(
        "copy_disk_to_disk",
        [](std::vector<Task<DiskAddress, DiskAddress>> tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDiskToDisk(std::move(tasks), numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from disk to disk using CUDA host function");

    module.def(
        "copy_disk_to_host",
        [](std::vector<Task<MemAddress, DiskAddress>> tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDiskToHost(std::move(tasks), numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from disk to host using CUDA host function");

    module.def(
        "copy_host_to_disk",
        [](std::vector<Task<DiskAddress, MemAddress>> tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyHostToDisk(std::move(tasks), numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from host to disk using CUDA host function");

    module.def(
        "copy_host_to_host",
        [](std::vector<Task<MemAddress, MemAddress>> tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyHostToHost(std::move(tasks), numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from host to host using CUDA host function");

    module.def(
        "copy_host_to_device",
        [](std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyHostToDevice(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from host to device using CUDA kernels");

    module.def(
        "copy_device_to_host",
        [](std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDeviceToHost(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from device to host using CUDA kernels");

    module.def(
        "copy_device_to_device",
        [](std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDeviceToDevice(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from device to device using CUDA kernels");

    module.def("gather_base_page_rows", &gatherBasePageRows, nb::arg("source"), nb::arg("destination"),
        nb::arg("copy_index"), nb::arg("num_blocks"), nb::call_guard<nb::gil_scoped_release>(),
        "Gather canonical V2 base-page rows into a dense host tensor");

    module.def(
        "copy_base_page_rows_to_device",
        [](at::Tensor input, at::Tensor output, SizeType32 numRows, uintptr_t stream)
        {
            TLLM_CHECK_WITH_INFO(input.device().is_cpu(), "input must be a CPU tensor");
            TLLM_CHECK_WITH_INFO(output.device().is_cuda(), "output must be a CUDA tensor");
            TLLM_CHECK_WITH_INFO(input.scalar_type() == at::kInt && output.scalar_type() == at::kInt,
                "input and output must contain int32 values");
            TLLM_CHECK_WITH_INFO(
                input.is_contiguous() && output.is_contiguous(), "input and output must be contiguous");
            auto _input = from_torch(input);
            auto _output = from_torch(output);
            TLLM_CHECK_WITH_INFO(_input.has_value() && _output.has_value(), "Invalid page-table tensor.");
            copyBasePageRowsToDevice(
                *(_input.value()), *(_output.value()), numRows, reinterpret_cast<CUstream>(stream));
        },
        nb::arg("input"), nb::arg("output"), nb::arg("num_rows"), nb::arg("stream"),
        nb::call_guard<nb::gil_scoped_release>(), "Copy dense V2 base-page rows into a 4-D device table");

    module.def(
        "copy_batch_block_offsets_to_device",
        [](at::Tensor input, at::Tensor output, at::Tensor copyIndex, at::Tensor indexScales, at::Tensor kvOffset,
            uintptr_t stream)
        {
            auto const checkInt32Contiguous = [](at::Tensor const& tensor, char const* name)
            {
                TLLM_CHECK_WITH_INFO(tensor.scalar_type() == at::kInt, "%s must contain int32 values", name);
                TLLM_CHECK_WITH_INFO(tensor.is_contiguous(), "%s must be contiguous", name);
            };
            checkInt32Contiguous(input, "input");
            checkInt32Contiguous(output, "output");
            checkInt32Contiguous(copyIndex, "copy_index");
            checkInt32Contiguous(indexScales, "index_scales");
            checkInt32Contiguous(kvOffset, "kv_offset");

            TLLM_CHECK_WITH_INFO(output.device().is_cuda(), "output must be a CUDA tensor");
            c10::cuda::CUDAGuard const deviceGuard(output.device());

            constexpr int64_t kKvFactor = 2;
            TLLM_CHECK_WITH_INFO(input.dim() == 4 && output.dim() == 4,
                "input and output must be [numPools, rowCapacity, 2, numBlocksPerSeq]");
            TLLM_CHECK_WITH_INFO(copyIndex.dim() == 1 && indexScales.dim() == 1 && kvOffset.dim() == 1,
                "copy_index, index_scales, and kv_offset must be one-dimensional");
            TLLM_CHECK_WITH_INFO(input.size(0) == output.size(0), "input and output pool counts must match");
            TLLM_CHECK_WITH_INFO(
                input.size(2) == kKvFactor && output.size(2) == kKvFactor, "input and output must have K/V factor 2");
            TLLM_CHECK_WITH_INFO(input.size(3) == output.size(3), "input and output block widths must match");
            TLLM_CHECK_WITH_INFO(
                output.size(1) >= copyIndex.size(0), "output must have at least one row per copy_index entry");
            TLLM_CHECK_WITH_INFO(indexScales.size(0) == input.size(0) && kvOffset.size(0) == input.size(0),
                "index_scales and kv_offset must have one entry per pool");

            if (copyIndex.numel() > 0)
            {
                checkTensorReadableByGpu(input, output.device(), "input");
                checkTensorReadableByGpu(copyIndex, output.device(), "copy_index");
                checkTensorReadableByGpu(indexScales, output.device(), "index_scales");
                checkTensorReadableByGpu(kvOffset, output.device(), "kv_offset");
            }

            auto _input = from_torch(input);
            auto _output = from_torch(output);
            auto _copyIndex = from_torch(copyIndex);
            auto _indexScales = from_torch(indexScales);
            auto _kvOffset = from_torch(kvOffset);
            TLLM_CHECK_WITH_INFO(_input.has_value(), "Invalid input tensor.");
            TLLM_CHECK_WITH_INFO(_output.has_value(), "Invalid output tensor.");
            TLLM_CHECK_WITH_INFO(_copyIndex.has_value(), "Invalid copy index tensor.");
            TLLM_CHECK_WITH_INFO(_indexScales.has_value(), "Invalid index scales tensor.");
            TLLM_CHECK_WITH_INFO(_kvOffset.has_value(), "Invalid kv offset tensor.");
            copyBatchBlockOffsetsToDevice(*(_input.value()), *(_output.value()), *(_copyIndex.value()),
                *(_indexScales.value()), *(_kvOffset.value()), reinterpret_cast<CUstream>(stream));
        },
        nb::arg("input"), nb::arg("output"), nb::arg("copy_index"), nb::arg("index_scales"), nb::arg("kv_offset"),
        nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Materialize attention block offsets from V2 base-page indices");
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
