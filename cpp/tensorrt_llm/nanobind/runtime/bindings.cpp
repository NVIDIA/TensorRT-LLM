/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "hostfunc.h"
#include "moeBindings.h"
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/kernels/communicationKernels/allReduceWorkspace.h"
#include "tensorrt_llm/kernels/communicationKernels/customLowPrecisionAllReduceKernels.h"
#include "tensorrt_llm/kernels/customAllReduceKernels.h"
#include "tensorrt_llm/kernels/delayStream.h"
#include "tensorrt_llm/kernels/globalTimerKernel.h"
#include "tensorrt_llm/nanobind/common/customCasters.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/ipcUtils.h"
#include "tensorrt_llm/runtime/locality_domain/locality_domain_utils.h"
#include "tensorrt_llm/runtime/loraCache.h"
#include "tensorrt_llm/runtime/mcastGPUBuffer.h"
#include "tensorrt_llm/runtime/speculativeDecodingMode.h"
#include "tensorrt_llm/runtime/torchView.h"
#include "tensorrt_llm/runtime/virtualMemory.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <nanobind/stl/vector.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/trampoline.h>
#include <torch/extension.h>
namespace tr = tensorrt_llm::runtime;
namespace te = tensorrt_llm::executor;

namespace tensorrt_llm::nanobind::runtime
{

void initBindings(nb::module_& m)
{

    nb::class_<tr::LoraCache::TaskLayerModuleConfig>(m, "TaskLayerModuleConfig")
        .def(nb::init<>())
        .def_rw("page_id", &tr::LoraCache::TaskLayerModuleConfig::pageId)
        .def_rw("slot_idx", &tr::LoraCache::TaskLayerModuleConfig::slotIdx)
        .def_rw("in_size", &tr::LoraCache::TaskLayerModuleConfig::inSize)
        .def_rw("out_size", &tr::LoraCache::TaskLayerModuleConfig::outSize)
        .def_rw("module_id", &tr::LoraCache::TaskLayerModuleConfig::moduleId)
        .def_rw("layer_id", &tr::LoraCache::TaskLayerModuleConfig::layerId)
        .def_rw("adapter_size", &tr::LoraCache::TaskLayerModuleConfig::adapterSize)
        .def_rw("num_slots", &tr::LoraCache::TaskLayerModuleConfig::numSlots)
        .def_rw("weights_in_pointer", &tr::LoraCache::TaskLayerModuleConfig::weightsInPointer)
        .def_rw("weights_out_pointer", &tr::LoraCache::TaskLayerModuleConfig::weightsOutPointer)
        .def_rw("scaling_vec_pointer", &tr::LoraCache::TaskLayerModuleConfig::scalingVecPointer)
        .def(nb::self == nb::self);

    nb::class_<tr::CudaVirtualMemoryManager>(m, "CudaVirtualMemoryManager")
        .def("release_with_tag", &tr::CudaVirtualMemoryManager::releaseWithTag, nb::arg("tag"),
            nb::call_guard<nb::gil_scoped_release>())
        .def("materialize_with_tag", &tr::CudaVirtualMemoryManager::materializeWithTag, nb::arg("tag"),
            nb::call_guard<nb::gil_scoped_release>());

    nb::class_<tr::CudaEvent>(m, "CudaEvent")
        .def(nb::init<unsigned int>(), nb::arg("flags") = cudaEventDisableTiming,
            nb::call_guard<nb::gil_scoped_release>())
        .def("synchronize", &tr::CudaEvent::synchronize, nb::call_guard<nb::gil_scoped_release>());

    m.def(
        "lamport_initialize_all",
        [](intptr_t buffer_0, intptr_t buffer_1, intptr_t buffer_2, size_t size)
        {
            tr::lamportInitializeAll(reinterpret_cast<void*>(buffer_0), reinterpret_cast<void*>(buffer_1),
                reinterpret_cast<void*>(buffer_2), size);
        },
        "Lamport initialize all buffers", nb::call_guard<nb::gil_scoped_release>());
    m.def(
        "lamport_initialize",
        [](intptr_t buffer, size_t size)
        { tensorrt_llm::kernels::ar_fusion::lamport_initialize(reinterpret_cast<void*>(buffer), size, 0); },
        "Lmaport initialize buffer", nb::call_guard<nb::gil_scoped_release>());
    m.def(
        "delay_kernel",
        [](int64_t delay_micro_secs, nb::object py_stream)
        {
            // Get the raw stream handle from PyTorch stream object
            auto stream_ptr = nb::cast<int64_t>(py_stream.attr("cuda_stream"));
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
            nb::gil_scoped_release release;
            tensorrt_llm::kernels::invokeDelayStreamKernel(delay_micro_secs, stream);
        },
        "Delay kernel launch on the default stream");
    m.def(
        "record_global_timer",
        [](int64_t data_ptr, nb::object py_stream)
        {
            auto* ptr = reinterpret_cast<uint64_t*>(data_ptr);
            auto stream_ptr = nb::cast<int64_t>(py_stream.attr("cuda_stream"));
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
            nb::gil_scoped_release release;
            tensorrt_llm::kernels::invokeReadGlobalTimer(ptr, stream);
        },
        "Record GPU global timer value to device memory");
    m.def(
        "max_workspace_size_lowprecision",
        [](int32_t tp_size) { return tensorrt_llm::kernels::max_workspace_size_lowprecision(tp_size); },
        "Calculate the maximum workspace size needed for low precision all-reduce operations",
        nb::call_guard<nb::gil_scoped_release>());

    nb::enum_<tr::CudaVirtualMemoryAllocator::RestoreMode>(m, "CudaVirtualMemoryAllocatorRestoreMode")
        .value("NONE", tr::CudaVirtualMemoryAllocator::RestoreMode::NONE)
        .value("CPU", tr::CudaVirtualMemoryAllocator::RestoreMode::CPU)
        .value("PINNED", tr::CudaVirtualMemoryAllocator::RestoreMode::PINNED)
        .value("MEMSET", tr::CudaVirtualMemoryAllocator::RestoreMode::MEMSET);

    m.def("get_virtual_memory_manager", &tr::getVirtualMemoryManager, "Get the virtual memory manager",
        nb::rv_policy::reference);

    m.def(
        "push_virtual_memory_allocator",
        [](std::string const& tag, tr::CudaVirtualMemoryAllocator::RestoreMode mode, uintptr_t stream)
        {
            static_assert(sizeof(uintptr_t) == sizeof(cudaStream_t));
            tr::pushVirtualMemoryAllocator(tag, mode,
                std::make_shared<tr::CudaStream>(
                    reinterpret_cast<cudaStream_t>(stream), tensorrt_llm::common::getDevice(), false));
        },
        "Push a virtual memory allocator onto the allocator stack.", nb::call_guard<nb::gil_scoped_release>());

    m.def("pop_virtual_memory_allocator", &tr::popVirtualMemoryAllocator,
        "Pop the top virtual memory allocator from the allocator stack", nb::call_guard<nb::gil_scoped_release>());

    nb::class_<tensorrt_llm::runtime::McastGPUBuffer>(m, "McastGPUBuffer")
        .def(nb::init<size_t, uint32_t, uint32_t, uint32_t, bool, int64_t>(), nb::arg("buf_size"),
            nb::arg("group_size"), nb::arg("group_rank"), nb::arg("device_idx"), nb::arg("mn_nvlink"),
            nb::arg("mpi_comm_fortran_handle"), nb::call_guard<nb::gil_scoped_release>())
        .def("get_uc_buffer", &tensorrt_llm::runtime::McastGPUBuffer::getUCBuffer,
            nb::call_guard<nb::gil_scoped_release>())
        .def("get_mc_buffer", &tensorrt_llm::runtime::McastGPUBuffer::getMCBuffer,
            nb::call_guard<nb::gil_scoped_release>())
        .def("checkpoint_prepare", &tensorrt_llm::runtime::McastGPUBuffer::checkpointPrepare,
            "Internal, experimental hook; the caller must establish engine-wide quiescence before invoking it.",
            nb::call_guard<nb::gil_scoped_release>())
        .def("checkpoint_restore", &tensorrt_llm::runtime::McastGPUBuffer::checkpointRestore,
            nb::arg("mpi_comm_fortran_handle"),
            "Internal, experimental hook; the restored communicator must have the original ordered membership and "
            "the engine must remain quiescent. A successful restore retains an owned communicator duplicate.",
            nb::call_guard<nb::gil_scoped_release>())
        .def("checkpoint_restore_complete", &tensorrt_llm::runtime::McastGPUBuffer::checkpointRestoreComplete,
            nb::arg("local_protocol_reset_succeeded"),
            "Collectively publish or abort a pending internal MNNVL restore after protocol reset.",
            nb::call_guard<nb::gil_scoped_release>())
        .def("is_mapped", &tensorrt_llm::runtime::McastGPUBuffer::isMapped);

    nb::enum_<tensorrt_llm::kernels::AllReduceFusionOp>(m, "AllReduceFusionOp")
        .value("NONE", tensorrt_llm::kernels::AllReduceFusionOp::NONE)
        .value("RESIDUAL_RMS_NORM", tensorrt_llm::kernels::AllReduceFusionOp::RESIDUAL_RMS_NORM)
        .value("LAST_PROCESS_FOR_UB", tensorrt_llm::kernels::AllReduceFusionOp::LAST_PROCESS_FOR_UB)
        .value("RESIDUAL_RMS_PREPOST_NORM", tensorrt_llm::kernels::AllReduceFusionOp::RESIDUAL_RMS_PREPOST_NORM)
        .value("RESIDUAL_RMS_NORM_QUANT_FP8", tensorrt_llm::kernels::AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_FP8)
        .value("RESIDUAL_RMS_NORM_QUANT_NVFP4", tensorrt_llm::kernels::AllReduceFusionOp::RESIDUAL_RMS_NORM_QUANT_NVFP4)
        .value("RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4",
            tensorrt_llm::kernels::AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4)
        .value("RESIDUAL_RMS_NORM_OUT_QUANT_FP8",
            tensorrt_llm::kernels::AllReduceFusionOp::RESIDUAL_RMS_NORM_OUT_QUANT_FP8);

    nb::enum_<tensorrt_llm::kernels::AllReduceStrategyType>(m, "AllReduceStrategy")
        .value("NCCL", tensorrt_llm::kernels::AllReduceStrategyType::NCCL)
        .value("MIN_LATENCY", tensorrt_llm::kernels::AllReduceStrategyType::MIN_LATENCY)
        .value("AUTO", tensorrt_llm::kernels::AllReduceStrategyType::AUTO)
        .value("UB", tensorrt_llm::kernels::AllReduceStrategyType::UB)
        .value("ONESHOT", tensorrt_llm::kernels::AllReduceStrategyType::ONESHOT)
        .value("TWOSHOT", tensorrt_llm::kernels::AllReduceStrategyType::TWOSHOT);

    // LOCALITY_DOMAIN Localization Handle bindings
    nb::class_<tensorrt_llm::locality_domain::LocalizationHandle>(m, "LocalizationHandle")
        .def(nb::init<>(), nb::call_guard<nb::gil_scoped_release>())
        .def("supports_localization", &tensorrt_llm::locality_domain::LocalizationHandle::supportsLocalization,
            nb::call_guard<nb::gil_scoped_release>())
        .def("supports_memory_localization",
            &tensorrt_llm::locality_domain::LocalizationHandle::supportsMemoryLocalization,
            nb::call_guard<nb::gil_scoped_release>())
        .def("supports_compute_localization",
            &tensorrt_llm::locality_domain::LocalizationHandle::supportsComputeLocalization,
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "locality_domain_malloc",
            [](tensorrt_llm::locality_domain::LocalizationHandle& self, size_t size, int localityDomainId) -> uintptr_t
            {
                void* ptr = nullptr;
                self.localityDomainMalloc(&ptr, size, localityDomainId);
                return reinterpret_cast<uintptr_t>(ptr);
            },
            nb::arg("size"), nb::arg("locality_domain_id"),
            "Allocate LOCALITY_DOMAIN localized memory and return pointer as integer address",
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "locality_domain_free",
            [](tensorrt_llm::locality_domain::LocalizationHandle& self, uintptr_t ptr)
            { self.localityDomainFree(reinterpret_cast<void*>(ptr)); },
            nb::arg("ptr"), "Free LOCALITY_DOMAIN localized memory from integer address",
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "create_localized_allocation_handle",
            [](tensorrt_llm::locality_domain::LocalizationHandle& self, size_t size, int localityDomainId,
                unsigned int requestedHandleTypes, bool gpuDirectRDMACapable,
                std::optional<unsigned int> usage) -> uintptr_t
            {
                CUmemGenericAllocationHandle const handle = usage.has_value()
                    ? self.createLocalizedAllocationHandle(
                        size, localityDomainId, requestedHandleTypes, gpuDirectRDMACapable, *usage)
                    : self.createLocalizedAllocationHandle(
                        size, localityDomainId, requestedHandleTypes, gpuDirectRDMACapable);
                return static_cast<uintptr_t>(handle);
            },
            nb::arg("size"), nb::arg("locality_domain_id"), nb::arg("requested_handle_types"),
            nb::arg("gpu_direct_rdma_capable"), nb::arg("usage") = nb::none(),
            "Create LOCALITY_DOMAIN localized generic allocation handle and return it as an integer",
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "try_create_localized_allocation_handle",
            [](tensorrt_llm::locality_domain::LocalizationHandle& self, size_t size, int localityDomainId,
                unsigned int requestedHandleTypes, bool gpuDirectRDMACapable,
                unsigned int usage) -> std::pair<int, uintptr_t>
            {
                CUmemGenericAllocationHandle handle{};
                CUresult const result = self.tryCreateLocalizedAllocationHandle(
                    &handle, size, localityDomainId, requestedHandleTypes, gpuDirectRDMACapable, usage);
                return {static_cast<int>(result), static_cast<uintptr_t>(handle)};
            },
            nb::arg("size"), nb::arg("locality_domain_id"), nb::arg("requested_handle_types"),
            nb::arg("gpu_direct_rdma_capable"), nb::arg("usage"),
            "Try to create a localized allocation and return (CUresult, handle)",
            nb::call_guard<nb::gil_scoped_release>())
        .def("get_localized_allocation_granularity",
            &tensorrt_llm::locality_domain::LocalizationHandle::getLocalizedAllocationGranularity,
            nb::arg("locality_domain_id"), nb::arg("requested_handle_types"), nb::arg("gpu_direct_rdma_capable"),
            nb::arg("usage"), "Get minimum allocation granularity for a localized VMM allocation",
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "try_get_localized_allocation_granularity",
            [](tensorrt_llm::locality_domain::LocalizationHandle& self, int localityDomainId,
                unsigned int requestedHandleTypes, bool gpuDirectRDMACapable,
                unsigned int usage) -> std::pair<int, size_t>
            {
                size_t granularity{};
                CUresult const result = self.tryGetLocalizedAllocationGranularity(
                    &granularity, localityDomainId, requestedHandleTypes, gpuDirectRDMACapable, usage);
                return {static_cast<int>(result), granularity};
            },
            nb::arg("locality_domain_id"), nb::arg("requested_handle_types"), nb::arg("gpu_direct_rdma_capable"),
            nb::arg("usage"), "Try to get localized VMM granularity and return (CUresult, granularity)",
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "create_localized_stream",
            [](tensorrt_llm::locality_domain::LocalizationHandle& self, int localityDomainId) -> uintptr_t
            {
                CUstream stream = self.createLocalizedStream(localityDomainId);
                return reinterpret_cast<uintptr_t>(stream);
            },
            nb::arg("locality_domain_id"),
            "Get a process-lifetime cached LOCALITY_DOMAIN localized stream as an integer address; callers must not "
            "destroy it",
            nb::call_guard<nb::gil_scoped_release>())
        .def("get_locality_domain_compute_sm_counts",
            &tensorrt_llm::locality_domain::LocalizationHandle::getLocalityDomainComputeSmCounts,
            nb::arg("locality_domain_id"),
            "Get (localized partition SM count, full-device SM count), or (0, 0) when unavailable",
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "get_reserved_remainder_stream",
            [](tensorrt_llm::locality_domain::LocalizationHandle& self) -> uintptr_t
            { return reinterpret_cast<uintptr_t>(self.getReservedRemainderStream()); },
            "Get the borrowed process-lifetime remainder Green Context stream, or 0 when unavailable",
            nb::call_guard<nb::gil_scoped_release>());

    m.def("device_supports_locality_domain", &tensorrt_llm::locality_domain::deviceSupportsLocalization,
        nb::arg("device"),
        "Return whether the device exposes public locality domains. Performs a driver attribute query only: it "
        "creates no CUDA context and does not partition the device, so it is safe to call before selecting a device.",
        nb::call_guard<nb::gil_scoped_release>());

    // Initialize MoeLoadBalancer bindings
    initMoeBindings(m);
    // Initialize HostFunc bindings
    initHostFuncBindings(m);
}

void initBindingsEarly(nb::module_& m)
{
    nb::class_<tr::BufferManager>(m, "BufferManager")
        .def(nb::init<tr::BufferManager::CudaStreamPtr, bool>(), nb::arg("stream"), nb::arg("trim_pool") = false,
            nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("stream", &tr::BufferManager::getStream);
}
} // namespace tensorrt_llm::nanobind::runtime
