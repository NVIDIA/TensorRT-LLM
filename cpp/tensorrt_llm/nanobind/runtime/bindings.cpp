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
#include "tensorrt_llm/runtime/lookaheadBuffers.h"
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

    nb::class_<tr::LookaheadDecodingBuffers>(m, "LookaheadDecodingBuffers")
        .def(nb::init<tr::SizeType32, tr::SizeType32, tr::BufferManager const&>(), nb::arg("max_num_sequences"),
            nb::arg("max_tokens_per_step"), nb::arg("buffer_manager"), nb::call_guard<nb::gil_scoped_release>())
        .def_rw("generation_lengths", &tr::LookaheadDecodingBuffers::generationLengths)
        .def_rw("position_offsets", &tr::LookaheadDecodingBuffers::positionOffsets)
        .def_rw("packed_masks", &tr::LookaheadDecodingBuffers::packedMasks)
        .def_rw("position_ids", &tr::LookaheadDecodingBuffers::positionIds);

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
            nb::call_guard<nb::gil_scoped_release>());

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

    nb::class_<tr::SpeculativeDecodingMode>(m, "SpeculativeDecodingMode")
        .def(nb::init<tr::SpeculativeDecodingMode::UnderlyingType>(), nb::arg("state"))
        .def_static("NoneType", &tr::SpeculativeDecodingMode::None)
        .def_static("DraftTokensExternal", &tr::SpeculativeDecodingMode::DraftTokensExternal)
        .def_static("Medusa", &tr::SpeculativeDecodingMode::Medusa)
        .def_static("Eagle", &tr::SpeculativeDecodingMode::Eagle)
        .def_static("LookaheadDecoding", &tr::SpeculativeDecodingMode::LookaheadDecoding)
        .def_static("ExplicitDraftTokens", &tr::SpeculativeDecodingMode::ExplicitDraftTokens)
        .def_prop_ro("is_none", &tr::SpeculativeDecodingMode::isNone)
        .def_prop_ro("is_draft_tokens_external", &tr::SpeculativeDecodingMode::isDraftTokensExternal)
        .def_prop_ro("is_medusa", &tr::SpeculativeDecodingMode::isMedusa)
        .def_prop_ro("is_eagle", &tr::SpeculativeDecodingMode::isEagle)
        .def_prop_ro("is_lookahead_decoding", &tr::SpeculativeDecodingMode::isLookaheadDecoding)
        .def_prop_ro("is_explicit_draft_tokens", &tr::SpeculativeDecodingMode::isExplicitDraftTokens)
        .def_prop_ro("updates_position_ids", &tr::SpeculativeDecodingMode::updatesPositionIds)
        .def_prop_ro("requires_attention_mask", &tr::SpeculativeDecodingMode::requiresAttentionMask)
        .def_prop_ro("predicts_draft_tokens", &tr::SpeculativeDecodingMode::predictsDraftTokens)
        .def_prop_ro("needs_kv_cache_rewind", &tr::SpeculativeDecodingMode::needsKVCacheRewind)
        .def_prop_ro("variable_draft_length", &tr::SpeculativeDecodingMode::variableDraftLength)
        .def_prop_ro("has_draft_logits", &tr::SpeculativeDecodingMode::hasDraftLogits)
        .def_prop_ro("needs_decoder_prologue", &tr::SpeculativeDecodingMode::needsDecoderPrologue);
}
} // namespace tensorrt_llm::nanobind::runtime
