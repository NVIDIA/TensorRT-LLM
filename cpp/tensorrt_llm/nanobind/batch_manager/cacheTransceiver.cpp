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

#include "cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/common/bindingUtils.h"
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/nanobind/common/customCasters.h"
#include <ATen/ATen.h>
#include <chrono>
#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>
#include <torch/extension.h>
#include <typeinfo>

using SizeType32 = tensorrt_llm::runtime::SizeType32;

namespace tb = tensorrt_llm::batch_manager;
namespace nb = nanobind;

namespace
{

class PyCacheTransceiver : public tb::BaseCacheTransceiver
{
public:
    // using BaseCacheTransceiver::BaseCacheTransceiver; // Inherit constructors
    NB_TRAMPOLINE(tb::BaseCacheTransceiver, 6);

    void respondAndSendAsync(std::shared_ptr<tb::LlmRequest> llmRequest) override
    {
        NB_OVERRIDE_PURE(respondAndSendAsync, llmRequest);
    }

    void requestAndReceiveSync(std::shared_ptr<tb::LlmRequest> llmRequest) override
    {
        NB_OVERRIDE_PURE(requestAndReceiveSync, llmRequest);
    }

    void requestAndReceiveAsync(std::shared_ptr<tb::LlmRequest> llmRequest) override
    {
        NB_OVERRIDE_PURE(requestAndReceiveAsync, llmRequest);
    }

    tb::RequestStatuses checkContextTransferStatus(
        std::optional<int> const& atLeastRequestNum = std::nullopt, bool markComplete = false) override
    {
        NB_OVERRIDE_PURE(checkContextTransferStatus, atLeastRequestNum, markComplete);
    }

    void checkGenTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) override
    {
        NB_OVERRIDE_PURE(checkGenTransferStatus, atLeastRequestNum);
    }

    bool checkGenTransferComplete() const override
    {
        NB_OVERRIDE_PURE(checkGenTransferComplete);
    }

    bool cancelRequest(std::shared_ptr<tb::LlmRequest> llmRequest) override
    {
        NB_OVERRIDE_PURE(cancelRequest, llmRequest);
    }
};
} // namespace

void tb::CacheTransceiverBindings::initBindings(nb::module_& m)
{
    nb::enum_<tb::LogicalDisposition>(m, "LogicalDisposition")
        .value("ACCEPTED", tb::LogicalDisposition::kACCEPTED)
        .value("ALREADY_TERMINAL", tb::LogicalDisposition::kALREADY_TERMINAL)
        .value("NOT_FOUND", tb::LogicalDisposition::kNOT_FOUND)
        .value("REJECTED", tb::LogicalDisposition::kREJECTED);

    nb::enum_<tb::PhysicalDisposition>(m, "PhysicalDisposition")
        .value("NOT_EXPOSED", tb::PhysicalDisposition::kNOT_EXPOSED)
        .value("ACTIVE", tb::PhysicalDisposition::kACTIVE)
        .value("QUIESCING", tb::PhysicalDisposition::kQUIESCING)
        .value("QUIESCED_SUCCESS", tb::PhysicalDisposition::kQUIESCED_SUCCESS)
        .value("QUIESCED_FAILURE", tb::PhysicalDisposition::kQUIESCED_FAILURE)
        .value("IN_DOUBT", tb::PhysicalDisposition::kIN_DOUBT);

    nb::class_<tb::TransceiverCapabilities>(m, "TransceiverCapabilities")
        .def_ro("protocol_version", &tb::TransceiverCapabilities::protocolVersion)
        .def_ro("qualified_legacy_mode", &tb::TransceiverCapabilities::qualifiedLegacyMode)
        .def_ro("attempt_identity", &tb::TransceiverCapabilities::attemptIdentity)
        .def_ro("endpoint_incarnation", &tb::TransceiverCapabilities::endpointIncarnation)
        .def_ro("allocation_generation_leases", &tb::TransceiverCapabilities::allocationGenerationLeases)
        .def_ro("cancel_before_create_tombstones", &tb::TransceiverCapabilities::cancelBeforeCreateTombstones)
        .def_ro("publication_gate", &tb::TransceiverCapabilities::publicationGate)
        .def_ro("in_flight_cancellation", &tb::TransceiverCapabilities::inFlightCancellation)
        .def_ro("exact_writer_tracking", &tb::TransceiverCapabilities::exactWriterTracking)
        .def_ro("submission_fence", &tb::TransceiverCapabilities::submissionFence)
        .def_ro("per_operation_quiescence", &tb::TransceiverCapabilities::perOperationQuiescence)
        .def_ro("endpoint_wide_quiescence", &tb::TransceiverCapabilities::endpointWideQuiescence)
        .def_ro("direct_transfer", &tb::TransceiverCapabilities::directTransfer)
        .def_ro("bounce_transfer", &tb::TransceiverCapabilities::bounceTransfer)
        .def_ro("multi_writer", &tb::TransceiverCapabilities::multiWriter)
        .def_ro("generation_first", &tb::TransceiverCapabilities::generationFirst)
        .def_ro("pipeline_parallel", &tb::TransceiverCapabilities::pipelineParallel)
        .def_ro("tensor_parallel", &tb::TransceiverCapabilities::tensorParallel)
        .def_ro("attention_data_parallel", &tb::TransceiverCapabilities::attentionDataParallel)
        .def_ro("terminal_result_replay", &tb::TransceiverCapabilities::terminalResultReplay);

    nb::class_<tb::CancelResult>(m, "CancelResult")
        .def_ro("logical", &tb::CancelResult::logical)
        .def_ro("physical", &tb::CancelResult::physical)
        .def_ro("retryable", &tb::CancelResult::retryable)
        .def_ro("reason", &tb::CancelResult::reason)
        .def_prop_ro("safe_to_reuse", &tb::CancelResult::safeToReuse);

    nb::class_<tb::ShutdownResult>(m, "ShutdownResult")
        .def_ro("physical", &tb::ShutdownResult::physical)
        .def_ro("in_doubt_context_count", &tb::ShutdownResult::inDoubtContextCount)
        .def_ro("fatal", &tb::ShutdownResult::fatal)
        .def_ro("reason", &tb::ShutdownResult::reason)
        .def_prop_ro("safe_to_release_managers", &tb::ShutdownResult::safeToReleaseManagers);

    nb::class_<tb::BaseCacheTransceiver, PyCacheTransceiver>(m, "BaseCacheTransceiver")
        .def("respond_and_send_async", &BaseCacheTransceiver::respondAndSendAsync)
        .def("request_and_receive_sync", &BaseCacheTransceiver::requestAndReceiveSync,
            nb::call_guard<nb::gil_scoped_release>())
        .def("request_and_receive_async", &BaseCacheTransceiver::requestAndReceiveAsync)
        .def("get_serialized_data_transceiver_state",
            [](tb::BaseCacheTransceiver& self)
            {
                auto serialized = self.getSerializedDataTransceiverState();
                return nb::bytes(serialized.data(), serialized.size());
            })
        .def(
            "check_context_transfer_status",
            [](tb::BaseCacheTransceiver& self, std::optional<int> const& atLeastRequestNum, bool markComplete = false)
            {
                RequestStatuses result;
                {
                    nb::gil_scoped_release release;
                    result = self.checkContextTransferStatus(atLeastRequestNum, markComplete);
                }

                auto completedRequestIds
                    = std::vector<int64_t>(result.completedRequestIds.begin(), result.completedRequestIds.end());
                auto errorRequestIds
                    = std::vector<int64_t>(result.errorRequestIds.begin(), result.errorRequestIds.end());
                return nb::make_tuple(completedRequestIds, errorRequestIds);
            },
            nb::arg("at_least_request_num") = std::nullopt, nb::arg("mark_complete") = false)
        .def("check_gen_transfer_status", &BaseCacheTransceiver::checkGenTransferStatus,
            nb::call_guard<nb::gil_scoped_release>())
        .def("check_gen_transfer_complete", &BaseCacheTransceiver::checkGenTransferComplete)
        .def("cancel_request", &BaseCacheTransceiver::cancelRequest)
        .def("capabilities", &BaseCacheTransceiver::getLifecycleCapabilities)
        .def("cancel_session", &BaseCacheTransceiver::cancelSession, nb::arg("request"), nb::arg("reason") = "")
        .def("poll_session", &BaseCacheTransceiver::pollSession, nb::arg("request_id"))
        .def("fence_submission", &BaseCacheTransceiver::fenceSubmission, nb::arg("request_id"))
        .def("quiesce_session", &BaseCacheTransceiver::quiesceSession, nb::arg("request_id"))
        .def(
            "shutdown_lifecycle",
            [](tb::BaseCacheTransceiver& self, std::int64_t deadline_ms)
            { return self.shutdownLifecycle(std::chrono::milliseconds(deadline_ms)); },
            nb::arg("deadline_ms"))
        .def("has_poisoned_transfer_buffer", &BaseCacheTransceiver::hasPoisonedTransferBuffer);

    nb::enum_<executor::kv_cache::CacheState::AttentionType>(m, "AttentionType")
        .value("DEFAULT", executor::kv_cache::CacheState::AttentionType::kDEFAULT)
        .value("MLA", executor::kv_cache::CacheState::AttentionType::kMLA);

    nb::class_<tb::CacheTransceiver, tb::BaseCacheTransceiver>(m, "CacheTransceiver")
        .def(nb::init<tb::kv_cache_manager::BaseKVCacheManager*, std::vector<SizeType32>, SizeType32, SizeType32,
                 runtime::WorldConfig, std::vector<SizeType32>, tensorrt_llm::DataType,
                 executor::kv_cache::CacheState::AttentionType, std::optional<executor::CacheTransceiverConfig>,
                 std::vector<SizeType32>>(),
            nb::arg("cache_manager"), nb::arg("num_kv_heads_per_layer"), nb::arg("size_per_head"),
            nb::arg("tokens_per_block"), nb::arg("world_config"), nb::arg("attention_layer_num_per_pp"),
            nb::arg("dtype"), nb::arg("attention_type"), nb::arg("cache_transceiver_config") = std::nullopt,
            nb::arg("rnn_layer_num_per_pp") = std::vector<SizeType32>{})
        .def("get_status_dump", &tb::CacheTransceiver::getStatusDump);

    nb::class_<tb::CacheTransceiverComm>(m, "CacheTransceiverComm")
        .def(
            "__init__",
            [](tb::CacheTransceiverComm* self, nb::object pg_obj, std::string pybind11_abi)
            {
                new (self) tb::CacheTransceiverComm(
                    common::get_intrusive_ptr<c10d::ProcessGroup, nb::python_error>(pg_obj.ptr(), pybind11_abi));
            },
            nb::arg("process_group"), nb::arg("pybind11_abi"))
        .def("get_rank", &tb::CacheTransceiverComm::getRank)
        .def("get_size", &tb::CacheTransceiverComm::getSize)
        .def("split", &tb::CacheTransceiverComm::split, nb::arg("color"), nb::arg("key"))
        .def(
            "allgather",
            [](tb::CacheTransceiverComm const& self, int64_t input)
            {
                std::vector<int64_t> out(static_cast<size_t>(self.getSize()));
                c10d::AllgatherOptions options;
                bool ok = self.allgather(input, std::ref(out), options);
                return nb::make_tuple(ok, out);
            },
            nb::arg("input"))
        .def(
            "allgather",
            [](tb::CacheTransceiverComm const& self, double input)
            {
                std::vector<double> out(static_cast<size_t>(self.getSize()));
                c10d::AllgatherOptions options;
                bool ok = self.allgather(input, std::ref(out), options);
                return nb::make_tuple(ok, out);
            },
            nb::arg("input"))
        .def(
            "allgather",
            [](tb::CacheTransceiverComm const& self, char input)
            {
                std::vector<char> out(static_cast<size_t>(self.getSize()));
                c10d::AllgatherOptions options;
                bool ok = self.allgather(input, std::ref(out), options);
                return nb::make_tuple(ok, out);
            },
            nb::arg("input"))
        .def(
            "allgatherv",
            [](tb::CacheTransceiverComm const& self, std::vector<int64_t> input, std::vector<int> const& sizes)
            {
                int total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
                std::vector<int64_t> output(total_size);
                bool ok = self.allgatherv(std::ref(input), std::ref(output), std::cref(sizes));
                return nb::make_tuple(ok, output);
            },
            nb::arg("input"), nb::arg("sizes"))
        .def(
            "allgatherv",
            [](tb::CacheTransceiverComm const& self, std::vector<double> input, std::vector<int> const& sizes)
            {
                int total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
                std::vector<double> output(total_size);
                bool ok = self.allgatherv(std::ref(input), std::ref(output), std::cref(sizes));
                return nb::make_tuple(ok, output);
            },
            nb::arg("input"), nb::arg("sizes"))
        .def(
            "allgatherv",
            [](tb::CacheTransceiverComm const& self, std::vector<char> input, std::vector<int> const& sizes)
            {
                int total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
                std::vector<char> output(total_size);
                bool ok = self.allgatherv(std::ref(input), std::ref(output), std::cref(sizes));
                return nb::make_tuple(ok, output);
            },
            nb::arg("input"), nb::arg("sizes"));

    nb::class_<tb::kv_cache_manager::CacheTransBufferManager>(m, "CacheTransBufferManager")
        .def(nb::init<tb::kv_cache_manager::BaseKVCacheManager*, std::optional<size_t>>(), nb::arg("cache_manager"),
            nb::arg("max_num_tokens") = std::nullopt)
        .def_static("pre_alloc_buffer_size", &tb::kv_cache_manager::CacheTransBufferManager::preAllocBufferSize,
            nb::arg("cache_size_bytes_per_token_per_window"), nb::arg("tokens_per_block"),
            nb::arg("cache_transceiver_config") = nb::none());
}
