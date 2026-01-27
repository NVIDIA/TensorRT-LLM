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

#include "tensorrt_llm/executor/transferAgent.h"

#ifdef ENABLE_NIXL
#include "transferAgent.h"
#endif

#ifdef ENABLE_MOONCAKE
#include "../mooncake_utils/transferAgent.h"
#endif

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
namespace kvc = tensorrt_llm::executor::kv_cache;

NB_MODULE(tensorrt_llm_transfer_agent_binding, m)
{
    m.doc() = "TensorRT-LLM Transfer Agent Python bindings (nanobind)";

    // MemoryType enum
    nb::enum_<kvc::MemoryType>(m, "MemoryType")
        .value("DRAM", kvc::MemoryType::kDRAM)
        .value("VRAM", kvc::MemoryType::kVRAM)
        .value("BLK", kvc::MemoryType::kBLK)
        .value("OBJ", kvc::MemoryType::kOBJ)
        .value("FILE", kvc::MemoryType::kFILE);

    // TransferOp enum
    nb::enum_<kvc::TransferOp>(m, "TransferOp")
        .value("READ", kvc::TransferOp::kREAD)
        .value("WRITE", kvc::TransferOp::kWRITE);

    // TransferState enum
    nb::enum_<kvc::TransferState>(m, "TransferState")
        .value("IN_PROGRESS", kvc::TransferState::kIN_PROGRESS)
        .value("SUCCESS", kvc::TransferState::kSUCCESS)
        .value("FAILURE", kvc::TransferState::kFAILURE);

    // MemoryDesc class
    nb::class_<kvc::MemoryDesc>(m, "MemoryDesc")
        .def(nb::init<uintptr_t, size_t, uint32_t>(), nb::arg("addr"), nb::arg("len"), nb::arg("device_id"))
        .def_prop_ro("addr", &kvc::MemoryDesc::getAddr)
        .def_prop_ro("len", &kvc::MemoryDesc::getLen)
        .def_prop_ro("device_id", &kvc::MemoryDesc::getDeviceId);

    // MemoryDescs class
    nb::class_<kvc::MemoryDescs>(m, "MemoryDescs")
        .def(nb::init<kvc::MemoryType, std::vector<kvc::MemoryDesc>>(), nb::arg("type"), nb::arg("descs"))
        // Batch constructor from list of tuples: [(ptr, size, device_id), ...]
        .def(
            "__init__",
            [](kvc::MemoryDescs* self, kvc::MemoryType type,
                std::vector<std::tuple<uintptr_t, size_t, uint32_t>> const& tuples)
            {
                std::vector<kvc::MemoryDesc> descs;
                descs.reserve(tuples.size());
                for (auto const& [addr, len, deviceId] : tuples)
                {
                    descs.emplace_back(addr, len, deviceId);
                }
                new (self) kvc::MemoryDescs(type, std::move(descs));
            },
            nb::arg("type"), nb::arg("tuples"))
        .def_prop_ro("type", &kvc::MemoryDescs::getType)
        .def_prop_ro("descs", &kvc::MemoryDescs::getDescs);

    // AgentDesc class
    nb::class_<kvc::AgentDesc>(m, "AgentDesc")
        .def(
            "__init__",
            [](kvc::AgentDesc* self, nb::bytes data)
            {
                std::string str(data.c_str(), data.size());
                new (self) kvc::AgentDesc{std::move(str)};
            },
            nb::arg("backend_agent_desc"))
        .def(nb::init<std::string>(), nb::arg("backend_agent_desc"))
        .def_prop_ro("backend_agent_desc",
            [](kvc::AgentDesc const& self)
            {
                auto const& desc = self.getBackendAgentDesc();
                return nb::bytes(desc.data(), desc.size());
            });

    // TransferRequest class
    nb::class_<kvc::TransferRequest>(m, "TransferRequest")
        .def(nb::init<kvc::TransferOp, kvc::TransferDescs, kvc::TransferDescs, std::string const&,
                 std::optional<kvc::SyncMessage>>(),
            nb::arg("op"), nb::arg("src_descs"), nb::arg("dst_descs"), nb::arg("remote_name"),
            nb::arg("sync_message") = std::nullopt)
        .def_prop_ro("op", &kvc::TransferRequest::getOp)
        .def_prop_ro("src_descs", &kvc::TransferRequest::getSrcDescs)
        .def_prop_ro("dst_descs", &kvc::TransferRequest::getDstDescs)
        .def_prop_ro("remote_name", &kvc::TransferRequest::getRemoteName)
        .def_prop_ro("sync_message", &kvc::TransferRequest::getSyncMessage);

    // TransferStatus base class
    nb::class_<kvc::TransferStatus>(m, "TransferStatus")
        .def("is_completed", &kvc::TransferStatus::isCompleted)
        .def("wait", &kvc::TransferStatus::wait, nb::arg("timeout_ms") = -1);

    // BaseAgentConfig struct
    nb::class_<kvc::BaseAgentConfig>(m, "BaseAgentConfig")
        .def(nb::init<>())
        .def(
            "__init__",
            [](kvc::BaseAgentConfig* self, std::string name, bool use_prog_thread, bool multi_thread,
                bool use_listen_thread, bool enable_telemetry,
                std::unordered_map<std::string, std::string> backend_params)
            {
                new (self) kvc::BaseAgentConfig{std::move(name), use_prog_thread, multi_thread, use_listen_thread,
                    enable_telemetry, std::move(backend_params)};
            },
            nb::arg("name"), nb::arg("use_prog_thread") = true, nb::arg("multi_thread") = false,
            nb::arg("use_listen_thread") = false, nb::arg("enable_telemetry") = false,
            nb::arg("backend_params") = std::unordered_map<std::string, std::string>{})
        .def_rw("name", &kvc::BaseAgentConfig::mName)
        .def_rw("use_prog_thread", &kvc::BaseAgentConfig::useProgThread)
        .def_rw("multi_thread", &kvc::BaseAgentConfig::multiThread)
        .def_rw("use_listen_thread", &kvc::BaseAgentConfig::useListenThread)
        .def_rw("enable_telemetry", &kvc::BaseAgentConfig::enableTelemetry)
        .def_rw("backend_params", &kvc::BaseAgentConfig::backendParams);

    // BaseTransferAgent class (abstract base)
    nb::class_<kvc::BaseTransferAgent>(m, "BaseTransferAgent")
        .def("register_memory", &kvc::BaseTransferAgent::registerMemory, nb::arg("descs"))
        .def("deregister_memory", &kvc::BaseTransferAgent::deregisterMemory, nb::arg("descs"))
        .def("load_remote_agent",
            nb::overload_cast<std::string const&, kvc::AgentDesc const&>(&kvc::BaseTransferAgent::loadRemoteAgent),
            nb::arg("name"), nb::arg("agent_desc"))
        .def("load_remote_agent_by_connection",
            nb::overload_cast<std::string const&, kvc::ConnectionInfoType const&>(
                &kvc::BaseTransferAgent::loadRemoteAgent),
            nb::arg("name"), nb::arg("connection_info"))
        .def("get_local_agent_desc", &kvc::BaseTransferAgent::getLocalAgentDesc)
        .def("invalidate_remote_agent", &kvc::BaseTransferAgent::invalidateRemoteAgent, nb::arg("name"))
        .def(
            "submit_transfer_requests",
            [](kvc::BaseTransferAgent& self, kvc::TransferRequest const& request)
            { return self.submitTransferRequests(request).release(); },
            nb::arg("request"), nb::rv_policy::take_ownership)
        .def(
            "notify_sync_message", &kvc::BaseTransferAgent::notifySyncMessage, nb::arg("name"), nb::arg("sync_message"))
        .def("get_notified_sync_messages", &kvc::BaseTransferAgent::getNotifiedSyncMessages)
        .def("get_local_connection_info", &kvc::BaseTransferAgent::getLocalConnectionInfo)
        .def("check_remote_descs", &kvc::BaseTransferAgent::checkRemoteDescs, nb::arg("name"), nb::arg("memory_descs"));

#ifdef ENABLE_NIXL
    // NixlTransferStatus class - release GIL for blocking operations
    nb::class_<kvc::NixlTransferStatus, kvc::TransferStatus>(m, "NixlTransferStatus")
        .def("is_completed", &kvc::NixlTransferStatus::isCompleted, nb::call_guard<nb::gil_scoped_release>())
        .def("wait", &kvc::NixlTransferStatus::wait, nb::arg("timeout_ms") = -1,
            nb::call_guard<nb::gil_scoped_release>());

    // NixlTransferAgent class
    nb::class_<kvc::NixlTransferAgent, kvc::BaseTransferAgent>(m, "NixlTransferAgent")
        .def(nb::init<kvc::BaseAgentConfig const&>(), nb::arg("config"))
        .def("register_memory", &kvc::NixlTransferAgent::registerMemory, nb::arg("descs"))
        .def("deregister_memory", &kvc::NixlTransferAgent::deregisterMemory, nb::arg("descs"))
        .def("load_remote_agent",
            nb::overload_cast<std::string const&, kvc::AgentDesc const&>(&kvc::NixlTransferAgent::loadRemoteAgent),
            nb::arg("name"), nb::arg("agent_desc"))
        .def("load_remote_agent_by_connection",
            nb::overload_cast<std::string const&, kvc::ConnectionInfoType const&>(
                &kvc::NixlTransferAgent::loadRemoteAgent),
            nb::arg("name"), nb::arg("connection_info"))
        .def("get_local_agent_desc", &kvc::NixlTransferAgent::getLocalAgentDesc)
        .def("get_local_connection_info", &kvc::NixlTransferAgent::getLocalConnectionInfo)
        .def("invalidate_remote_agent", &kvc::NixlTransferAgent::invalidateRemoteAgent, nb::arg("name"))
        .def(
            "submit_transfer_requests",
            [](kvc::NixlTransferAgent& self, kvc::TransferRequest const& request)
            { return self.submitTransferRequests(request).release(); },
            nb::arg("request"), nb::rv_policy::take_ownership, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "notify_sync_message", &kvc::NixlTransferAgent::notifySyncMessage, nb::arg("name"), nb::arg("sync_message"))
        .def("get_notified_sync_messages", &kvc::NixlTransferAgent::getNotifiedSyncMessages)
        .def("check_remote_descs", &kvc::NixlTransferAgent::checkRemoteDescs, nb::arg("name"), nb::arg("memory_descs"));
#endif

#ifdef ENABLE_MOONCAKE
    // MooncakeTransferStatus class - release GIL for blocking operations
    nb::class_<kvc::MooncakeTransferStatus, kvc::TransferStatus>(m, "MooncakeTransferStatus")
        .def("is_completed", &kvc::MooncakeTransferStatus::isCompleted, nb::call_guard<nb::gil_scoped_release>())
        .def("wait", &kvc::MooncakeTransferStatus::wait, nb::arg("timeout_ms") = -1,
            nb::call_guard<nb::gil_scoped_release>());

    // MooncakeTransferAgent class
    nb::class_<kvc::MooncakeTransferAgent, kvc::BaseTransferAgent>(m, "MooncakeTransferAgent")
        .def(nb::init<kvc::BaseAgentConfig const&>(), nb::arg("config"))
        .def("register_memory", &kvc::MooncakeTransferAgent::registerMemory, nb::arg("descs"))
        .def("deregister_memory", &kvc::MooncakeTransferAgent::deregisterMemory, nb::arg("descs"))
        .def("load_remote_agent",
            nb::overload_cast<std::string const&, kvc::AgentDesc const&>(&kvc::MooncakeTransferAgent::loadRemoteAgent),
            nb::arg("name"), nb::arg("agent_desc"))
        .def("load_remote_agent_by_connection",
            nb::overload_cast<std::string const&, kvc::ConnectionInfoType const&>(
                &kvc::MooncakeTransferAgent::loadRemoteAgent),
            nb::arg("name"), nb::arg("connection_info"))
        .def("get_local_agent_desc", &kvc::MooncakeTransferAgent::getLocalAgentDesc)
        .def("get_local_connection_info", &kvc::MooncakeTransferAgent::getLocalConnectionInfo)
        .def("invalidate_remote_agent", &kvc::MooncakeTransferAgent::invalidateRemoteAgent, nb::arg("name"))
        .def(
            "submit_transfer_requests",
            [](kvc::MooncakeTransferAgent& self, kvc::TransferRequest const& request)
            { return self.submitTransferRequests(request).release(); },
            nb::arg("request"), nb::rv_policy::take_ownership, nb::call_guard<nb::gil_scoped_release>())
        .def("notify_sync_message", &kvc::MooncakeTransferAgent::notifySyncMessage, nb::arg("name"),
            nb::arg("sync_message"))
        .def("get_notified_sync_messages", &kvc::MooncakeTransferAgent::getNotifiedSyncMessages)
        .def("check_remote_descs", &kvc::MooncakeTransferAgent::checkRemoteDescs, nb::arg("name"),
            nb::arg("memory_descs"));
#endif

    // Factory function to create transfer agent by backend name (uses dynamic loading)
    m.def(
        "make_transfer_agent",
        [](std::string const& backend, kvc::BaseAgentConfig const& config) -> kvc::BaseTransferAgent*
        { return kvc::makeTransferAgent(backend, &config).release(); },
        nb::arg("backend"), nb::arg("config"), nb::rv_policy::take_ownership,
        "Create a transfer agent by backend name ('nixl' or 'mooncake'). Uses dynamic loading.");

    // Expose which backends are available
#ifdef ENABLE_NIXL
    m.attr("NIXL_ENABLED") = true;
#else
    m.attr("NIXL_ENABLED") = false;
#endif

#ifdef ENABLE_MOONCAKE
    m.attr("MOONCAKE_ENABLED") = true;
#else
    m.attr("MOONCAKE_ENABLED") = false;
#endif
}
