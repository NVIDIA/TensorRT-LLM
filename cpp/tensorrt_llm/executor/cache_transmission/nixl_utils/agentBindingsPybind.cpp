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

#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace kvc = tensorrt_llm::executor::kv_cache;

PYBIND11_MODULE(tensorrt_llm_transfer_agent_binding, m)
{
    m.doc() = "TensorRT-LLM Transfer Agent Python bindings (pybind11)";

    // MemoryType enum
    py::enum_<kvc::MemoryType>(m, "MemoryType")
        .value("DRAM", kvc::MemoryType::kDRAM)
        .value("VRAM", kvc::MemoryType::kVRAM)
        .value("BLK", kvc::MemoryType::kBLK)
        .value("OBJ", kvc::MemoryType::kOBJ)
        .value("FILE", kvc::MemoryType::kFILE);

    // TransferOp enum
    py::enum_<kvc::TransferOp>(m, "TransferOp")
        .value("READ", kvc::TransferOp::kREAD)
        .value("WRITE", kvc::TransferOp::kWRITE);

    // TransferState enum
    py::enum_<kvc::TransferState>(m, "TransferState")
        .value("IN_PROGRESS", kvc::TransferState::kIN_PROGRESS)
        .value("SUCCESS", kvc::TransferState::kSUCCESS)
        .value("FAILURE", kvc::TransferState::kFAILURE);

    // MemoryDesc class
    py::class_<kvc::MemoryDesc>(m, "MemoryDesc")
        .def(py::init<uintptr_t, size_t, uint32_t>(), py::arg("addr"), py::arg("len"), py::arg("device_id"))
        .def_property_readonly("addr", &kvc::MemoryDesc::getAddr)
        .def_property_readonly("len", &kvc::MemoryDesc::getLen)
        .def_property_readonly("device_id", &kvc::MemoryDesc::getDeviceId);

    // MemoryDescs class
    py::class_<kvc::MemoryDescs>(m, "MemoryDescs")
        .def(py::init<kvc::MemoryType, std::vector<kvc::MemoryDesc>>(), py::arg("type"), py::arg("descs"))
        .def_property_readonly("type", &kvc::MemoryDescs::getType)
        .def_property_readonly("descs", &kvc::MemoryDescs::getDescs);

    // AgentDesc class
    py::class_<kvc::AgentDesc>(m, "AgentDesc")
        .def(py::init(
                 [](py::bytes data)
                 {
                     std::string str(PyBytes_AsString(data.ptr()), PyBytes_Size(data.ptr()));
                     return kvc::AgentDesc{std::move(str)};
                 }),
            py::arg("backend_agent_desc"))
        .def(py::init<std::string>(), py::arg("backend_agent_desc"))
        .def_property_readonly("backend_agent_desc",
            [](kvc::AgentDesc const& self)
            {
                auto const& desc = self.getBackendAgentDesc();
                return py::bytes(desc.data(), desc.size());
            });

    // TransferRequest class
    py::class_<kvc::TransferRequest>(m, "TransferRequest")
        .def(py::init<kvc::TransferOp, kvc::TransferDescs, kvc::TransferDescs, std::string const&,
                 std::optional<kvc::SyncMessage>>(),
            py::arg("op"), py::arg("src_descs"), py::arg("dst_descs"), py::arg("remote_name"),
            py::arg("sync_message") = std::nullopt)
        .def_property_readonly("op", &kvc::TransferRequest::getOp)
        .def_property_readonly("src_descs", &kvc::TransferRequest::getSrcDescs)
        .def_property_readonly("dst_descs", &kvc::TransferRequest::getDstDescs)
        .def_property_readonly("remote_name", &kvc::TransferRequest::getRemoteName)
        .def_property_readonly("sync_message", &kvc::TransferRequest::getSyncMessage);

    // TransferStatus base class
    py::class_<kvc::TransferStatus>(m, "TransferStatus")
        .def("is_completed", &kvc::TransferStatus::isCompleted)
        .def("wait", &kvc::TransferStatus::wait, py::arg("timeout_ms") = -1);

    // BaseAgentConfig struct
    py::class_<kvc::BaseAgentConfig>(m, "BaseAgentConfig")
        .def(py::init<>())
        .def(py::init(
                 [](std::string name, bool use_prog_thread, bool multi_thread, bool use_listen_thread,
                     unsigned int num_workers) {
                     return kvc::BaseAgentConfig{
                         std::move(name), use_prog_thread, multi_thread, use_listen_thread, num_workers};
                 }),
            py::arg("name"), py::arg("use_prog_thread") = true, py::arg("multi_thread") = false,
            py::arg("use_listen_thread") = false, py::arg("num_workers") = 1)
        .def_readwrite("name", &kvc::BaseAgentConfig::mName)
        .def_readwrite("use_prog_thread", &kvc::BaseAgentConfig::useProgThread)
        .def_readwrite("multi_thread", &kvc::BaseAgentConfig::multiThread)
        .def_readwrite("use_listen_thread", &kvc::BaseAgentConfig::useListenThread)
        .def_readwrite("num_workers", &kvc::BaseAgentConfig::numWorkers);

    // BaseTransferAgent class (abstract base)
    py::class_<kvc::BaseTransferAgent>(m, "BaseTransferAgent")
        .def("register_memory", &kvc::BaseTransferAgent::registerMemory, py::arg("descs"))
        .def("deregister_memory", &kvc::BaseTransferAgent::deregisterMemory, py::arg("descs"))
        .def("load_remote_agent",
            py::overload_cast<std::string const&, kvc::AgentDesc const&>(&kvc::BaseTransferAgent::loadRemoteAgent),
            py::arg("name"), py::arg("agent_desc"))
        .def("load_remote_agent_by_connection",
            py::overload_cast<std::string const&, kvc::ConnectionInfoType const&>(
                &kvc::BaseTransferAgent::loadRemoteAgent),
            py::arg("name"), py::arg("connection_info"))
        .def("get_local_agent_desc", &kvc::BaseTransferAgent::getLocalAgentDesc)
        .def("invalidate_remote_agent", &kvc::BaseTransferAgent::invalidateRemoteAgent, py::arg("name"))
        .def(
            "submit_transfer_requests",
            [](kvc::BaseTransferAgent& self, kvc::TransferRequest const& request)
            { return self.submitTransferRequests(request).release(); },
            py::arg("request"), py::return_value_policy::take_ownership)
        .def(
            "notify_sync_message", &kvc::BaseTransferAgent::notifySyncMessage, py::arg("name"), py::arg("sync_message"))
        .def("get_notified_sync_messages", &kvc::BaseTransferAgent::getNotifiedSyncMessages)
        .def("get_local_connection_info", &kvc::BaseTransferAgent::getLocalConnectionInfo)
        .def("check_remote_descs", &kvc::BaseTransferAgent::checkRemoteDescs, py::arg("name"), py::arg("memory_descs"));

#ifdef ENABLE_NIXL
    // NixlTransferStatus class - release GIL for blocking operations
    py::class_<kvc::NixlTransferStatus, kvc::TransferStatus>(m, "NixlTransferStatus")
        .def("is_completed", &kvc::NixlTransferStatus::isCompleted, py::call_guard<py::gil_scoped_release>())
        .def("wait", &kvc::NixlTransferStatus::wait, py::arg("timeout_ms") = -1,
            py::call_guard<py::gil_scoped_release>());

    // NixlTransferAgent class
    py::class_<kvc::NixlTransferAgent, kvc::BaseTransferAgent>(m, "NixlTransferAgent")
        .def(py::init<kvc::BaseAgentConfig const&>(), py::arg("config"))
        .def("register_memory", &kvc::NixlTransferAgent::registerMemory, py::arg("descs"))
        .def("deregister_memory", &kvc::NixlTransferAgent::deregisterMemory, py::arg("descs"))
        .def("load_remote_agent",
            py::overload_cast<std::string const&, kvc::AgentDesc const&>(&kvc::NixlTransferAgent::loadRemoteAgent),
            py::arg("name"), py::arg("agent_desc"))
        .def("load_remote_agent_by_connection",
            py::overload_cast<std::string const&, kvc::ConnectionInfoType const&>(
                &kvc::NixlTransferAgent::loadRemoteAgent),
            py::arg("name"), py::arg("connection_info"))
        .def("get_local_agent_desc", &kvc::NixlTransferAgent::getLocalAgentDesc)
        .def("get_local_connection_info", &kvc::NixlTransferAgent::getLocalConnectionInfo)
        .def("invalidate_remote_agent", &kvc::NixlTransferAgent::invalidateRemoteAgent, py::arg("name"))
        .def(
            "submit_transfer_requests",
            [](kvc::NixlTransferAgent& self, kvc::TransferRequest const& request)
            { return self.submitTransferRequests(request).release(); },
            py::arg("request"), py::return_value_policy::take_ownership, py::call_guard<py::gil_scoped_release>())
        .def(
            "notify_sync_message", &kvc::NixlTransferAgent::notifySyncMessage, py::arg("name"), py::arg("sync_message"))
        .def("get_notified_sync_messages", &kvc::NixlTransferAgent::getNotifiedSyncMessages)
        .def("check_remote_descs", &kvc::NixlTransferAgent::checkRemoteDescs, py::arg("name"), py::arg("memory_descs"));
#endif

#ifdef ENABLE_MOONCAKE
    // MooncakeTransferStatus class - release GIL for blocking operations
    py::class_<kvc::MooncakeTransferStatus, kvc::TransferStatus>(m, "MooncakeTransferStatus")
        .def("is_completed", &kvc::MooncakeTransferStatus::isCompleted, py::call_guard<py::gil_scoped_release>())
        .def("wait", &kvc::MooncakeTransferStatus::wait, py::arg("timeout_ms") = -1,
            py::call_guard<py::gil_scoped_release>());

    // MooncakeTransferAgent class
    py::class_<kvc::MooncakeTransferAgent, kvc::BaseTransferAgent>(m, "MooncakeTransferAgent")
        .def(py::init<kvc::BaseAgentConfig const&>(), py::arg("config"))
        .def("register_memory", &kvc::MooncakeTransferAgent::registerMemory, py::arg("descs"))
        .def("deregister_memory", &kvc::MooncakeTransferAgent::deregisterMemory, py::arg("descs"))
        .def("load_remote_agent",
            py::overload_cast<std::string const&, kvc::AgentDesc const&>(&kvc::MooncakeTransferAgent::loadRemoteAgent),
            py::arg("name"), py::arg("agent_desc"))
        .def("load_remote_agent_by_connection",
            py::overload_cast<std::string const&, kvc::ConnectionInfoType const&>(
                &kvc::MooncakeTransferAgent::loadRemoteAgent),
            py::arg("name"), py::arg("connection_info"))
        .def("get_local_agent_desc", &kvc::MooncakeTransferAgent::getLocalAgentDesc)
        .def("get_local_connection_info", &kvc::MooncakeTransferAgent::getLocalConnectionInfo)
        .def("invalidate_remote_agent", &kvc::MooncakeTransferAgent::invalidateRemoteAgent, py::arg("name"))
        .def(
            "submit_transfer_requests",
            [](kvc::MooncakeTransferAgent& self, kvc::TransferRequest const& request)
            { return self.submitTransferRequests(request).release(); },
            py::arg("request"), py::return_value_policy::take_ownership, py::call_guard<py::gil_scoped_release>())
        .def("notify_sync_message", &kvc::MooncakeTransferAgent::notifySyncMessage, py::arg("name"),
            py::arg("sync_message"))
        .def("get_notified_sync_messages", &kvc::MooncakeTransferAgent::getNotifiedSyncMessages)
        .def("check_remote_descs", &kvc::MooncakeTransferAgent::checkRemoteDescs, py::arg("name"),
            py::arg("memory_descs"));
#endif

    // Factory function to create transfer agent by backend name (uses dynamic loading)
    m.def(
        "make_transfer_agent",
        [](std::string const& backend, kvc::BaseAgentConfig const& config) -> kvc::BaseTransferAgent*
        { return kvc::makeTransferAgent(backend, &config).release(); },
        py::arg("backend"), py::arg("config"), py::return_value_policy::take_ownership,
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
