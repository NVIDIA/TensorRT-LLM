/*
 * Python bindings for DOCA RDMA Endpoint
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "doca_rdma_endpoint.h"

namespace py = pybind11;

PYBIND11_MODULE(doca_rdma, m)
{
    m.doc() = "DOCA RDMA Endpoint for GPU-initiated RDMA operations";

    // QpInfo structure
    py::class_<QpInfo>(m, "QpInfo")
        .def(py::init<>())
        .def_readwrite("qpn", &QpInfo::qpn)
        .def_readwrite("psn", &QpInfo::psn)
        .def_readwrite("lid", &QpInfo::lid)
        .def_property(
            "gid", [](QpInfo const& q) { return py::bytes(reinterpret_cast<const char*>(q.gid), 16); },
            [](QpInfo& q, py::bytes b)
            {
                std::string s = b;
                if (s.size() == 16)
                {
                    memcpy(q.gid, s.data(), 16);
                }
            });

    // BufferInfo structure
    py::class_<BufferInfo>(m, "BufferInfo")
        .def(py::init<>())
        .def_readwrite("addr", &BufferInfo::addr)
        .def_readwrite("rkey", &BufferInfo::rkey)
        .def_readwrite("lkey", &BufferInfo::lkey)
        .def_readwrite("length", &BufferInfo::length)
        .def_readwrite("buf_arr_index", &BufferInfo::buf_arr_index)
        .def_readwrite("gpu_buf_arr", &BufferInfo::gpu_buf_arr)
        .def_readwrite("original_addr", &BufferInfo::original_addr)
        .def("__repr__",
            [](BufferInfo const& b)
            {
                char buf[256];
                snprintf(buf, sizeof(buf), "BufferInfo(addr=0x%lx, rkey=0x%x, lkey=0x%x, length=%zu, buf_arr_idx=%d)",
                    b.addr, b.rkey, b.lkey, b.length, b.buf_arr_index);
                return std::string(buf);
            });

    // DocaRdmaEndpoint class
    py::class_<DocaRdmaEndpoint>(m, "DocaRdmaEndpoint")
        .def(py::init<>())

        // Initialization
        .def("init", &DocaRdmaEndpoint::init, py::arg("gpu_id"), py::arg("nic_name"), "Initialize DOCA RDMA endpoint")
        .def("cleanup", &DocaRdmaEndpoint::cleanup, "Cleanup resources")

        // Buffer registration
        .def(
            "register_buffer",
            [](DocaRdmaEndpoint& self, uint64_t gpu_ptr, size_t length)
            { return self.register_buffer(reinterpret_cast<void*>(gpu_ptr), length); },
            py::arg("gpu_ptr"), py::arg("length"), "Register GPU buffer for RDMA (pass tensor.data_ptr())")
        .def("deregister_buffer", &DocaRdmaEndpoint::deregister_buffer, py::arg("buf"), "Deregister GPU buffer")

        // Connection management
        .def("connect_to_peer", &DocaRdmaEndpoint::connect_to_peer, py::arg("host"), py::arg("port"),
            py::arg("peer_name"), "Connect to remote peer")
        .def("accept_peer", &DocaRdmaEndpoint::accept_peer, py::arg("port"), py::arg("peer_name"),
            "Listen for and accept one remote peer")
        .def("exchange_buffer_info", &DocaRdmaEndpoint::exchange_buffer_info, py::arg("peer_name"),
            py::arg("local_send"), py::arg("local_recv"), "Exchange buffer information with peer")
        .def("send_ready_signal", &DocaRdmaEndpoint::send_ready_signal, py::arg("peer_name"),
            "Signal that GPU is ready to receive (call after posting recv WRs)")
        .def("alloc_doca_gpu_buffer", &DocaRdmaEndpoint::alloc_doca_gpu_buffer, py::arg("length"),
            "Allocate GPU memory using DOCA and register for RDMA")

        // Device pointers (for CUDA kernels)
        .def(
            "get_qp_dev_ptr",
            [](DocaRdmaEndpoint& self, std::string const& peer_name)
            { return reinterpret_cast<uint64_t>(self.get_qp_dev_ptr(peer_name)); },
            py::arg("peer_name"), "Get device pointer for QP (for use in CUDA kernels)")
        .def(
            "get_send_cq_dev_ptr",
            [](DocaRdmaEndpoint& self, std::string const& peer_name)
            { return reinterpret_cast<uint64_t>(self.get_send_cq_dev_ptr(peer_name)); },
            py::arg("peer_name"), "Get device pointer for send CQ")
        .def(
            "get_recv_cq_dev_ptr",
            [](DocaRdmaEndpoint& self, std::string const& peer_name)
            { return reinterpret_cast<uint64_t>(self.get_recv_cq_dev_ptr(peer_name)); },
            py::arg("peer_name"), "Get device pointer for recv CQ")

        // Buffer info access
        .def("get_remote_recv_buf", &DocaRdmaEndpoint::get_remote_recv_buf, py::arg("peer_name"),
            "Get remote peer's receive buffer info")
        .def("get_local_send_buf", &DocaRdmaEndpoint::get_local_send_buf, py::arg("peer_name"),
            "Get local send buffer info for peer")

        // Post receive work requests
        .def(
            "post_recv",
            [](DocaRdmaEndpoint& self, std::string const& peer_name, uint64_t buf, size_t length, uint32_t lkey)
            { return self.post_recv(peer_name, reinterpret_cast<void*>(buf), length, lkey); },
            py::arg("peer_name"), py::arg("buf"), py::arg("length"), py::arg("lkey"),
            "Post receive work request for incoming WRITE_WITH_IMM")

        // Host-side RDMA (for testing)
        .def(
            "rdma_write_with_imm",
            [](DocaRdmaEndpoint& self, std::string const& peer_name, uint64_t local_buf, size_t length,
                uint32_t imm_value)
            { return self.rdma_write_with_imm(peer_name, reinterpret_cast<void*>(local_buf), length, imm_value); },
            py::arg("peer_name"), py::arg("local_buf"), py::arg("length"), py::arg("imm_value"),
            "Perform RDMA WRITE WITH IMMEDIATE (host-side, for testing)")
        .def(
            "poll_for_completion",
            [](DocaRdmaEndpoint& self, std::string const& peer_name)
            {
                uint32_t imm = 0;
                bool completed = self.poll_for_completion(peer_name, &imm);
                return py::make_tuple(completed, imm);
            },
            py::arg("peer_name"), "Poll for completion, returns (completed, imm_value)")

        // CPU_PROXY support
        .def("cpu_proxy_enabled", &DocaRdmaEndpoint::cpu_proxy_enabled, py::arg("peer_name"),
            "Query whether CPU proxy mode is enabled for this QP")
        .def("cpu_proxy_progress", &DocaRdmaEndpoint::cpu_proxy_progress, py::arg("peer_name"),
            "Tick CPU proxy progress (poll producer counter, ring NIC doorbell)")
        .def(
            "poll_send_cq_cpu",
            [](DocaRdmaEndpoint& self, std::string const& peer_name, int max_out)
            {
                std::vector<uint64_t> out(max_out > 0 ? max_out : 0);
                int n = self.poll_send_cq_cpu(peer_name, out.data(), max_out);
                py::list lst;
                if (n > 0)
                {
                    for (int i = 0; i < n; ++i)
                        lst.append(out[(size_t) i]);
                }
                return py::make_tuple(n, lst);
            },
            py::arg("peer_name"), py::arg("max_out") = 16,
            "CPU-side send CQ poll, returns (n, [packed_cqe_words]); negative n on error")

        // Properties
        .def_property_readonly("local_name", &DocaRdmaEndpoint::get_local_name)
        .def_property_readonly("is_initialized", &DocaRdmaEndpoint::is_initialized);

    // Version info
    m.attr("__version__") = "0.1.0";
}
