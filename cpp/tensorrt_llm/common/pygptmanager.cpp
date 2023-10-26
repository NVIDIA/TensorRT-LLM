#include "tensorrt_llm/batch_manager/GptManager.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(pygptmanager, m)
{
    m.doc() = "Python bindings for GptManager";

    py::class_<tensorrt_llm::batch_manager::GptManager>(m, "GptManager")
        .def(py::init<const std::filesystem::path&, tensorrt_llm::batch_manager::TrtGptModelType, int32_t,
                 tensorrt_llm::batch_manager::batch_scheduler::SchedulerPolicy,
                 tensorrt_llm::batch_manager::GetInferenceRequestsCallback,
                 tensorrt_llm::batch_manager::SendResponseCallback, tensorrt_llm::batch_manager::PollStopSignalCallback,
                 tensorrt_llm::batch_manager::ReturnBatchManagerStatsCallback>(),
            py::arg("trtEnginePath"), py::arg("modelType"), py::arg("maxBeamWidth"), py::arg("schedulerPolicy"),
            py::arg("getInferenceRequestsCb"), py::arg("sendResponseCb"), py::arg("pollStopSignalCb") = nullptr,
            py::arg("returnBatchManagerStatsCb") = nullptr)
        .def("fetchNewRequests", &tensorrt_llm::batch_manager::GptManager::fetchNewRequests)
        .def("returnCompletedRequests", &tensorrt_llm::batch_manager::GptManager::returnCompletedRequests)
        .def("pollStopSignals", &tensorrt_llm::batch_manager::GptManager::pollStopSignals)
        .def("returnBatchManagerStats", &tensorrt_llm::batch_manager::GptManager::returnBatchManagerStats);
}