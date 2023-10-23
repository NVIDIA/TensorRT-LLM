#include "tensorrt_llm/batch_manager/GptManager.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(pygptmanager, m)
{
    m.doc() = "Python bindings for GptManager";

    py::class_<tensorrt_llm::batch_manager::GptManager>(m, "GptManager")
        .def(py::init<const std::filesystem::path&, tensorrt_llm::batch_manager::TrtGptModelType, int32_t, int32_t,
                 int32_t, tensorrt_llm::batch_manager::GetInferenceRequestsCallback,
                 tensorrt_llm::batch_manager::SendResponseCallback,
                 tensorrt_llm::batch_manager::PollStopSignalCallback>(),
            py::arg("trtEnginePath"), py::arg("modelType"), py::arg("maxSeqLen"), py::arg("maxNumRequests"),
            py::arg("maxBeamWidth"), py::arg("getInferenceRequestsCb"), py::arg("sendResponseCb"),
            py::arg("pollStopSignalCb") = nullptr)
        .def("fetchNewRequests", &tensorrt_llm::batch_manager::GptManager::fetchNewRequests)
        .def("return_completed_requests", &tensorrt_llm::batch_manager::GptManager::return_completed_requests)
        .def("pollStopSignals", &tensorrt_llm::batch_manager::GptManager::pollStopSignals);
}