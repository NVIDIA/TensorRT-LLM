#pragma once

#include <pybind11/pybind11.h>

namespace tensorrt_llm::pybind::executor
{

void bindDataTransceiverStateUtils(pybind11::module_& m);

} // namespace tensorrt_llm::pybind::executor
