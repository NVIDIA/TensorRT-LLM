/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tllmExceptions.h"

namespace py = pybind11;
namespace tc = tensorrt_llm::common;

namespace tensorrt_llm::pybind::common
{

void initExceptionsBindings(py::module_& m)
{
    // Bind the RequestErrorCode enum
    py::enum_<tc::RequestErrorCode>(m, "RequestErrorCode")
        .value("UNKNOWN_ERROR", tc::RequestErrorCode::kUNKNOWN_ERROR)
        .value("NETWORK_ERROR", tc::RequestErrorCode::kNETWORK_ERROR)
        .export_values();

    // Create the RequestSpecificException Python exception class
    static PyObject* request_specific_exc
        = PyErr_NewException("tensorrt_llm.RequestSpecificException", nullptr, nullptr);

    // Add attributes to the Python exception class
    py::handle(request_specific_exc).attr("request_id") = py::none();
    py::handle(request_specific_exc).attr("error_code") = py::none();

    m.add_object("RequestSpecificException", py::handle(request_specific_exc));

    // Register exception translator to convert C++ exceptions to Python
    py::register_exception_translator(
        [](std::exception_ptr p)
        {
            try
            {
                if (p)
                    std::rethrow_exception(p);
            }
            catch (const tc::RequestSpecificException& e)
            {
                // Create a Python exception with the request ID and error code information
                py::object py_exc = py::cast(e);
                py::object request_id = py::cast(e.getRequestId());
                py::object error_code = py::cast(static_cast<uint32_t>(e.getErrorCode()));

                // Set additional attributes on the exception
                py_exc.attr("request_id") = request_id;
                py_exc.attr("error_code") = error_code;

                PyErr_SetObject(request_specific_exc, py_exc.ptr());
            }
        });
}

} // namespace tensorrt_llm::pybind::common
