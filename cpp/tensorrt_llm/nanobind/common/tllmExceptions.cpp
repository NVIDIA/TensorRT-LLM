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
#include "tensorrt_llm/common/tllmException.h"
#include <nanobind/nanobind.h>

namespace tc = tensorrt_llm::common;
namespace nb = nanobind;

namespace tensorrt_llm::nanobind::common
{

void initExceptionsBindings(nb::module_& m)
{
    // Bind the RequestErrorCode enum
    nb::enum_<tc::RequestErrorCode>(m, "RequestErrorCode")
        .value("UNKNOWN_ERROR", tc::RequestErrorCode::kUNKNOWN_ERROR)
        .value("NETWORK_ERROR", tc::RequestErrorCode::kNETWORK_ERROR)
        .export_values();

    // Create the RequestSpecificException Python exception class
    static nb::object request_specific_exc = nb::exception<tc::RequestSpecificException>(m, "RequestSpecificException");

    // Add attributes to the Python exception class
    request_specific_exc.attr("request_id") = nb::none();
    request_specific_exc.attr("error_code") = nb::none();

    // Register exception translator to convert C++ exceptions to Python
    nb::register_exception_translator(
        [](std::exception_ptr const& p, void*)
        {
            try
            {
                if (p)
                    std::rethrow_exception(p);
            }
            catch (const tc::RequestSpecificException& e)
            {
                // Create a Python exception with the request ID and error code information
                nb::object py_exc = nb::cast(e);
                nb::object request_id = nb::cast(e.getRequestId());
                nb::object error_code = nb::cast(static_cast<uint32_t>(e.getErrorCode()));

                // Set additional attributes on the exception
                py_exc.attr("request_id") = request_id;
                py_exc.attr("error_code") = error_code;

                PyErr_SetObject(request_specific_exc.ptr(), py_exc.ptr());
            }
        });
}

} // namespace tensorrt_llm::nanobind::common
