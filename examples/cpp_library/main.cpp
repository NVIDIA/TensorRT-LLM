/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <iostream>
#include <stdexcept>
#include <string>

#include "tensorrt_llm_libutils.h"

int main(int argc, char* argv[])
{
    class TRTLogger : public nvinfer1::ILogger
    {
    public:
        void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
        {
            if (severity <= nvinfer1::ILogger::Severity::kERROR)
                std::cerr << "[TensorRT-LLM ERR]: " << msg << std::endl;
            else if (severity == nvinfer1::ILogger::Severity::kWARNING)
                std::cerr << "[TensorRT-LLM WARNING]: " << msg << std::endl;
            else
                std::cout << "[TensorRT-LLM LOG]: " << msg << std::endl;
        }
    };

    TRTLogger* trtLogger = new TRTLogger();

    std::string libname = "libtensorrt_llm_plugin.so";

    /* =============== initLibNvInferPlugins =============== */

    typedef bool (*initLibNvInferPlugins_sig)(void*, const void*);

    auto initLibNvInferPlugins = getTrtLLMFunction<initLibNvInferPlugins_sig>(
        /*libFileSoName=*/libname,
        /*symbol=*/"initLibNvInferPlugins");

    std::cout << std::endl;

    std::string libNamespace = "tensorrt_llm";
    const char* libNamespace_cstr = libNamespace.data();

    bool status1 = initLibNvInferPlugins(trtLogger, libNamespace_cstr);
    std::cout << "Success Status: " << status1 << std::endl << std::endl;

    bool status2 = initLibNvInferPlugins(trtLogger, libNamespace_cstr);
    std::cout << "Success Status: " << status2 << std::endl;

    /* =============== getInferLibVersion =============== */

    std::cout << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;

    typedef int32_t (*getInferLibVersion_sig)();

    auto getInferLibVersion = getTrtLLMFunction<getInferLibVersion_sig>(
        /*libFileSoName=*/libname,
        /*symbol=*/"getInferLibVersion");

    std::cout << std::endl;

    int32_t version = getInferLibVersion();
    std::cout << "Version: " << version << std::endl;

    return 0;
}
