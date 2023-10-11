/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/stringUtils.h"

#include <mpi.h>

#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE

#define TLLM_MPI_CHECK(cmd, logger)                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        auto e = cmd;                                                                                                  \
        if (e != MPI_SUCCESS)                                                                                          \
        {                                                                                                              \
            (logger).log(nvinfer1::ILogger::Severity::kERROR,                                                          \
                tensorrt_llm::common::fmtstr("Failed: MPI error %s:%d '%d'", __FILE__, __LINE__, e).c_str());          \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#if ENABLE_MULTI_DEVICE
#define TLLM_NCCL_CHECK(cmd, logger)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        ncclResult_t r = cmd;                                                                                          \
        if (r != ncclSuccess)                                                                                          \
        {                                                                                                              \
            (logger).log(nvinfer1::ILogger::Severity::kERROR,                                                          \
                tensorrt_llm::common::fmtstr(                                                                          \
                    "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r))                      \
                    .c_str());                                                                                         \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
#endif // ENABLE_MULTI_DEVICE
