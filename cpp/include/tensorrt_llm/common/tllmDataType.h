/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "tensorrt_llm/common/config.h"

#include <cstdint>

//! \file tllmDataType.h
//!
//! Standalone, TensorRT-free runtime types that replace the few \c nvinfer1
//! types the shared C++ core historically used as common currency:
//! \c tensorrt_llm::DataType, \c tensorrt_llm::Dims and \c tensorrt_llm::ILogger.
//! These are defined here so the retained tree (runtime, batch manager, executor,
//! kernels and the nanobind bridge) compiles and links without the TensorRT
//! library.
//!
//! The \c DataType enumerator values intentionally mirror the legacy
//! \c nvinfer1::DataType integer values so that previously-serialized executor
//! configs and KV-cache metadata remain byte-compatible. The \c Dims layout
//! mirrors the legacy \c nvinfer1::Dims (\c int32_t \c nbDims followed by
//! \c int64_t \c d[8]) for the same reason.

TRTLLM_NAMESPACE_BEGIN

//! \brief Standalone data-type enum. Values mirror the legacy
//! \c nvinfer1::DataType for serialization/format compatibility.
enum class DataType : int32_t
{
    kFLOAT = 0,
    kHALF = 1,
    kINT8 = 2,
    kINT32 = 3,
    kBOOL = 4,
    kUINT8 = 5,
    kFP8 = 6,
    kBF16 = 7,
    kINT64 = 8,
    kINT4 = 9,
    kFP4 = 10,
    kE8M0 = 11,
};

//! \brief Standalone dimensions type. Layout mirrors the legacy
//! \c nvinfer1::Dims (rank plus up to \c MAX_DIMS 64-bit extents) so serialized
//! shapes remain compatible.
class Dims
{
public:
    //! The maximum rank (number of dimensions) supported for a tensor.
    static constexpr int32_t MAX_DIMS{8};

    //! The rank (number of dimensions).
    int32_t nbDims;

    //! The extent of each dimension.
    int64_t d[MAX_DIMS];
};

//! \brief Severity levels for the internal logger, mirroring the legacy
//! \c nvinfer1::ILogger::Severity values.
enum class ILoggerSeverity : int32_t
{
    kINTERNAL_ERROR = 0,
    kERROR = 1,
    kWARNING = 2,
    kINFO = 3,
    kVERBOSE = 4,
};

//! \brief Minimal, TensorRT-free logger interface replacing
//! \c nvinfer1::ILogger for the retained C++ tree.
class ILogger
{
public:
    using Severity = ILoggerSeverity;

    virtual void log(Severity severity, char const* msg) noexcept = 0;

    virtual ~ILogger() = default;
};

TRTLLM_NAMESPACE_END
