/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/workspace.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/plugins/common/checkMacrosPlugin.h"

#include <NvInferRuntime.h>

#include <NvInferRuntimeBase.h>
#include <cstring>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <memory>
#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>

namespace tensorrt_llm::plugins
{

class BasePlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    [[nodiscard]] char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

protected:
    std::string mNamespace{api::kDefaultNamespace};
};

class BaseCreator : public nvinfer1::IPluginCreator
{
public:
    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    [[nodiscard]] char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

protected:
    std::string mNamespace{api::kDefaultNamespace};
};

// Write values into buffer
template <typename T>
void write(char*& buffer, const T& val)
{
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
void read(const char*& buffer, T& val)
{
    std::memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
}

inline size_t typeSize(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kBOOL: return 1UL;
    case nvinfer1::DataType::kFP8: return 1UL;
    case nvinfer1::DataType::kHALF: return 2UL;
    case nvinfer1::DataType::kBF16: return 2UL;
    case nvinfer1::DataType::kFLOAT: return 4UL;
    case nvinfer1::DataType::kINT8: return 1UL;
    case nvinfer1::DataType::kUINT8: return 1UL;
    case nvinfer1::DataType::kINT32: return 4UL;
    case nvinfer1::DataType::kINT64: return 8UL;
    }

    TLLM_THROW("Unknown dtype %d", static_cast<int>(type));
    return 0;
}

inline cudaDataType_t trtToCublasDtype(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT: return CUDA_R_32F;
    case nvinfer1::DataType::kHALF: return CUDA_R_16F;
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 9
    case nvinfer1::DataType::kBF16: return CUDA_R_16BF;
#endif
    default: TLLM_THROW("Not supported data type for cuBLAS");
    }
}

// Like std::unique_ptr, but does not prevent generation of default copy constructor when used as class members.
// The copy constructor produces nullptr. So the plugin default copy constructor will not really copy this, and
// your clone() implementation is responsible for initializing such data members.
// With this we can simplify clone() implementation when there are many data members including at least one unique_ptr.
template <typename T, typename Del = std::default_delete<T>>
class UniqPtrWNullCopy : public std::unique_ptr<T, Del>
{
public:
    using std::unique_ptr<T, Del>::unique_ptr;

    // for compatibility with std::make_unique
    explicit UniqPtrWNullCopy(std::unique_ptr<T, Del>&& src)
        : std::unique_ptr<T, Del>::unique_ptr{std::move(src)}
    {
    }

    // copy constructor produces nullptr
    UniqPtrWNullCopy(UniqPtrWNullCopy const&)
        : std::unique_ptr<T, Del>::unique_ptr{}
    {
    }
};

} // namespace tensorrt_llm::plugins

inline bool isBuilding()
{
    auto constexpr key = "IS_BUILDING";
    auto const val = getenv(key);
    return val != nullptr && std::string(val) == "1";
}

#if ENABLE_MULTI_DEVICE
#define NCCLCHECK(cmd)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        ncclResult_t r = cmd;                                                                                          \
        if (r != ncclSuccess)                                                                                          \
        {                                                                                                              \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

std::unordered_map<nvinfer1::DataType, ncclDataType_t>* getDtypeMap();

std::map<std::set<int>, ncclComm_t>* getCommMap();

void initCommMap(std::set<int> const& group);
#endif // ENABLE_MULTI_DEVICE

//! To save GPU memory, all the plugins share the same cublas and cublasLt handle globally.
//! Get cublas and cublasLt handle for current cuda context
std::shared_ptr<cublasHandle_t> getCublasHandle();
std::shared_ptr<cublasLtHandle_t> getCublasLtHandle();

#ifndef DEBUG

#define PLUGIN_CHECK(status)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        if (status != 0)                                                                                               \
            abort();                                                                                                   \
    } while (0)

#define ASSERT_PARAM(exp)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
            return STATUS_BAD_PARAM;                                                                                   \
    } while (0)

#define ASSERT_FAILURE(exp)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
            return STATUS_FAILURE;                                                                                     \
    } while (0)

#define CSC(call, err)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = call;                                                                                 \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            return err;                                                                                                \
        }                                                                                                              \
    } while (0)

#define DEBUG_PRINTF(...)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)

#else

#define ASSERT_PARAM(exp)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
        {                                                                                                              \
            fprintf(stderr, "Bad param - " #exp ", %s:%d\n", __FILE__, __LINE__);                                      \
            return STATUS_BAD_PARAM;                                                                                   \
        }                                                                                                              \
    } while (0)

#define ASSERT_FAILURE(exp)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(exp))                                                                                                    \
        {                                                                                                              \
            fprintf(stderr, "Failure - " #exp ", %s:%d\n", __FILE__, __LINE__);                                        \
            return STATUS_FAILURE;                                                                                     \
        }                                                                                                              \
    } while (0)

#define CSC(call, err)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = call;                                                                                 \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            printf("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus));                        \
            return err;                                                                                                \
        }                                                                                                              \
    } while (0)

#define PLUGIN_CHECK(status)                                                                                           \
    {                                                                                                                  \
        if (status != 0)                                                                                               \
        {                                                                                                              \
            DEBUG_PRINTF("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(status));                      \
            abort();                                                                                                   \
        }                                                                                                              \
    }

#define DEBUG_PRINTF(...)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        printf(__VA_ARGS__);                                                                                           \
    } while (0)

#endif // DEBUG

// Init with O(n) and retrieve with O(1)
class PluginFieldParser
{
public:
    // field array must remain valid when calling getScalar() later.
    PluginFieldParser(int32_t nbFields, nvinfer1::PluginField const* fields);
    // delete to remind accidental mis-use (copy) which may result in false-alarm warnings about unused fields.
    PluginFieldParser(PluginFieldParser const&) = delete;
    PluginFieldParser& operator=(PluginFieldParser const&) = delete;
    // check if all fields are retrieved and emit warning if some of them are not.
    ~PluginFieldParser();
    template <typename T>
    std::optional<T> getScalar(std::string_view const& name);

private:
    nvinfer1::PluginField const* mFields;

    struct Record
    {
        Record(int32_t idx)
            : index{idx}
        {
        }

        int32_t const index;
        bool retrieved{false};
    };

    std::unordered_map<std::string_view, Record> mMap;
};
