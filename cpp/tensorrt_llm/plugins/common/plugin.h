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
#ifndef TRT_PLUGIN_H
#define TRT_PLUGIN_H
#include "NvInferPlugin.h"
#include "tensorrt_llm/plugins/common/checkMacrosPlugin.h"
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

typedef enum
{
    STATUS_SUCCESS = 0,
    STATUS_FAILURE = 1,
    STATUS_BAD_PARAM = 2,
    STATUS_NOT_SUPPORTED = 3,
    STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

namespace nvinfer1
{

namespace pluginInternal
{

class BasePlugin : public IPluginV2
{
protected:
    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

    std::string mNamespace;
};

class BaseCreator : public IPluginCreator
{
public:
    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

protected:
    std::string mNamespace;
};

} // namespace pluginInternal

namespace plugin
{

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

inline int32_t getTrtSMVersionDec(int32_t smVersion)
{
    // Treat SM89 as SM86 temporarily.
    return (smVersion == 89) ? 86 : smVersion;
}

inline int32_t getTrtSMVersionDec(int32_t majorVersion, int32_t minorVersion)
{
    return getTrtSMVersionDec(majorVersion * 10 + minorVersion);
}

inline int32_t elementSize(DataType type) noexcept
{
    switch (type)
    {
    case DataType::kFLOAT: return 4;
    case DataType::kHALF: return 2;
    case DataType::kINT8: return 1;
    case DataType::kINT32: return 4;
    case DataType::kBOOL: return 1;
    case DataType::kUINT8: return 1;
    case DataType::kFP8: return 1;
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 9
    case DataType::kBF16: return 2;
    case DataType::kINT64: return 8;
#endif
    }
    PLUGIN_FAIL("unreachable code path");
}

int8_t* alignPtr(int8_t* ptr, uintptr_t to);

int8_t* nextWorkspacePtr(int8_t* const base, uintptr_t& offset, const uintptr_t size);

int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize);

size_t calculateTotalWorkspaceSize(size_t* workspaces, int count);

} // namespace plugin
} // namespace nvinfer1

inline bool isBuilding()
{
    std::string const& key = "IS_BUILDING";
    char* val = getenv(key.c_str());
    if (val == nullptr || std::string(val) != "1")
    {
        return false;
    }
    else
    {
        return true;
    }
}

#define MPICHECK(cmd)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        int e = cmd;                                                                                                   \
        if (e != MPI_SUCCESS)                                                                                          \
        {                                                                                                              \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);                                           \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

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

#endif // TRT_PLUGIN_H
