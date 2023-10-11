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

#include <memory>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "NvInferRuntimeBase.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/kernels/customAllReduceKernels.h"

namespace tensorrt_llm
{

class CustomAllReduceComm
{
public:
    CustomAllReduceComm(size_t TPSize, size_t PPSize, int deviceId, size_t bufferSize);
    ~CustomAllReduceComm();

    void customAllReduce(
        void* data, size_t elts, size_t size_per_elem, nvinfer1::DataType dataType, cudaStream_t stream);
    void* getShareBuffer();

    static bool isAvailable();

private:
    void setP2P(bool activate = true);

    void IpcGetMemHandle();
    void IpcOpenMemHandle();
    void IpcCloseMemHandle();
    void IpcSyncMemHandle();

    void allocate();

    kernels::AllReduceParams param_;
    bool is_ipc_handle_opened_ = false;
    size_t mTPSize;
    size_t mTPRank;
    size_t mPPSize;
    size_t mPPRank;
    size_t mBufferSize;
    int mDeviceId;
    mpi::MpiComm group_comm_;
};

template <typename T>
struct CustomARCommTypeConverter
{
    using Type = uint32_t;
};

template <>
struct CustomARCommTypeConverter<half>
{
    using Type = uint16_t;
};

#ifdef ENABLE_BF16
template <>
struct CustomARCommTypeConverter<__nv_bfloat16>
{
    using Type = __nv_bfloat16;
};
#endif

} // namespace tensorrt_llm
