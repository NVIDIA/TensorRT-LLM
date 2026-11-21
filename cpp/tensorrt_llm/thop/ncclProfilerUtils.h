/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda.h>
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext::detail
{

//! Observe capture state without consuming or overwriting the workload's Runtime error.
inline char const* getNcclCudaGraphState(cudaStream_t stream) noexcept
{
    if (cudaPeekAtLastError() != cudaSuccess)
    {
        return "unknown";
    }
    // Driver queries return errors directly without setting the Runtime's per-thread last error.
    CUstreamCaptureStatus captureStatus{};
    auto const captureError = cuStreamIsCapturing(stream, &captureStatus);
    return captureError != CUDA_SUCCESS                  ? "unknown"
        : captureStatus == CU_STREAM_CAPTURE_STATUS_NONE ? "eager"
                                                         : "capture";
}

} // namespace torch_ext::detail

TRTLLM_NAMESPACE_END
