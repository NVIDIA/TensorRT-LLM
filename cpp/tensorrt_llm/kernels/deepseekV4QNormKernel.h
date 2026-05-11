#pragma once

#include "tensorrt_llm/common/config.h"

#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

void invokeDeepseekV4QNorm(
    void const* input, void* output, int totalRows, int headDim, bool isBfloat16, float eps, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
