#pragma once

#include "tensorrt_llm/common/config.h"

#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

void invokeDeepseekV4QNorm(
    void const* input, void* output, int totalRows, int headDim, bool isBfloat16, float eps, cudaStream_t stream);

// Fused variant: in one pass, performs per-row RMSNorm and writes:
//   * the first `nopeDim` columns of each normalized row as FP8E4M3 into `quant_q_nope`
//     (scaled by `*quant_scale_qkv_ptr` if non-null, otherwise 1.0f). The per-row
//     stride in bytes is `quantQNopeRowStrideBytes`; pass `nopeDim` for a packed
//     [totalRows, nopeDim] output buffer, or `headDim` to interleave with the rope
//     segment of a shared `[totalRows, headDim]` FP8 Q buffer (consumed by FMHA
//     after applyMLARopeAndAssignQKVKernelOptContext fills the rope slot).
//   * the remaining `headDim - nopeDim` columns as the input dtype into `q_pe_out`
//     (always packed [totalRows, headDim - nopeDim]).
// Eliminates the trailing bf16→FP8 quant_copy pass that follows q_b_layernorm in the
// MLA absorption-mode prefill path.
//
// Only headDim==512, nopeDim==448 is currently supported (DeepSeek-V4 absorption shape).
void invokeDeepseekV4QNormFusedFp8(void const* input, void* quant_q_nope, void* q_pe_out,
    void const* quant_scale_qkv_ptr, int totalRows, int headDim, int nopeDim, int quantQNopeRowStrideBytes,
    bool isBfloat16, float eps, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
