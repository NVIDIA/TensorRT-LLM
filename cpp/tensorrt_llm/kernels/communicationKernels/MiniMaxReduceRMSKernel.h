#pragma once
#include "tensorrt_llm/common/assert.h"
#include <NvInferRuntime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/runtime/ipcUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::minimax_ar
{
template <typename DType>
struct ElemsPerAccess;

template <>
struct ElemsPerAccess<half>
{
    static constexpr int value = 8;
};

template <>
struct ElemsPerAccess<nv_bfloat16>
{
    static constexpr int value = 8;
};

template <>
struct ElemsPerAccess<float>
{
    static constexpr int value = 4;
};

template <typename DType>
static constexpr int kElemsPerAccess = ElemsPerAccess<DType>::value;

struct MiniMaxReduceRMSParams
{
    int nranks{};
    int rank{};
    nvinfer1::DataType dtype;
    int size{};
    int hidden_dim{};
    void** workspace{};
    void* allreduce_in{};
    void* rms_norm_out{};
    void* rms_gamma{};
    float rms_eps{};
    float* scale_factor{};
    cudaStream_t stream{};
    bool trigger_completion_at_end = true;
};

void allreduce_fusion_op(AllReduceFusionParams const& params);

} // namespace kernels::minimax_ar

TRTLLM_NAMESPACE_END
