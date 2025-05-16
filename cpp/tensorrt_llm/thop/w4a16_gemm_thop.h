#pragma once

#include <torch/extension.h>

torch::Tensor w4a16_gemm_op_forward(torch::Tensor A, // Activation (FP16)
    torch::Tensor B_packed,                          // Pre-processed & packed INT4 Weights (as int8)
    torch::Tensor scales,                            // Scales (FP16)
    torch::Tensor zeros,                             // Zeros (FP16) - Can be an empty tensor if not used by QuantMode
    torch::Tensor bias,                              // Bias (FP16) - Can be an empty tensor if no bias
    int64_t group_size);
