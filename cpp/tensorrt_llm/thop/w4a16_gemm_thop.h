#pragma once

#include <torch/extension.h>

torch::Tensor w4a16_gemm_op_forward(torch::Tensor A, torch::Tensor B_packed, torch::Tensor scales, torch::Tensor zeros,
    torch::Tensor bias, int64_t group_size);
