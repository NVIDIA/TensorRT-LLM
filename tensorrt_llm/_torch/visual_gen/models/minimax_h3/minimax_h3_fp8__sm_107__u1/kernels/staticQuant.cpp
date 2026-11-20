// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <vector>

void quantOut(at::Tensor input, at::Tensor scale, at::Tensor factor, at::Tensor output, int64_t mode, bool pdl);
at::Tensor quantize(at::Tensor input, at::Tensor scale, at::Tensor factor, int64_t mode, bool pdl);
at::Tensor makeFactors(at::Tensor scales, bool exact_rn, bool pdl);
std::vector<int64_t> quantInfo(int64_t mode, int64_t device);
std::vector<int64_t> quantPlan(int64_t numel, bool pdl);

void quantOutConfig(at::Tensor input, at::Tensor scale, at::Tensor factor, at::Tensor output, int64_t mode, bool pdl,
    int64_t blocks, int64_t threads);
at::Tensor quantizeConfig(
    at::Tensor input, at::Tensor scale, at::Tensor factor, int64_t mode, bool pdl, int64_t blocks, int64_t threads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("out", &quantOut);
    m.def("quantize", &quantize);
    m.def("factors", &makeFactors);
    m.def("kernel_info", &quantInfo);
    m.def("launch_plan", &quantPlan);
    m.def("out_config", &quantOutConfig);
    m.def("quantize_config", &quantizeConfig);
}
