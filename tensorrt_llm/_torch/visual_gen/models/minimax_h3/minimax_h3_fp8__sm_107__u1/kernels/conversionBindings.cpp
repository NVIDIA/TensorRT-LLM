// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/core/Tensor.h>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

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

void castOut(at::Tensor input, at::Tensor output, int64_t mode, int64_t blocks, int64_t threads);
at::Tensor cast(at::Tensor input, int64_t mode, int64_t blocks, int64_t threads);
std::vector<int64_t> castInfo(int64_t mode, int64_t device);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    auto quant = module.def_submodule("quant");
    quant.def("out", &quantOut);
    quant.def("quantize", &quantize);
    quant.def("factors", &makeFactors);
    quant.def("kernel_info", &quantInfo);
    quant.def("launch_plan", &quantPlan);
    quant.def("out_config", &quantOutConfig);
    quant.def("quantize_config", &quantizeConfig);

    auto castModule = module.def_submodule("cast");
    castModule.def("out", &castOut);
    castModule.def("cast", &cast);
    castModule.def("kernel_info", &castInfo);
}
