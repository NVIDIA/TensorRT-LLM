// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <vector>
void castOut(at::Tensor input, at::Tensor output, int64_t mode, int64_t blocks, int64_t threads);
at::Tensor cast(at::Tensor input, int64_t mode, int64_t blocks, int64_t threads);
std::vector<int64_t> castInfo(int64_t mode, int64_t device);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("out", &castOut);
    m.def("cast", &cast);
    m.def("kernel_info", &castInfo);
}
