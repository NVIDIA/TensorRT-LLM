# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

import numpy as np
import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@pytest.mark.parametrize(
    "k_n",
    [(7168, 2112), (8192, 8192), (8192, 57344), (28672, 8192)],
)
@pytest.mark.parametrize(
    "m",
    [1, 8, 16],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
)
def test_cublas_mm(dtype, m, k_n):
    k, n = k_n
    torch.random.manual_seed(0)
    shape_x = (m, k)
    shape_w = (n, k)
    x = torch.randn(shape_x, device="cuda").to(dtype)
    w = torch.randn(shape_w, device="cuda").to(dtype)
    output = torch.ops.trtllm.cublas_mm(
        x,
        w.t(),
        bias=None,
    )
    ref = torch.matmul(x, w.t())
    np.testing.assert_allclose(ref.float().cpu(), output.float().cpu())


if __name__ == '__main__':
    test_cublas_mm(torch.float16, 12, (8192, 10240))
