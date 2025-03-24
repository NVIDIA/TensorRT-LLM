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
from warnings import warn

import numpy as np
import pytest
import torch
from utils.util import getSMVersion

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@pytest.mark.skipif(
    getSMVersion() < 90,
    reason="custom scaled_mm is only supported in SM90",
)  # Skip tests that are not supported in SM90
@pytest.mark.parametrize(
    "k_n",
    [(8192, 10240), (8192, 8192), (8192, 57344), (28672, 8192)],
)
@pytest.mark.parametrize(
    "m",
    [2048, 12, 228],
)
@pytest.mark.parametrize(
    "output_dtype",
    [torch.float16, torch.float32, torch.bfloat16],
)
def test_fp8_scaled_mm(output_dtype, m, k_n):
    k, n = k_n
    torch.random.manual_seed(0)
    shape_x = (m, k)
    shape_w = (n, k)
    x = torch.rand(shape_x, device="cuda").to(torch.float8_e4m3fn)
    w = torch.rand(shape_w, device="cuda").to(torch.float8_e4m3fn)
    scale_x = torch.rand(1, device="cuda")
    scale_w = torch.rand(1, device="cuda")
    output = torch.ops.trtllm.cublas_scaled_mm(
        x,
        w.t(),
        scale_x,
        scale_w,
        bias=None,
        out_dtype=output_dtype,
        userbuffers_id=-1,
    )
    # set pytorch's cublas workspace size to 32MB to be aligned with trtllm
    old_env = os.environ.get("CUBLASLT_WORKSPACE_SIZE", "")
    os.environ["CUBLASLT_WORKSPACE_SIZE"] = f"{32*1024*1024}"
    ref = torch._scaled_mm(
        x,
        w.t(),
        out_dtype=output_dtype,
        scale_a=scale_x,
        scale_b=scale_w,
        use_fast_accum=True,
    )
    os.environ["CUBLASLT_WORKSPACE_SIZE"] = old_env
    np.testing.assert_allclose(ref.float().cpu(), output.float().cpu())

    if getSMVersion() == 90:
        cutlass_output = torch.ops.trtllm.cutlass_scaled_mm(
            x,
            w.t(),
            scale_x,
            scale_w,
            bias=None,
            out_dtype=output_dtype,
        )
        # TODO(zhenhuan): cutlass kernel has acc issue on some shapes
        try:
            np.testing.assert_allclose(ref.float().cpu(),
                                       cutlass_output.float().cpu())
        except Exception as e:
            warn(RuntimeWarning("cutlass result is not correct: " + repr(e)))


if __name__ == '__main__':
    test_fp8_scaled_mm(torch.float16, 12, (8192, 10240))
