# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch
import torch.nn.functional as F
from utils.util import getSMVersion


@pytest.mark.skipif(
    getSMVersion() not in (100, 103),
    reason="The large-M MXFP8 tactics are validated on SM100 and SM103 only. Current SM is %d."
    % getSMVersion(),
)
@pytest.mark.parametrize(
    "m,n,k",
    [
        (6599, 9216, 6144),
        (6599, 6144, 3072),
        (14906, 6144, 8192),
        (14906, 6144, 6144),
        (29765, 24576, 6144),
        (29765, 6144, 12288),
        (8193, 9216, 6144),
    ],
)
def test_mxfp8_mxfp8_gemm_large_m(m: int, n: int, k: int):
    """Selected tactics and an out-of-band fallback agree with BF16 GEMM."""
    torch.manual_seed(42)
    mat_a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    mat_b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

    fp8_a, a_block_sf = torch.ops.trtllm.mxfp8_quantize(mat_a, True)
    fp8_b, b_block_sf = torch.ops.trtllm.mxfp8_quantize(mat_b, True)
    global_scale = torch.ones((1,), device="cuda", dtype=torch.float32)

    output = torch.ops.trtllm.mxfp8_mxfp8_gemm(
        fp8_a,
        a_block_sf,
        fp8_b,
        b_block_sf,
        global_scale,
        torch.bfloat16,
    )
    output_ref = mat_a @ mat_b.t()

    assert F.cosine_similarity(output.flatten(), output_ref.flatten(), dim=0).item() > 0.98
