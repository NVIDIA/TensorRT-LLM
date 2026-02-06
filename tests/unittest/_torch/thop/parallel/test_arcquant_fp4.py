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


import pytest
import torch
import torch.nn.functional as F
from utils.util import skip_pre_blackwell

import tensorrt_llm  # noqa: F401


@skip_pre_blackwell
@pytest.mark.parametrize(
    "mnk", [(128, 4096, 4096), (128, 1024, 4096), (128, 12288, 4096), (128, 4096, 12288)]
)
def test_arcquant_fp4(mnk):
    try:
        import sys

        sys.path.append("/data/ARCQuant/kernels/build")
        import agemm
    except Exception:
        pytest.skip("ARCQuant not found, skip test.")

    M, N, K = mnk
    step = 256
    for i in range(4096 // step + 1):
        KE = step * i
        torch.manual_seed(45510)
        X = torch.rand(M, K, dtype=torch.bfloat16, device="cuda") - 0.5
        W = torch.rand(N, K, dtype=torch.bfloat16, device="cuda") - 0.5
        W = torch.load("/data/TensorRT-LLM-main/qwen3-8b-q-proj.pt").cuda()
        # reorder_index = torch.arange(K, dtype=torch.int16, device="cuda")
        reorder_index = torch.randperm(K, dtype=torch.int16, device="cuda")

        scale_w = torch.max(W.abs()) / (448.0 * 6.0)
        scale_x = torch.max(X.abs()) / (448.0 * 6.0)

        A, SFA = torch.ops.trtllm.fp4_quantize_with_reorder_residual(X / scale_x, reorder_index, KE)
        B, SFB = agemm.reorder_quantize_w(W / scale_w, reorder_index, KE)

        C = torch.ops.trtllm.nvfp4_gemm(A, B, SFA, SFB, (scale_x * scale_w).float(), torch.bfloat16)

        D = F.linear(X, W)
        assert F.cosine_similarity(C.flatten(), D.flatten(), dim=0).item() > 0.98
