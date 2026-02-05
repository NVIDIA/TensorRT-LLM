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

import sys

sys.path.append("/data/ARCQuant/kernels/build")

import agemm
import torch
import torch.nn.functional as F

# from utils.util import skip_pre_blackwell_unittest, unittest_name_func


def test_arcquant_fp4_linear():
    M, N, K = 128, 4096, 4096
    group = 16
    step = K // group
    for i in range(K // step + 1):
        KE = step * i
        _, KS, KO = K - 512, 512 - 128, 128
        torch.manual_seed(45510)
        signs = torch.randint(0, 2, (M, K), device="cuda", dtype=torch.bfloat16) * 2 - 1
        X = torch.rand(M, K, dtype=torch.bfloat16, device="cuda") * 3
        X[:, -KS:] = torch.rand(M, KS, dtype=torch.bfloat16, device="cuda") * 3 + 3
        X[:, -KO:] = torch.rand(M, KO, dtype=torch.bfloat16, device="cuda") * 8 + 8
        X[:, -16:] = torch.rand(M, 16, dtype=torch.bfloat16, device="cuda") * 32 + 32
        X = X * signs
        W = torch.rand(N, K, dtype=torch.bfloat16, device="cuda") * 3
        # W = torch.eye(K, dtype=torch.bfloat16, device='cuda')
        reorder_index = torch.arange(K, dtype=torch.int16, device="cuda")

        scale_w = torch.max(W.abs()) / (448.0 * 6.0)
        scale_x = torch.max(X.abs()) / (448.0 * 6.0)
        # scale_w = 1.0
        # scale_x = 1.0

        A, SFA = agemm.reorder_quantize_x(X / scale_x, reorder_index, KE)
        trt_A, trt_SFA = torch.ops.trtllm.fp4_quantize_with_reorder_residual(
            X / scale_x, reorder_index, KE
        )
        assert torch.allclose(A, trt_A)
        assert torch.allclose(SFA[: SFA.shape[0] // 2], trt_SFA)
        B, SFB = agemm.reorder_quantize_w(W / scale_w, reorder_index, KE)
        torch.cuda.synchronize()

        # C = agemm.matmul(A, B, SFA, SFB, scale_x * scale_w)
        C = torch.ops.trtllm.nvfp4_gemm(
            trt_A, B, trt_SFA, SFB[: SFB.shape[0] // 2], (scale_x * scale_w).float(), torch.bfloat16
        )
        torch.cuda.synchronize()

        D = F.linear(X, W)

        mse = F.mse_loss(D, C).item()
        print(f"MSE(k={KE:<4}): {mse:<15.8e}")


def test_arcquant_nvfp4_reorder_residual():
    k = 4096
    ke = 256
    m = 1
    X = torch.randn([m, k], dtype=torch.bfloat16).cuda()
    reorder_index = torch.arange(k, dtype=torch.int16).cuda()

    # Call the quantization function
    qx, sfx = torch.ops.trtllm.fp4_quantize_with_reorder_residual(X, reorder_index, ke)


if __name__ == "__main__":
    test_arcquant_fp4_linear()
