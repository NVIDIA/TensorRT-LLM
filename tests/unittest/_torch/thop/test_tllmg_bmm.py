# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from dataclasses import dataclass

import pytest
import torch
from utils.util import getSMVersion

from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm.quantization.utils.fp4_utils import shuffle_matrix_a

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# TODO: add test for fp4


@dataclass(frozen=True)
class BatchedGemmTestCase:
    b: int
    m: int
    n: int
    k: int
    dtype_c: torch.dtype
    low_latency: bool
    tile_size: int
    use_deep_seek_fp8: bool = False


def bmm_deep_seek_fp8_reference(
    B: int,
    M: int,
    N: int,
    K: int,
    a: torch.Tensor,
    b: torch.Tensor,
    dq_sf_a: torch.Tensor,
    dq_sf_b: torch.Tensor,
    tile_size: int,
) -> torch.Tensor:
    c = torch.zeros((B, M, N), dtype=torch.float32)

    for bi in range(B):
        sf_stride = M * bi
        for mi in range(M):
            for ni in range(0, N, tile_size):
                acc = torch.zeros(tile_size, dtype=torch.float32)
                for nj in range(tile_size):
                    nk = ni + nj
                    for ki in range(0, K, tile_size):
                        tmp = (a[bi, mi, ki:ki + tile_size]
                               @ b[bi, nk, ki:ki + tile_size])
                        dp_sf_a = dq_sf_a[ki // tile_size, mi + sf_stride]
                        dp_sf_b = dq_sf_b[bi, ni // tile_size, ki // tile_size]
                        acc[nj] += (dp_sf_a * dp_sf_b) * tmp

                for nj in range(tile_size):
                    c[bi, mi, ni + nj] = acc[nj]
    return c


def fp8_bmm_reference(a_fp8: torch.Tensor, b_fp8: torch.Tensor,
                      dq_sf_a: torch.Tensor, dq_sf_b: torch.Tensor,
                      global_dq_a: torch.Tensor, global_dq_b: torch.Tensor,
                      dtype_c: torch.dtype,
                      use_deep_seek_fp8: bool) -> torch.Tensor:
    b, m, k = a_fp8.shape
    n = b_fp8.shape[1]
    quantization_tile_size = 128
    assert b_fp8.shape[2] == k
    assert k % quantization_tile_size == 0
    assert n % quantization_tile_size == 0

    a_fp8_host = a_fp8.clone().to(torch.float32).cpu()
    b_fp8_host = b_fp8.clone().to(torch.float32).cpu()

    if use_deep_seek_fp8:
        dq_sf_a_host = dq_sf_a.cpu()
        dq_sf_b_host = dq_sf_b.cpu()
        if dtype_c not in [torch.float16, torch.bfloat16, torch.float8_e4m3fn]:
            raise NotImplementedError

        c_fp32 = bmm_deep_seek_fp8_reference(
            b,
            m,
            n,
            k,
            a_fp8_host,
            b_fp8_host,
            dq_sf_a_host,
            dq_sf_b_host,
            quantization_tile_size,
        )
    else:
        scaling_factor_host = (global_dq_a.view(b, 1, 1) *
                               global_dq_b.view(b, 1, 1)).cpu()
        c_fp8fp8 = torch.bmm(a_fp8_host, b_fp8_host.transpose(1, 2))
        c_fp8fp8 *= scaling_factor_host
        c_fp32 = c_fp8fp8

    if dtype_c == torch.float8_e4m3fn:
        if use_deep_seek_fp8:
            return quant_ds_fp8(c_fp32, activations=True)
        else:
            return quant_fp8(c_fp32)
    return c_fp32


def quant_ds_fp8(x_fp32: torch.Tensor,
                 activations: bool) -> tuple[torch.Tensor, torch.Tensor]:
    # Get tensor dimensions
    b, m, k = x_fp32.shape

    m_stride = 1 if activations else 128
    # Calculate number of blocks
    num_m_blocks = (m + m_stride - 1) // m_stride
    num_k_blocks = (k + 128 - 1) // 128

    # Initialize max values tensor
    max_vals_shape = (num_k_blocks, b, m) if activations else (b, num_m_blocks,
                                                               num_k_blocks)
    max_vals = torch.zeros(max_vals_shape,
                           dtype=torch.float32,
                           device=x_fp32.device)

    # Scale to E4M3 range and convert
    E4M3_MAX = 448.0
    # Compute max for each block
    for bi in range(b):
        for mi in range(num_m_blocks):
            for ki in range(num_k_blocks):
                start_m = mi * m_stride
                end_m = min(start_m + m_stride, m)
                start_k = ki * 128
                end_k = min(start_k + 128, k)
                # Get absolute max over the m_stride x 128 block
                max_vals_index = (ki, bi, mi) if activations else (bi, mi, ki)
                max_vals[max_vals_index] = torch.abs(
                    x_fp32[bi, start_m:end_m, start_k:end_k]).max()
                x_fp32[bi, start_m:end_m,
                       start_k:end_k] *= (E4M3_MAX / max_vals[max_vals_index])

    # Convert to fp8
    dq_sfs = max_vals / E4M3_MAX
    x_fp8 = x_fp32.to(torch.float8_e4m3fn)

    return x_fp8, dq_sfs


def quant_fp8(x_fp32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Get batch size
    b = x_fp32.shape[0]

    # Initialize outputs
    amax = torch.zeros(b, device=x_fp32.device)

    # Process each batch
    for bi in range(b):
        amax[bi] = x_fp32[bi].abs().max()

    # Calculate scales
    E4M3_MAX = 448.0
    scale = amax / E4M3_MAX

    # Scale and convert to fp8, broadcasting the scale across m,k dimensions
    x_fp8 = (x_fp32 / scale.view(b, 1, 1)).to(torch.float8_e4m3fn)

    return x_fp8, scale


@pytest.mark.skipif(
    getSMVersion() not in [100],
    reason=
    "The kernel is only supported with compute capability 100. Current compute capability is %d."
    % getSMVersion(),
)
@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(
            BatchedGemmTestCase(b=4,
                                m=8,
                                n=256,
                                k=512,
                                dtype_c=torch.float8_e4m3fn,
                                use_deep_seek_fp8=False,
                                low_latency=True,
                                tile_size=8),
            id="num_batches_4_8x256x512_ll_e4m3_ts8",
        ),
        pytest.param(
            BatchedGemmTestCase(b=4,
                                m=128,
                                n=256,
                                k=512,
                                dtype_c=torch.float8_e4m3fn,
                                use_deep_seek_fp8=True,
                                low_latency=True,
                                tile_size=8),
            id="num_batches_4_128x256x512_ll_e4m3_ds_ts8",
        ),
        pytest.param(
            BatchedGemmTestCase(b=4,
                                m=128,
                                n=256,
                                k=512,
                                dtype_c=torch.bfloat16,
                                use_deep_seek_fp8=False,
                                low_latency=True,
                                tile_size=8),
            id="num_batches_4_128x256x512_ll_bf16_ts8",
        ),
        pytest.param(
            BatchedGemmTestCase(b=4,
                                m=128,
                                n=256,
                                k=512,
                                dtype_c=torch.bfloat16,
                                use_deep_seek_fp8=True,
                                low_latency=True,
                                tile_size=8),
            id="num_batches_4_128x256x512_ll_bf16_ds_ts8",
        ),
        pytest.param(
            BatchedGemmTestCase(b=4,
                                m=2,
                                n=128,
                                k=512,
                                dtype_c=torch.float8_e4m3fn,
                                use_deep_seek_fp8=True,
                                low_latency=True,
                                tile_size=8),
            id="num_batches_4_128x2x512_ll_e4m3_ds_ts8",
        ),
    ],
)
class TestFP8BatchedGemmTRTLLMGen:

    def test_thop(self, test_case: BatchedGemmTestCase) -> None:
        torch.random.manual_seed(42)

        b = test_case.b
        m = test_case.m
        n = test_case.n
        k = test_case.k
        use_deep_seek_fp8 = test_case.use_deep_seek_fp8
        dtype_c = test_case.dtype_c
        low_latency = test_case.low_latency
        tile_size = test_case.tile_size

        a_fp32 = torch.randn((b, m, k), device="cuda", dtype=torch.float32)
        b_fp32 = torch.randn((b, n, k), device="cuda", dtype=torch.float32)
        # Pad to the tile size. It is needed for the TRT-LLM Gen BMM input requirements.
        if m % tile_size:
            tiled_shape = ((m + tile_size - 1) // tile_size) * tile_size
            a_fp32 = torch.nn.functional.pad(a_fp32, (0, 0, 0, tiled_shape - m),
                                             "constant", 0)
        m_padded = ((m + tile_size - 1) // tile_size) * tile_size

        dq_sf_a = None
        dq_sf_b = None
        global_dq_a = None
        global_dq_b = None
        out_global_scaling_factor = None

        if use_deep_seek_fp8:
            a_fp8, dq_sf_a = quant_ds_fp8(a_fp32, activations=True)
            dq_sf_a = dq_sf_a.reshape(k // 128, -1).contiguous()
            b_fp8, dq_sf_b = quant_ds_fp8(b_fp32, activations=False)
            dq_sf_b = dq_sf_b.contiguous()
        else:
            a_fp8, global_dq_a = quant_fp8(a_fp32)
            b_fp8, global_dq_b = quant_fp8(b_fp32)
            out_global_scaling_factor = global_dq_a * global_dq_b

        # Compute reference batched matrix multiplication
        output = fp8_bmm_reference(a_fp8, b_fp8, dq_sf_a, dq_sf_b, global_dq_a,
                                   global_dq_b, dtype_c, use_deep_seek_fp8)

        c_dq_sf_ref = None
        if dtype_c == torch.float8_e4m3fn:
            if use_deep_seek_fp8:
                c_ref, c_dq_sf_ref = output
                c_dq_sf_ref = c_dq_sf_ref.reshape(n // 128, -1)
            else:
                c_ref, c_scale = output
                out_global_scaling_factor /= c_scale.cuda()
        else:
            c_ref = output

        epilogue_tile_m = 64 if use_deep_seek_fp8 else 128
        if low_latency and not use_deep_seek_fp8:
            b_fp8_shuffled = []
            for bi in range(b):
                b_fp8_shuffled.append(
                    shuffle_matrix_a(b_fp8[bi].view(torch.uint8).clone(),
                                     epilogue_tile_m))

            # Stack weights for all experts
            b_fp8 = torch.stack(b_fp8_shuffled).view(torch.float8_e4m3fn)

        if not use_deep_seek_fp8:
            out_global_scaling_factor = out_global_scaling_factor.contiguous(
            ).to(torch.float32)

        c_actual, c_dq_sf = torch.ops.trtllm.fp8_batched_gemm_trtllmgen(
            a_fp8.contiguous(),
            b_fp8.contiguous(),
            tile_size=tile_size,
            epilogue_tile_m=epilogue_tile_m,
            use_deep_seek_fp8=use_deep_seek_fp8,
            low_latency=low_latency,
            out_dtype=dtype_c,
            dq_sfs_a=dq_sf_a,
            dq_sfs_b=dq_sf_b,
            scale_c=out_global_scaling_factor)

        c_actual = c_actual.detach().cpu()
        c_ref = c_ref.detach().cpu()

        torch.testing.assert_close(c_actual.to(torch.float32)[:, :m],
                                   c_ref.to(torch.float32)[:, :m],
                                   atol=1e-2,
                                   rtol=1e-2)
        if use_deep_seek_fp8 and dtype_c == torch.float8_e4m3fn:
            c_dq_sf = c_dq_sf.detach().cpu()
            for bi in range(b):
                torch.testing.assert_close(
                    c_dq_sf[:,
                            bi * m_padded:bi * m_padded + m].to(torch.float32),
                    c_dq_sf_ref[:, bi * m_padded:bi * m_padded + m].to(
                        torch.float32),
                    atol=1e-2,
                    rtol=1e-2)

    def test_autotuned_thop(self, test_case: BatchedGemmTestCase) -> None:
        torch.random.manual_seed(42)

        b = test_case.b
        m = test_case.m
        n = test_case.n
        k = test_case.k
        use_deep_seek_fp8 = test_case.use_deep_seek_fp8
        dtype_c = test_case.dtype_c
        low_latency = test_case.low_latency
        tile_size = test_case.tile_size

        a_fp32 = torch.randn((b, m, k), device="cuda", dtype=torch.float32)
        b_fp32 = torch.randn((b, n, k), device="cuda", dtype=torch.float32)
        # Pad to the tile size. It is needed for the TRT-LLM Gen BMM input requirements.
        if m % tile_size:
            tiled_shape = ((m + tile_size - 1) // tile_size) * tile_size
            a_fp32 = torch.nn.functional.pad(a_fp32, (0, 0, 0, tiled_shape - m),
                                             "constant", 0)
        m_padded = ((m + tile_size - 1) // tile_size) * tile_size

        dq_sf_a = None
        dq_sf_b = None
        global_dq_a = None
        global_dq_b = None
        out_global_scaling_factor = None

        if use_deep_seek_fp8:
            a_fp8, dq_sf_a = quant_ds_fp8(a_fp32, activations=True)
            dq_sf_a = dq_sf_a.reshape(k // 128, -1).contiguous()
            b_fp8, dq_sf_b = quant_ds_fp8(b_fp32, activations=False)
            dq_sf_b = dq_sf_b.contiguous()
        else:
            a_fp8, global_dq_a = quant_fp8(a_fp32)
            b_fp8, global_dq_b = quant_fp8(b_fp32)
            out_global_scaling_factor = global_dq_a * global_dq_b

        # Compute reference batched matrix multiplication
        output = fp8_bmm_reference(a_fp8, b_fp8, dq_sf_a, dq_sf_b, global_dq_a,
                                   global_dq_b, dtype_c, use_deep_seek_fp8)

        c_dq_sf_ref = None
        if dtype_c == torch.float8_e4m3fn:
            if use_deep_seek_fp8:
                c_ref, c_dq_sf_ref = output
                c_dq_sf_ref = c_dq_sf_ref.reshape(n // 128, -1)
            else:
                c_ref, c_scale = output
                out_global_scaling_factor /= c_scale.cuda()
        else:
            c_ref = output

        epilogue_tile_m = 64 if use_deep_seek_fp8 else 128
        if low_latency and not use_deep_seek_fp8:
            b_fp8_shuffled = []
            for bi in range(b):
                b_fp8_shuffled.append(
                    shuffle_matrix_a(b_fp8[bi].view(torch.uint8).clone(),
                                     epilogue_tile_m))

            # Stack weights for all experts
            b_fp8 = torch.stack(b_fp8_shuffled).view(torch.float8_e4m3fn)

        if not use_deep_seek_fp8:
            out_global_scaling_factor = out_global_scaling_factor.contiguous(
            ).to(torch.float32)

        with autotune():
            c_actual, c_dq_sf = torch.ops.trtllm.fp8_batched_gemm_trtllmgen(
                a_fp8.contiguous(),
                b_fp8.contiguous(),
                tile_size=tile_size,
                epilogue_tile_m=epilogue_tile_m,
                use_deep_seek_fp8=use_deep_seek_fp8,
                low_latency=low_latency,
                out_dtype=dtype_c,
                dq_sfs_a=dq_sf_a,
                dq_sfs_b=dq_sf_b,
                scale_c=out_global_scaling_factor)

        c_actual = c_actual.detach().cpu()
        c_ref = c_ref.detach().cpu()

        torch.testing.assert_close(c_actual.to(torch.float32)[:, :m],
                                   c_ref.to(torch.float32)[:, :m],
                                   atol=1e-2,
                                   rtol=1e-2)
        if use_deep_seek_fp8 and dtype_c == torch.float8_e4m3fn:
            c_dq_sf = c_dq_sf.detach().cpu()
            for bi in range(b):
                torch.testing.assert_close(
                    c_dq_sf[:,
                            bi * m_padded:bi * m_padded + m].to(torch.float32),
                    c_dq_sf_ref[:, bi * m_padded:bi * m_padded + m].to(
                        torch.float32),
                    atol=1e-2,
                    rtol=1e-2)
