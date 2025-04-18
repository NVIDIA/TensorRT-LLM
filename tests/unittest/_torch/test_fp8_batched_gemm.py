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
from dataclasses import dataclass

import pytest
import torch
from utils.util import getSMVersion

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


@dataclass(frozen=True)
class Fp8BatchedGemmTestCase:
    b: int
    m: int
    n: int
    k: int
    dtypeC: torch.dtype
    ds: bool
    batch_m: bool
    tile_size: int


def deepSeekFp8ComputeBatchedGemmReference(
    B: int,
    mM: int,
    mN: int,
    mK: int,
    valsC: torch.Tensor,
    dqSfsC: torch.Tensor,
    valsA: torch.Tensor,
    dqSfsA: torch.Tensor,
    valsB: torch.Tensor,
    dqSfsB: torch.Tensor,
    batch_m: bool,
    quantizeOutput: bool,
    tileSize: int,
) -> None:

    if not batch_m:
        valsA, valsB = valsB, valsA
        dqSfsA, dqSfsB = dqSfsB, dqSfsA
        mM, mN = mN, mM

    for bi in range(B):
        sf_stride = mM * bi
        for mi in range(mM):
            for ni in range(0, mN, tileSize):
                acc = torch.zeros(tileSize, dtype=torch.float32)
                for nj in range(tileSize):
                    nk = ni + nj
                    for ki in range(0, mK, tileSize):
                        tmp = (valsA[bi, mi, ki:ki + tileSize]
                               @ valsB[bi, nk, ki:ki + tileSize])
                        dpSfA = dqSfsA[ki // tileSize, mi + sf_stride]
                        dpSfB = dqSfsB[bi, ni // tileSize, ki // tileSize]
                        acc[nj] += (dpSfA * dpSfB) * tmp

                aMax = 0
                for nj in range(tileSize):
                    aMax = max(aMax, abs(acc[nj]))
                E4m3MaxVal = 448
                if dqSfsC is not None:
                    dqSfsC[bi, ni // tileSize, mi] = aMax / E4m3MaxVal
                for nj in range(tileSize):
                    val = acc[nj]
                    if quantizeOutput:
                        val = val / aMax * E4m3MaxVal
                    valsC[bi, mi, ni + nj] = val


def fp8_bmm_reference(
    input_batch_a_device: torch.Tensor,
    m: int,
    input_batch_b_device: torch.Tensor,
    n: int,
    ds_per_input_a_scaling_factors_device: torch.Tensor,
    ds_per_input_b_scaling_factors_device: torch.Tensor,
    ds_per_output_scaling_factors_device: torch.Tensor,
    global_output_scale_device: torch.Tensor,
    dtypeC: torch.dtype,
    ds: bool,
    batch_m: bool,
    tile_size: int = 128,
) -> torch.Tensor:
    b, mPadded, k = input_batch_a_device.shape
    nPadded = input_batch_b_device.shape[1]
    assert input_batch_b_device.shape[2] == k
    assert k % tile_size == 0

    if batch_m:
        assert nPadded % tile_size == 0
    else:
        assert mPadded % tile_size == 0

    input_batch_a_host = input_batch_a_device.to(torch.float32).cpu()
    input_batch_b_host = input_batch_b_device.to(torch.float32).cpu()

    if ds:
        output_m = mPadded if batch_m else nPadded
        output_n = n if batch_m else m

        c = torch.zeros((b, output_m, output_n), dtype=torch.float32)

        a_scale = ds_per_input_a_scaling_factors_device.cpu()
        b_scale = ds_per_input_b_scaling_factors_device.cpu()
        c_scale = ds_per_output_scaling_factors_device.cpu()

        if dtypeC == torch.bfloat16:
            quantize = False
        elif dtypeC == torch.float8_e4m3fn:
            quantize = True
        else:
            raise NotImplementedError

        deepSeekFp8ComputeBatchedGemmReference(
            b,
            mPadded,
            nPadded,
            k,
            c,
            c_scale,
            input_batch_a_host,
            a_scale,
            input_batch_b_host,
            b_scale,
            batch_m,
            quantize,
            tile_size,
        )

    else:
        global_c_scale = global_output_scale_device.cpu()
        c = global_c_scale[:, None, None] * input_batch_a_host.bmm(
            input_batch_b_host.transpose(1, 2))
        if not batch_m:
            c = c.transpose(1, 2)

    c = c.narrow(1, 0, m) if batch_m else c.narrow(1, 0, n)
    return c.to(dtypeC)


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
            Fp8BatchedGemmTestCase(b=4,
                                   m=128,
                                   n=256,
                                   k=512,
                                   dtypeC=torch.float8_e4m3fn,
                                   ds=False,
                                   batch_m=True,
                                   tile_size=128),
            id="num_batches_4_128x256x512_batchM_e4m3_ts128",
        ),
        pytest.param(
            Fp8BatchedGemmTestCase(b=4,
                                   m=128,
                                   n=256,
                                   k=512,
                                   dtypeC=torch.float8_e4m3fn,
                                   ds=False,
                                   batch_m=False,
                                   tile_size=128),
            id="num_batches_4_128x256x512_batchN_e4m3_ts128",
        ),
        pytest.param(
            Fp8BatchedGemmTestCase(b=4,
                                   m=128,
                                   n=256,
                                   k=512,
                                   dtypeC=torch.float8_e4m3fn,
                                   ds=True,
                                   batch_m=True,
                                   tile_size=128),
            id="num_batches_4_128x256x512_batchM_e4m3_ds_ts128",
        ),
        pytest.param(
            Fp8BatchedGemmTestCase(b=4,
                                   m=128,
                                   n=256,
                                   k=512,
                                   dtypeC=torch.float8_e4m3fn,
                                   ds=True,
                                   batch_m=False,
                                   tile_size=128),
            id="num_batches_4_128x256x512_batchN_e4m3_ds_ts128",
        ),
        pytest.param(
            Fp8BatchedGemmTestCase(b=4,
                                   m=128,
                                   n=256,
                                   k=512,
                                   dtypeC=torch.bfloat16,
                                   ds=False,
                                   batch_m=True,
                                   tile_size=128),
            id="num_batches_4_128x256x512_batchM_bf16_ts128",
        ),
        pytest.param(
            Fp8BatchedGemmTestCase(b=4,
                                   m=128,
                                   n=256,
                                   k=512,
                                   dtypeC=torch.bfloat16,
                                   ds=False,
                                   batch_m=False,
                                   tile_size=128),
            id="num_batches_4_128x256x512_batchN_bf16_ts128",
        ),
        pytest.param(
            Fp8BatchedGemmTestCase(b=4,
                                   m=128,
                                   n=256,
                                   k=512,
                                   dtypeC=torch.bfloat16,
                                   ds=True,
                                   batch_m=True,
                                   tile_size=128),
            id="num_batches_4_128x256x512_batchM_bf16_ds_ts128",
        ),
        pytest.param(
            Fp8BatchedGemmTestCase(b=4,
                                   m=128,
                                   n=256,
                                   k=512,
                                   dtypeC=torch.bfloat16,
                                   ds=True,
                                   batch_m=False,
                                   tile_size=128),
            id="num_batches_4_128x256x512_batchN_bf16_ds_ts128",
        ),
        pytest.param(
            Fp8BatchedGemmTestCase(b=4,
                                   m=2,
                                   n=256,
                                   k=512,
                                   dtypeC=torch.float8_e4m3fn,
                                   ds=True,
                                   batch_m=True,
                                   tile_size=128),
            id="num_batches_4_2x256x512_batchM_e4m3_ds_ts128",
        ),
        pytest.param(
            Fp8BatchedGemmTestCase(b=4,
                                   m=128,
                                   n=2,
                                   k=512,
                                   dtypeC=torch.float8_e4m3fn,
                                   ds=True,
                                   batch_m=False,
                                   tile_size=128),
            id="num_batches_4_128x2x512_batchN_e4m3_ds_ts128",
        ),
    ],
)
def test_fp8_batched_gemm_trtllmgen(test_case: Fp8BatchedGemmTestCase) -> None:
    torch.random.manual_seed(42)

    b = test_case.b
    m = test_case.m
    n = test_case.n
    k = test_case.k
    ds = test_case.ds
    dtypeC = test_case.dtypeC
    batch_m = test_case.batch_m
    tile_size = test_case.tile_size

    if dtypeC == torch.bfloat16:
        quantize = False
    elif dtypeC == torch.float8_e4m3fn:
        quantize = True
    else:
        raise NotImplementedError

    input_batch_a_device = torch.randn(
        (b, m, k), device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    input_batch_b_device = torch.randn(
        (b, n, k), device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)

    ds_per_input_a_scaling_factors_device = None
    ds_per_input_b_scaling_factors_device = None
    ds_per_output_scaling_factors_device = None
    output_scaling_factor_device = None

    if ds:
        dqSfAShape = ((k // tile_size, ((m + tile_size - 1) // tile_size) *
                       tile_size * b) if batch_m else
                      (b, m // tile_size, k // tile_size))
        dqSfBShape = ((b, n // tile_size, k // tile_size) if batch_m else
                      (k // tile_size,
                       ((n + tile_size - 1) // tile_size) * tile_size * b))
        dqSfCShape = ((b, n // tile_size, ((m + tile_size - 1) // tile_size) *
                       tile_size) if batch_m else
                      (b, m // tile_size,
                       ((n + tile_size - 1) // tile_size) * tile_size))

        ds_per_input_a_scaling_factors_device = torch.randint(
            1, 8, dqSfAShape, device="cuda", dtype=torch.float32)

        ds_per_input_b_scaling_factors_device = torch.randint(
            1, 8, dqSfBShape, device="cuda", dtype=torch.float32)

        ds_per_output_scaling_factors_device = torch.ones(dqSfCShape,
                                                          device="cuda",
                                                          dtype=torch.float32)
    else:
        output_scaling_factor_device = torch.ones((b),
                                                  device="cuda",
                                                  dtype=torch.float32)

    # Padded data is written by the previous kernel
    if batch_m:
        if m % tile_size:
            tiled_shape = ((m + tile_size - 1) // tile_size) * tile_size
            input_batch_a_device = torch.nn.functional.pad(
                input_batch_a_device, (0, 0, 0, tiled_shape - m), "constant", 0)
    else:
        if n % tile_size:
            tiled_shape = ((n + tile_size - 1) // tile_size) * tile_size
            input_batch_b_device = torch.nn.functional.pad(
                input_batch_b_device, (0, 0, 0, tiled_shape - n), "constant", 0)

    c_expected = fp8_bmm_reference(
        input_batch_a_device,
        m,
        input_batch_b_device,
        n,
        ds_per_input_a_scaling_factors_device,
        ds_per_input_b_scaling_factors_device,
        ds_per_output_scaling_factors_device,
        output_scaling_factor_device,
        dtypeC,
        ds,
        batch_m,
        tile_size,
    )

    c_actual = torch.ops.trtllm.fp8_batched_gemm(
        input_batch_a_device,
        m,
        ds_per_input_a_scaling_factors_device,
        input_batch_b_device,
        n,
        ds_per_input_b_scaling_factors_device,
        ds_per_output_scaling_factors_device,
        output_scaling_factor_device,
        tile_size,
        quantize,
        ds,
        batch_m,
    )

    c_actual = c_actual.detach().cpu()

    torch.testing.assert_close(c_actual.to(torch.float32),
                               c_expected.to(torch.float32),
                               atol=1e-4,
                               rtol=1e-4)
