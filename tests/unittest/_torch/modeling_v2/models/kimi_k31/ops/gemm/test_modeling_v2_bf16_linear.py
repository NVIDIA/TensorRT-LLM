# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``bf16_linear`` is the FP32 product rounded to BF16.

The median certified cell of the cuBLAS branch and of the default one runs through ``bf16_linear`` (its guard on),
with the strided input where the target passes a column slice, against the entry's FP32 reference within the BF16
band of the other GEMM entries' tests.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.bf16_linear import bf16_linear

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

pytestmark = on_sm(103, 107)

BF16 = torch.bfloat16


def _fill(*tensors: torch.Tensor, seed: int) -> None:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for t in tensors:
        t.data.copy_((torch.randn(t.shape, generator=gen) * 0.02).to(t.dtype))


def _x(rows: int, k: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(rows, k, generator=gen, device="cuda").to(BF16)


CELLS = sample_cells(bf16_linear.CELLS, size_key="m", group_keys=("cublas",))[1::3]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why for c in CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    m, n, k, cublas = spec["m"], spec["n"], spec["k"], spec["cublas"]
    weight = torch.empty(n, k, dtype=BF16, device="cuda")
    seed = zlib.crc32(cell.why.encode()) % 2**31
    _fill(weight, seed=seed)
    x = _x(m, spec["row_stride"], seed=seed + 1)[:, :k]
    with validating(bf16_linear):
        actual = bf16_linear(x, weight, cublas)
    assert_within_ulp(
        actual, bf16_linear.reference(x, weight, cublas), element_ulp=2.0, rms_ulp=0.5
    )
