# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_copy entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.elementwise import (  # noqa: E402, E501
    cute_dsl_copy as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

op = entry.cute_dsl_copy


@pytest.fixture(scope="module")
def compiled():
    return entry.compile()


def _sample(dtype, shape, g) -> torch.Tensor:
    if dtype.is_floating_point:
        x = torch.randn(shape, generator=g, device="cuda") * 1000
        flat = x.view(-1)
        n = min(3, flat.numel())
        flat[:n] = torch.tensor([float("nan"), float("inf"), -0.0], device="cuda")[:n]
        return x.to(dtype)
    if dtype is torch.int64:
        return torch.randint(-(2**31), 2**31, shape, generator=g, device="cuda", dtype=torch.int64)
    return torch.randint(-(2**31), 2**31 - 1, shape, generator=g, device="cuda").to(dtype)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled, certifying) -> None:  # noqa: F811 — fixture by name
    """Bit-exact against Tensor.copy_, NaN, inf and -0.0 included."""
    certifying(op)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    shape, sd, dd = cell.spec["shape"], cell.spec["src"], cell.spec["dst"]
    src = _sample(sd, shape, g)
    if cell.spec.get("strided"):
        wide = torch.zeros(shape[:-1] + (shape[-1] + 9,), dtype=sd, device="cuda")
        wide[..., 4 : 4 + shape[-1]] = src
        src = wide[..., 4 : 4 + shape[-1]]
        dst = torch.zeros(shape[:-1] + (shape[-1] + 5,), dtype=dd, device="cuda")[
            ..., 2 : 2 + shape[-1]
        ]
    else:
        dst = torch.zeros(shape, dtype=dd, device="cuda")
    assert entry.takes(dst, src)
    want = op.reference(dst, src, compiled)
    got = op(dst, src, compiled)
    assert got is dst
    assert torch.equal(dst.contiguous().view(torch.uint8), want.contiguous().view(torch.uint8))


def test_the_op_refuses_what_it_does_not_take(compiled) -> None:
    x = torch.zeros(4, 4, device="cuda")
    for dst, src in (
        (x, torch.zeros(4, device="cuda")),  # broadcast
        (x[:, 1:], x[:, :-1]),  # overlap
        (torch.zeros(4, dtype=torch.int32, device="cuda"), torch.zeros(4, device="cuda")),
    ):
        assert not entry.takes(dst, src)
        with pytest.raises(ValueError, match="not a pair this kernel takes"):
            op(dst, src, compiled)
