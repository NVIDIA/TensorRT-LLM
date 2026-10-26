# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the native-code fixture's scale_add entry.

Importing the entry imports its target package, which loads the fixture family
library through ``_native.load_target_ops``: a pass here means the library was
built for this GPU, packaged, opened, listed the target in its built_targets()
and registered its fake op -- before the first cell runs.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.fixture.scale_add__sm_80__default import (  # noqa: E501
    native_ops,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.fixture.scale_add__sm_80__default.ops.elementwise.scale_add import (  # noqa: E501
    scale_add as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "scale_add requires a CUDA device"

_ALPHA, _BETA = 1.5, -0.25


def _input(spec, why):
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(why.encode()))
    return torch.randn(spec["numel"], generator=g, device="cuda").to(spec["dtype"])


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    x = _input(cell.spec, cell.why)
    with validating(op):
        result = op(x, _ALPHA, _BETA)
    expected = op.reference(x, _ALPHA, _BETA)
    assert result.dtype == x.dtype and result.shape == x.shape
    # Both sides round the product and the sum to fp32 separately, then once to
    # the output dtype: the band is zero.
    assert torch.equal(result, expected)


def test_is_valid_rejects_a_strided_input() -> None:
    x = torch.randn(64, device="cuda")[::2]
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(x, _ALPHA, _BETA)


def test_the_op_rejects_an_unsupported_dtype() -> None:
    with pytest.raises(RuntimeError, match="float32 or bfloat16"):
        native_ops.scale_add(torch.ones(4, device="cuda", dtype=torch.float16), _ALPHA, _BETA)


def test_the_fake_op_describes_the_output() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(10, 3, device="cuda", dtype=torch.bfloat16)
        out = native_ops.scale_add(x, _ALPHA, _BETA)
    assert out.shape == (10, 3) and out.dtype == torch.bfloat16
