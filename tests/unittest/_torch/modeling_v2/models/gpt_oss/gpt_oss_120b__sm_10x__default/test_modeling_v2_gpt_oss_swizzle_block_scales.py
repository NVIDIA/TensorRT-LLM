# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gpt-oss's own kernel: the MXFP4 block-scale swizzle of its weight loader.

Importing the target package loads ``libth_modeling_v2_gpt_oss.so`` through
``_native.load_target_ops``, as routing does: a pass here means the family library
was built for this GPU, packaged, opened, carries the target, and registered the
fake. The kernel only moves bytes, so it must match the plain-torch swizzle it
replaces bit for bit.
"""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tensorrt_llm._torch._experimental.modeling_v2.models.gpt_oss.gpt_oss_120b__sm_10x__default import (  # noqa: E501
    native_ops,
    weights,
)

assert torch.cuda.is_available(), "the gpt_oss family library needs a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] != 10, reason="built for sm_100f only"
)


def _reference(s: torch.Tensor) -> torch.Tensor:
    """The plain-torch swizzle the loader used before the kernel."""
    e, m, c = s.shape
    v = s.reshape(e, m // 128, 4, 32, c // 4, 4)
    return v.permute(0, 1, 4, 3, 2, 5).reshape(e, m, c).contiguous()


def _scales(experts: int, rows: int, cols: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(
        0, 256, (experts, rows, cols), generator=g, device="cuda", dtype=torch.uint8
    )


@pytest.mark.parametrize(
    "experts, rows, cols",
    [
        (1, 128, 4),  # one 128x4 tile
        (2, 256, 8),  # tiles in both directions
        (3, 384, 96),  # gpt-oss-120b FC2: H / 32 = 90 scales, padded to 96
        (4, 5888, 92),  # FC1 rows, many tiles
        (0, 128, 4),  # no experts
    ],
)
def test_the_kernel_matches_the_torch_swizzle(experts, rows, cols):
    s = _scales(experts, rows, cols, seed=rows * 131 + cols)
    expected = _reference(s)
    assert torch.equal(native_ops.swizzle_block_scales(s), expected)
    # The loader's call site, which the FC1 and FC2 scale preparation both use.
    assert torch.equal(weights._swizzle_scales(s), expected)


@pytest.mark.parametrize(
    "scales, match",
    [
        (lambda: _scales(1, 100, 4, 0), "rows % 128 == 0"),
        (lambda: _scales(1, 128, 6, 0), "cols % 4 == 0"),
        (lambda: _scales(1, 128, 4, 0).float(), "uint8"),
        (lambda: _scales(2, 128, 8, 0)[:, :, :4], "contiguous"),
        (lambda: _scales(1, 128, 4, 0).cpu(), "CUDA"),
    ],
    ids=["rows", "cols", "dtype", "strided", "host"],
)
def test_the_op_rejects_what_the_kernel_cannot_swizzle(scales, match):
    with pytest.raises(RuntimeError, match=match):
        native_ops.swizzle_block_scales(scales())


def test_the_fake_op_describes_the_output():
    with FakeTensorMode():
        s = torch.empty(2, 256, 8, device="cuda", dtype=torch.uint8)
        out = native_ops.swizzle_block_scales(s)
    assert out.shape == (2, 256, 8) and out.dtype == torch.uint8
