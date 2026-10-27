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


def test_swizzle_block_scales():
    g = torch.Generator(device="cuda").manual_seed(0)
    # gpt-oss-120b's FC1 and FC2 scales (FC2's 90 scales per row padded to 96), and no experts.
    for shape in ((4, 5888, 92), (3, 384, 96), (0, 128, 4)):
        s = torch.randint(0, 256, shape, generator=g, device="cuda", dtype=torch.uint8)
        assert torch.equal(weights._swizzle_scales(s), _reference(s)), shape

    strided = torch.zeros(2, 128, 8, device="cuda", dtype=torch.uint8)[:, :, :4]
    with pytest.raises(RuntimeError, match="contiguous"):
        native_ops.swizzle_block_scales(strided)

    with FakeTensorMode():
        out = native_ops.swizzle_block_scales(
            torch.empty(2, 256, 8, device="cuda", dtype=torch.uint8)
        )
    assert out.shape == (2, 256, 8) and out.dtype == torch.uint8
