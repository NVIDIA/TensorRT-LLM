# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``rmsnorm_release_at_launch`` entry (the K3 q_a RMSNorm with its PDL trigger at launch) on
its ``.CELLS``: within one BF16 rounding of its FP32 reference and bitwise FlashInfer's
``rmsnorm`` (the same kernel code), on the strided q_a rows of the fused projection output."""

import zlib

import pytest
import torch

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="certified on sm_103 and sm_107",
)


def _entry():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention.rmsnorm_release_at_launch import (  # noqa: E501
        rmsnorm_release_at_launch,
    )

    return rmsnorm_release_at_launch


@pytest.mark.parametrize("cell", _entry().CELLS, ids=[c.why[:40] for c in _entry().CELLS])
def test_rmsnorm_release_at_launch_cells(cell):
    from flashinfer.norm import rmsnorm

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_mla_q import (  # noqa: E501
        q_norm_pdl,
    )

    op = _entry()
    spec = cell.spec
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    rows = torch.randn(spec["tokens"], spec["row_stride"], device="cuda", generator=gen)
    x = rows.to(torch.bfloat16)[:, : spec["width"]]
    weight = (1 + 0.1 * torch.randn(spec["width"], device="cuda", generator=gen)).to(torch.bfloat16)
    eps = spec["eps"]
    assert q_norm_pdl.can_use(x, weight), cell.why
    with validating(op):
        y = op(x, weight, eps)
    assert y.is_contiguous() and y.shape == x.shape and y.dtype == torch.bfloat16
    # One BF16 rounding of the FP32 result, against the row's largest magnitude.
    assert_within_ulp(y, op.reference(x, weight, eps), element_ulp=1, rms_ulp=0.5)
    from flashinfer.norm import _use_cuda_norm

    if _use_cuda_norm(x.device):
        return  # FlashInfer runs its CUDA JIT kernel here, not the CuTe DSL one repeated
    expected = torch.empty_like(y)
    rmsnorm(x, weight, eps, out=expected, enable_pdl=True)
    assert torch.equal(y.view(torch.int16), expected.view(torch.int16)), cell.why
