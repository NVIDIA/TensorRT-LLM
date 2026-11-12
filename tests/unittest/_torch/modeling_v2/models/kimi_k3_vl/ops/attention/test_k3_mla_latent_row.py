# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``k3_mla_latent_row`` entry (kv_a RMSNorm of the latent and the K-pe copy in one launch)
on its ``.CELLS``: the norm within one BF16 rounding of the FP32 reference, the K-pe columns
copied exactly, and the row bitwise FlashInfer's ``rmsnorm`` followed by the copy."""

import zlib

import pytest
import torch

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="certified on sm_103 and sm_107",
)

LORA, ROPE = 512, 64


def _entry():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention.k3_mla_latent_row import (  # noqa: E501
        k3_mla_latent_row,
    )

    return k3_mla_latent_row


@pytest.mark.parametrize("cell", _entry().CELLS, ids=[c.why[:40] for c in _entry().CELLS])
@pytest.mark.parametrize("use_pdl", [True, False])
def test_k3_mla_latent_row_cells(cell, use_pdl):
    from flashinfer.norm import rmsnorm

    op = _entry()
    spec = cell.spec
    tokens, stride, eps = spec["tokens"], spec["row_stride"], spec["eps"]
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    proj = torch.randn(tokens, stride, device="cuda", generator=gen).to(torch.bfloat16)
    c_kv = proj[:, stride - LORA - ROPE : stride - ROPE]
    k_pe = proj[:, stride - ROPE :]
    weight = (1 + 0.1 * torch.randn(LORA, device="cuda", generator=gen)).to(torch.bfloat16)
    with validating(op):
        out = op(c_kv, k_pe, weight, eps, use_pdl=use_pdl)
    assert out.shape == (tokens, LORA + ROPE) and out.is_contiguous()
    expected = op.reference(c_kv, k_pe, weight, eps, use_pdl=use_pdl)
    assert_within_ulp(out[:, :LORA], expected[:, :LORA], element_ulp=1, rms_ulp=0.5)
    assert torch.equal(out[:, LORA:], expected[:, LORA:]), cell.why
    from flashinfer.norm import _use_cuda_norm

    if _use_cuda_norm(c_kv.device):
        return  # FlashInfer runs its CUDA JIT kernel here, not the CuTe DSL one repeated
    normed = torch.empty((tokens, LORA), dtype=torch.bfloat16, device="cuda")
    rmsnorm(c_kv, weight, eps, out=normed)
    assert torch.equal(out[:, :LORA].view(torch.int16), normed.view(torch.int16)), cell.why
