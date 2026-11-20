# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.catalog.attention.prims_dense_attention import (
    prims_dense_attention,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="PrimTS needs Blackwell",
)
@pytest.mark.parametrize("cell", prims_dense_attention.CELLS, ids=lambda c: c.why)
def test_prims_dense_attention(cell):
    rng = torch.Generator(device="cuda").manual_seed(528)

    def sample(length):
        return torch.randn(1, length, 2, 128, device="cuda", dtype=torch.bfloat16, generator=rng)

    q, k, v = sample(137), sample(265), sample(265)
    kwargs = dict(plans={}, quantize_qk=cell.spec["quantize_qk"])
    if cell.spec.get("grouping"):
        kwargs["vc_permutation"] = (
            torch.arange(264, -1, -1, dtype=torch.int32, device="cuda")
            .view(1, 1, -1)
            .expand(1, 2, -1)
            .contiguous()
        )
    if cell.spec.get("repair"):
        kwargs["vc_repair_budget"] = 0.25
    prims_dense_attention.is_valid(q, k, v, **kwargs)
    actual = prims_dense_attention(q, k, v, **kwargs)
    expected = prims_dense_attention.reference(q, k, v, **kwargs)
    assert actual.dtype == q.dtype
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.1, atol=0.035)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="PrimTS needs Blackwell",
)
@pytest.mark.parametrize("quantize_qk", [False, True])
def test_vc_repair_excludes_padded_keys(quantize_qk):
    q = torch.zeros((1, 265, 2, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.ones_like(q)
    y = prims_dense_attention(q, q, v, plans={}, quantize_qk=quantize_qk, vc_repair_budget=0.25)
    torch.testing.assert_close(y, v, rtol=0, atol=0)


def test_vc_requires_matching_q_kv_heads():
    q = torch.empty(1, 128, 2, 128, dtype=torch.bfloat16)
    k = torch.empty(1, 128, 1, 128, dtype=torch.bfloat16)
    prims_dense_attention.is_valid(q, k, k, plans={})
    with pytest.raises(AssertionError):
        prims_dense_attention.is_valid(q, k, k, plans={}, vc_repair_budget=0.25)
