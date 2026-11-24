# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.attention.prims_dense_attention import (
    prims_dense_attention,
    vc_token_permutation,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VC grouping cell requires CUDA")
@pytest.mark.parametrize("cell", vc_token_permutation.CELLS, ids=lambda c: c.why)
def test_vc_token_permutation(cell):
    tokens = cell.spec["tokens"]
    token_ids = torch.arange(tokens, device="cuda") * 17 % tokens
    offsets = torch.arange(4, device="cuda").reshape(2, 1, 2)
    coords = token_ids[None, :, None] + offsets
    v = torch.stack((coords, coords / 2), dim=-1).bfloat16()
    initial = None
    if cell.spec["warm"]:
        active = torch.tensor([0, 1, 3, 10], device=v.device) + offsets.reshape(4, 1)
        initial = torch.full((4, min(tokens, 64), 2), -512.0, device=v.device)
        initial[:, :4] = torch.stack((active, active / 2), dim=-1)
    generator = torch.Generator(device=v.device).manual_seed(125)
    oracle_rng = torch.Generator(device=v.device).set_state(generator.get_state())
    initial_rng = generator.get_state()
    initial_copy = initial.clone() if initial is not None else None
    vc_token_permutation.is_valid(v, init_centroids=initial, generator=generator)
    permutation, centroids = vc_token_permutation(v, init_centroids=initial, generator=generator)
    expected_indices, expected_centroids = vc_token_permutation.reference(
        v, init_centroids=initial, generator=oracle_rng
    )
    torch.testing.assert_close(permutation, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(centroids, expected_centroids, rtol=0, atol=0)
    assert torch.equal(generator.get_state(), oracle_rng.get_state())
    assert torch.equal(generator.get_state(), initial_rng) == cell.spec["warm"]
    if initial_copy is not None:
        assert torch.equal(initial, initial_copy)
        assert torch.equal(centroids[:, 4:], initial_copy[:, 4:])


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="PrimTS needs Blackwell",
)
@pytest.mark.parametrize("cell", prims_dense_attention.CELLS, ids=lambda c: c.why)
def test_prims_dense_attention(cell):
    rng = torch.Generator(device="cuda").manual_seed(528)

    def sample(length):
        return torch.randn(1, length, 2, 128, device="cuda", dtype=torch.bfloat16, generator=rng)

    kv_len = cell.spec.get("kv_len", 265)
    q, k, v = sample(137), sample(kv_len), sample(kv_len)
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
    from tensorrt_llm._torch.attention.backends.prims_ts import BatchPrefillTSWrapper

    (plan,) = kwargs["plans"].values()
    assert type(plan) is BatchPrefillTSWrapper
    expected = prims_dense_attention.reference(q, k, v, **kwargs)
    assert actual.dtype == q.dtype
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.1, atol=0.035)


def test_vc_requires_matching_q_kv_heads():
    q = torch.empty(1, 128, 2, 128, dtype=torch.bfloat16)
    k = torch.empty(1, 128, 1, 128, dtype=torch.bfloat16)
    prims_dense_attention.is_valid(q, k, k, plans={})
    with pytest.raises(AssertionError):
        prims_dense_attention.is_valid(q, k, k, plans={}, vc_repair_budget=0.25)
