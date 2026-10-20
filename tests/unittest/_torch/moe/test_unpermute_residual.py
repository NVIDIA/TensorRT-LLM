# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""unpermute_residual is bitwise moe_unpermute_inplace (+ the ATen bf16 add of the residual), on adversarial data where
the top-k summation order and the fma contraction matter: mixed magnitudes (1e-3 .. 1e3), cancellation (paired +x / -x
rows), fp32 scales with full mantissas, -1 (non-local) slots, and the M3 CTX rank shape."""

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401  (registers torch.ops.trtllm)
import tensorrt_llm._torch.moe.fused_moe.unpermute_residual as ur

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _case(t, hidden, top_k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    m = t * top_k
    mag = 10.0 ** torch.randint(-3, 4, (m, 1), device="cuda", generator=g).float()
    rows = (torch.randn(m, hidden, device="cuda", generator=g) * mag).to(torch.bfloat16)
    rows[1::2] = -rows[0::2] * 1.0009765625  # near-cancelling neighbours
    perm = torch.randperm(m, device="cuda", generator=g).int()
    e2p = perm.clone()
    e2p[torch.rand(m, device="cuda", generator=g) < 0.4] = -1  # non-local experts
    scales = torch.rand(t, top_k, device="cuda", generator=g) * 1.7 + 1e-3
    res = (torch.randn(t, hidden, device="cuda", generator=g) * 50.0).to(torch.bfloat16)
    return rows, e2p.view(t, top_k), scales, res


@pytest.mark.parametrize("t,hidden", [(97, 6144), (2048, 6144)])
def test_bitwise_vs_moe_unpermute_plus_add(t, hidden) -> None:
    top_k = 4
    rows, e2p, scales, res = _case(t, hidden, top_k, 7 + t)
    ref = torch.empty(t, hidden, dtype=torch.bfloat16, device="cuda")
    torch.ops.trtllm.moe_unpermute_inplace(
        permuted_input=rows, output=ref, expanded_idx_to_permuted_idx=e2p, topk_scales=scales
    )
    ref_add = res.clone().add_(ref)
    match = {}
    for contract in (True, False):
        o = torch.empty_like(ref)
        ur.unpermute_residual(rows, o, e2p, scales, None, contract)
        o2 = torch.empty_like(ref)
        ur.unpermute_residual(rows, o2, e2p, scales, res, contract)
        match[contract] = (torch.equal(o.view(torch.int16), ref.view(torch.int16)),
                           torch.equal(o2.view(torch.int16), ref_add.view(torch.int16)))  # fmt: skip
    print(f"\nUNPERM t={t}: contract {match[True]} separate {match[False]}", flush=True)
    assert all(match[True]), "the fma form must reproduce moe_unpermute (+ add) bit for bit"


@pytest.mark.parametrize("t", [16384, 32768])
def test_every_launch_config_bitwise(t, monkeypatch) -> None:
    """Every (BLOCK, num_warps) launch config of unpermute_residual (with residual) is bitwise moe_unpermute_inplace +
    add_ at the CTX shape (hidden 6144, top-4, 50 % local rows)."""
    hidden, top_k = 6144, 4
    rows, e2p, scales, res = _case(t, hidden, top_k, 3)
    ref = torch.empty(t, hidden, dtype=torch.bfloat16, device="cuda")
    torch.ops.trtllm.moe_unpermute_inplace(
        permuted_input=rows, output=ref, expanded_idx_to_permuted_idx=e2p, topk_scales=scales
    )
    want = res.clone().add_(ref)
    out = torch.empty_like(ref)
    for block in (512, 1024, 2048):
        for warps in (2, 4, 8):
            monkeypatch.setattr(ur, "_CFG", (block, warps))
            ur.unpermute_residual(rows, out, e2p, scales, res)
            assert torch.equal(out.view(torch.int16), want.view(torch.int16)), (block, warps)
