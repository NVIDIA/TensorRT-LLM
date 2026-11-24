# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the inkling2_attention_prologue entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.inkling2.ops.attention.inkling2_attention_prologue import (
    inkling2_attention_prologue as op,
)
from tensorrt_llm._torch.modeling_v2.models.inkling2.ops.attention.inkling2_attention_prologue import (
    log_scaling,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "inkling2_attention_prologue requires a CUDA device"

QK_CLIP, REL_CLIP, EPS = 22.627416997969522, 8.0, 1e-6


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, h, kv, d, dr = (
        spec["tokens"],
        spec["heads"],
        spec["kv_heads"],
        spec["head_dim"],
        spec["d_rel"],
    )

    def bf16(*shape, scale=1.0):
        return (scale * torch.randn(*shape, generator=g, device="cuda")).to(torch.bfloat16)

    # Norms around the clip thresholds, so some heads clip and some do not.
    q, k, v, r = (
        bf16(t, h, d, scale=2.0),
        bf16(t, kv, d, scale=2.0),
        bf16(t, kv, d),
        bf16(t, h, dr, scale=2.0),
    )
    q_scale, r_scale = (
        (1.0 + 0.3 * torch.rand(h, d, generator=g, device="cuda")).to(torch.bfloat16),
        bf16(h, dr),
    )
    tau = None
    if spec["tau"]:
        positions = torch.arange(20000, 20000 + t, device="cuda")
        tau = log_scaling(positions, 0.1, 8192.0)
    return q, k, v, r, q_scale, r_scale, tau, QK_CLIP, REL_CLIP, EPS


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    args = _build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    for out, ref in zip(op(*args), op.reference(*args)):
        assert_within_ulp(
            out.reshape(-1, out.shape[-1]),
            ref.reshape(-1, ref.shape[-1]),
            element_ulp=1.0,
            rms_ulp=0.1,
        )


def test_clip_bounds_and_tau() -> None:
    args = _build(dict(tokens=9, heads=12, kv_heads=2, head_dim=128, d_rel=16, tau=False), 1)
    q, k, v, r = op(*args)
    assert (
        q.float().norm(dim=-1).max() <= QK_CLIP * 1.01
        and k.float().norm(dim=-1).max() <= QK_CLIP * 1.01
    )
    assert r.float().norm(dim=-1).max() <= REL_CLIP * 1.01
    torch.testing.assert_close(
        v.float().pow(2).mean(-1), torch.ones(9, 2, device="cuda"), rtol=0.02, atol=0.02
    )
    tau = log_scaling(torch.tensor([0, 8190, 8191, 8192 * 4 - 1], device="cuda"), 0.1, 8192.0)
    torch.testing.assert_close(
        tau, torch.tensor([1.0, 1.0, 1.0, 1.0 + 0.1 * 1.3862944], device="cuda")
    )
