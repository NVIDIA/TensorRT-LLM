# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for `mid_rows_mm`'s conv mode (`mid_mm_conv_bf16`): the bf16 input projection with
the lean conv as its epilogue is bitwise the two launches it replaces -- `mid_mm_raw` at the same
plan and MMA N, then `gdn_conv_lean` on its q|k|v columns -- in the conv output, the saved raw
rows and the z | a | b columns, eagerly and under CUDA graph replay; a request on the pad slot
writes nothing; the pool is read-only."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.gemm.mid_rows_mm import (
    Plan,
    mid_mm_conv_bf16,
    mid_mm_raw,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.ssm.gdn_conv_lean import conv_lean

assert torch.cuda.is_available(), "mid_mm_conv_bf16 requires a CUDA device"

#: An attention-DP rank's GDN input projection: `[q | k | v | z | a | b]` = 10240 | 6144 | 48 |
#: 48 over hidden 2560, the conv over the first 10240 columns.
N, K, C, T = 16480, 2560, 10240, 4
PAD = -1
#: `(requests, MMA N, plan)`: the input-projection release table's plans at 2..16 requests
#: (`_INPROJ_RELEASE_PLANS`, "op" MMA N up to 64 rows), and one "fine" MMA N past it.
_CASES = (
    (2, 32, Plan(1, 8, 5, 2, 16)),
    (3, 32, Plan(1, 8, 5, 2, 16)),
    (5, 32, Plan(1, 8, 5, 2, 16)),
    (8, 32, Plan(1, 8, 5, 2, 16)),
    (12, 64, Plan(1, 8, 5, 2, 16)),
    (16, 64, Plan(1, 8, 5, 2, 16)),
    (20, 80, Plan(1, 4, 8, 4, 16)),
)


def _operands(n_req, seed, pad=None):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(torch.bfloat16)

    rows = n_req * T
    x = randn(rows, K)
    w = randn(N, K) * 0.05
    taps = randn(C, 4)
    page = randn(40, 2, C, 3)  # a two-layer page: the layer is the second one's view
    slots = torch.randperm(40, generator=g, device="cuda")[:n_req].int()
    if pad is not None:
        slots[pad] = PAD
    xrows = (torch.arange(n_req, device="cuda").flip(0) + 1).int()
    return x, w, taps, page, slots, xrows


@pytest.mark.parametrize("n_req, mpad, plan", _CASES, ids=[f"{c[0]}req-n{c[1]}" for c in _CASES])
def test_conv_mode_is_bitwise_the_two_launches(n_req, mpad, plan) -> None:
    x, w, taps, page, slots, xrows = _operands(n_req, seed=n_req)
    pool = page[:, 1]
    before = page.clone()
    xs_fused = torch.zeros(n_req + 2, T, C, dtype=torch.bfloat16, device="cuda")
    xs_ref = xs_fused.clone()
    conv_out, out = mid_mm_conv_bf16(
        x, w, plan, taps, pool, slots, xrows, xs_fused, T, mpad=mpad, pad_slot_id=PAD
    )
    ref_out = mid_mm_raw(x, w, plan, pdl=True, mpad=mpad)
    ref_conv = conv_lean(ref_out[:, :C], taps, pool, slots, xrows, xs_ref, T, pad_slot_id=PAD)
    torch.cuda.synchronize()
    assert torch.equal(conv_out, ref_conv), "the conv output"
    assert torch.equal(xs_fused, xs_ref), "the saved raw rows"
    assert torch.equal(out[:, C:], ref_out[:, C:]), "z | a | b"
    assert torch.equal(page, before), "the pool is read-only"


def test_a_request_on_the_pad_slot_writes_nothing() -> None:
    n_req = 6
    x, w, taps, page, slots, xrows = _operands(n_req, seed=99, pad=2)
    xs = torch.full((n_req + 2, T, C), 7.0, dtype=torch.bfloat16, device="cuda")
    conv_out, _ = mid_mm_conv_bf16(
        x, w, Plan(1, 8, 5, 2, 16), taps, page[:, 1], slots, xrows, xs, T, mpad=32
    )
    ref = conv_lean(
        mid_mm_raw(x, w, Plan(1, 8, 5, 2, 16), mpad=32)[:, :C], taps, page[:, 1], slots, xrows,
        torch.zeros_like(xs), T,
    )  # fmt: skip
    torch.cuda.synchronize()
    live = torch.ones(n_req * T, dtype=torch.bool, device="cuda")
    live[2 * T : 3 * T] = False
    assert torch.equal(conv_out[live], ref[live])
    assert torch.all(xs[int(xrows[2])] == 7.0), "a pad request's xsave row is untouched"


def test_conv_mode_under_graph_replay() -> None:
    """Captured once, replayed over fresh activations and windows: still the two launches'."""
    n_req, mpad, plan = 8, 32, Plan(1, 8, 5, 2, 16)
    x, w, taps, page, slots, xrows = _operands(n_req, seed=5)
    pool = page[:, 1]
    xs = torch.zeros(n_req + 2, T, C, dtype=torch.bfloat16, device="cuda")
    mid_mm_conv_bf16(x, w, plan, taps, pool, slots, xrows, xs, T, mpad=mpad)  # compile
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        conv_out, out = mid_mm_conv_bf16(x, w, plan, taps, pool, slots, xrows, xs, T, mpad=mpad)
    g = torch.Generator(device="cuda").manual_seed(11)
    for _ in range(4):
        x.copy_(torch.randn(x.shape, generator=g, device="cuda").to(torch.bfloat16))
        page.copy_(torch.randn(page.shape, generator=g, device="cuda").to(torch.bfloat16))
        graph.replay()
        ref_out = mid_mm_raw(x, w, plan, mpad=mpad)
        xs_ref = torch.zeros_like(xs)
        ref_conv = conv_lean(ref_out[:, :C], taps, pool, slots, xrows, xs_ref, T)
        torch.cuda.synchronize()
        assert torch.equal(conv_out, ref_conv) and torch.equal(out[:, C:], ref_out[:, C:])
        assert torch.equal(xs[xrows.long()], xs_ref[xrows.long()])
