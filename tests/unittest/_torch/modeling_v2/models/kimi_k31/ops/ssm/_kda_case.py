# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs of the ULL target's ``ssm`` entry tests: one TP16 KDA layer shard (6 heads) with MR8's
prepared core built as the target's ``FlatKda`` builds it, its packed rows and checkpoint caches.

The forget-gate operands are small dyadic values (``f_a`` multiples of 1/16 or 1/4, ``f_b`` of 1/64)
so ``f_a . f_b^T`` is exact in FP32 in any summation order: the kernels round it to BF16 before the
gate, and an order-dependent FP32 sum could land on the other side of a BF16 rounding boundary than
the FP64 reference, which moves a decay by up to a few percent.
"""

import math

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.vendor.vllm_mr8 import (  # noqa: E501
    module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm._rubin_ull_kda_reference import (  # noqa: E501
    scalar,
)

LOWER = -5.0  # the checkpoint's gate_lower_bound
NORM_EPS = 1e-5
HEAD_DIM = 128


def pitch_of(heads: int) -> int:
    """``layers.kda_pitch``: [q, k, v, g, f_a, beta] rows padded to 16."""
    return (4 * heads * HEAD_DIM + HEAD_DIM + heads + 15) // 16 * 16


def layer_weights(heads: int, g: torch.Generator) -> dict:
    dim, dev = heads * HEAD_DIM, "cuda"
    a_log = torch.empty(heads, device=dev).uniform_(1.0, 16.0, generator=g).log()
    dt_bias = torch.empty(dim, device=dev).uniform_(math.log(1e-3), math.log(1e-1), generator=g)
    return dict(
        conv=(torch.randn(3, 4, dim, generator=g, device=dev) * 0.5).contiguous(),
        f_b=(torch.randint(-16, 17, (dim, HEAD_DIM), generator=g, device=dev) / 64).bfloat16(),
        a_log=a_log.float().contiguous(),
        dt_bias=dt_bias.float().contiguous(),
        norm=(1.0 + 0.1 * torch.randn(HEAD_DIM, generator=g, device=dev)).float(),
    )


def make_core(weights: dict, heads: int, policies: dict, caches=None):
    """MR8's ``PreparedKda`` exactly as ``kda.FlatKda`` binds it (convolution weights already in
    the kernel layout); compiled for ``policies`` over ``caches`` when given."""
    from cutlass import Float32

    prepared = module("cutedsl.kda.prepared")
    core = prepared.PreparedKda.__new__(prepared.PreparedKda)
    core.heads, core.dim, core.has_bias = heads, heads * HEAD_DIM, False
    core.weights = (
        weights["conv"],
        weights["dt_bias"],
        weights["f_b"],
        weights["a_log"],
        weights["dt_bias"],
        weights["norm"],
    )
    core.device = weights["conv"].device
    core.policies = policies
    core.scalars = tuple(Float32(v) for v in (LOWER, HEAD_DIM**-0.5, NORM_EPS))
    core.conv_dim_first = True
    core.kernels = {}
    core.prepared = frozenset()
    core.prepared_kernels = {}
    core.readers = torch.empty(0, dtype=torch.int32, device=core.device)
    assert scalar(core.scalars[0]) == LOWER, "the reference cannot read the core's scalars"
    if caches is not None:
        core.prepare_caches(caches)
    return core


def packed_rows(rows: int, heads: int, g: torch.Generator) -> torch.Tensor:
    """BF16 [rows, pitch]: random q / k / v / g / beta, dyadic f_a (multiples of 1/16), junk pad."""
    dim = heads * HEAD_DIM
    x = torch.randn(rows, pitch_of(heads), generator=g, device="cuda")
    x[:, 4 * dim : 4 * dim + HEAD_DIM] = (
        torch.randint(-16, 17, (rows, HEAD_DIM), generator=g, device="cuda") / 16
    )
    return x.bfloat16()


def checkpoint_caches(slots: int, heads: int, width: int, g: torch.Generator):
    """(conv BF16 [slots, 3 dim, width], state FP32 [slots, heads, 128, 128])."""
    dim = heads * HEAD_DIM
    conv = torch.randn(slots, 3 * dim, width, generator=g, device="cuda").bfloat16()
    state = torch.randn(slots, heads, HEAD_DIM, HEAD_DIM, generator=g, device="cuda") * 0.1
    return conv, state


def verify_slots(requests: int, tokens: int, padded, slots: int, g: torch.Generator):
    """int32 [requests, tokens]: distinct positions 1..slots-1 per request, 0 rows for padding."""
    perm = torch.randperm(slots - 1, generator=g, device="cuda") + 1
    table = perm[: requests * tokens].view(requests, tokens).to(torch.int32).clone()
    for b in padded:
        table[b] = 0
    return table


def assert_core_matches(out, caches, before, slots, accepted, tokens, ref) -> None:
    """``out`` and the written state rows / conv pages against the FP64 reference; every other
    row bit-for-bit unchanged. ``slots`` [requests, tokens] (q = 1: [requests, 1])."""
    ref_out, ref_states, ref_pages = ref
    conv, state = caches
    conv_before, state_before = before
    written_state, written_conv = set(), set()
    table = slots.tolist()
    for b, row in enumerate(table):
        rows = slice(b * tokens, (b + 1) * tokens)
        if ref_states[b] is None:
            assert bool((out[rows] == 0).all()), f"request {b}: padding rows not zero"
            continue
        got, want = out[rows].double(), ref_out[rows]
        bound = 2.0**-8 * want.abs() + 2.0**-12 * want.abs().amax(-1, keepdim=True)
        err = (got - want).abs()
        assert bool((err <= bound).all()), f"request {b}: out max err {err.max().item():.3e}"
        for t, slot in enumerate(row[:tokens]):
            if slot <= 0:
                continue
            want_s = ref_states[b][t]
            err_s = (state[slot].double() - want_s).abs().max().item()
            assert err_s <= 2.0**-16 * want_s.abs().max().item(), (
                f"request {b} token {t}: state err {err_s:.3e}"
            )
            written_state.add(slot)
        page = ref_pages[b]
        assert torch.equal(conv[row[0], :, : page.shape[1]].double(), page), (
            f"request {b}: conv page differs"
        )
        written_conv.add(row[0])
    for slot in range(state.shape[0]):
        if slot not in written_state:
            assert torch.equal(state[slot], state_before[slot]), f"state row {slot} was written"
        if slot not in written_conv:
            assert torch.equal(conv[slot], conv_before[slot]), f"conv page {slot} was written"
