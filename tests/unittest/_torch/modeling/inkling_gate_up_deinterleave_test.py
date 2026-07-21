#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit test pinning the Inkling fused gate/up DE-INTERLEAVE on weight load.

Root cause of the incoherent assembled-text bug: the SGLang-format Inkling NVFP4
checkpoint stores every fused gate+up weight (``w13_dn`` dense, ``w13_weight``
routed, ``shared_w13_weight`` shared) with the two projections INTERLEAVED along
the ``2*inter`` output dim -- ``[g0, u0, g1, u1, ...]`` -- because SGLang runs with
``inference_moe_w13_interleaved=True`` (its default) and reads it as
``silu(z[..., ::2]) * z[..., 1::2]``. The TRT-LLM mapper used to split it with a
plain contiguous ``chunk(2)`` (``[first half | second half]``), which pairs the
wrong gate/up channels in every dense-MLP / routed-expert / shared-expert SwiGLU.
Isolated single-layer tests missed it because their local reference made the SAME
contiguous mis-read (reference-loop drift), so both agreed while the real model
produced garbage.

``_split_interleaved_gate_up`` returns STRIDED VIEWS (gate = even, up = odd) rather
than a contiguous copy, so the multi-hundred-GiB fused w13 is not materialized into
a private per-rank host copy on load (that OOM-killed the TP=4 load); the fused-MoE
/ gate_up loaders shard then ``.contiguous()`` the small per-rank slice.

No GPU or checkpoint needed: this verifies the pure split math and the mapper
regexes against the SGLang strided semantics (SGLang scores GSM8K 0.9553 reading
w13 interleaved, so that read is authoritative).
"""

import torch

from tensorrt_llm._torch.models.checkpoints.hf.inkling_weight_mapper import (
    _split_interleaved_gate_up, _DENSE_W13_RE, _EXPERT_RE)


def _interleave(gate: torch.Tensor, up: torch.Tensor, dim: int) -> torch.Tensor:
    """Build an Inkling-interleaved [g0, u0, g1, u1, ...] tensor from separate
    gate/up halves (the on-disk layout), independent of the code under test."""
    stacked = torch.stack([gate, up], dim=dim + 1)  # [..., k, 2, ...]
    shape = list(gate.shape)
    shape[dim] *= 2
    return stacked.reshape(shape).contiguous()


def test_split_recovers_interleaved_gate_up():
    """``_split_interleaved_gate_up`` returns exactly the gate (even) / up (odd)
    halves that were interleaved -- i.e. SGLang's ``z[::2]`` / ``z[1::2]`` read."""
    torch.manual_seed(0)
    for dim, shape in [(0, (8, 5)), (0, (8, 3)), (1, (2, 8, 4)), (2, (2, 3, 8))]:
        k = shape[dim] // 2
        gshape = list(shape)
        gshape[dim] = k
        gate = torch.randn(gshape)
        up = torch.randn(gshape)
        interleaved = _interleave(gate, up, dim)
        d_gate, d_up = _split_interleaved_gate_up(interleaved, dim=dim)
        assert torch.equal(d_gate, gate), (dim, shape)
        assert torch.equal(d_up, up), (dim, shape)


def test_split_returns_views_not_copies():
    """Memory-safety contract: the split must NOT materialize a copy (that OOM'd
    the TP=4 load). gate/up must alias the source storage."""
    t = torch.randn(16, 6)
    gate, up = _split_interleaved_gate_up(t, dim=0)
    assert gate.data_ptr() == t.data_ptr()  # even rows start at offset 0
    assert up.data_ptr() == t[1].data_ptr()  # odd rows start at row 1
    assert not gate.is_contiguous()  # strided view over every other row


def test_split_is_row_permutation_safe_for_packed_and_scale():
    """Reorders whole output rows only, so it is valid for a packed-fp4 uint8
    weight ([2*inter, hidden/2]) and its per-block fp8 scale ([2*inter, nblk])."""
    inter = 6
    gate_w = torch.arange(inter * 4, dtype=torch.uint8).reshape(inter, 4)
    up_w = (torch.arange(inter * 4, dtype=torch.uint8) + 100).reshape(inter, 4)
    interleaved = _interleave(gate_w, up_w, dim=0)
    d_gate, d_up = _split_interleaved_gate_up(interleaved, dim=0)
    assert torch.equal(d_gate, gate_w) and torch.equal(d_up, up_w)
    assert d_gate.dtype == torch.uint8

    scale = torch.randn(inter, 12).to(torch.float8_e4m3fn)
    scale_up = torch.randn(inter, 12).to(torch.float8_e4m3fn)
    il = _interleave(scale.float(), scale_up.float(), dim=0).to(torch.float8_e4m3fn)
    dg, du = _split_interleaved_gate_up(il, dim=0)
    assert torch.equal(dg.float(), scale.float())
    assert torch.equal(du.float(), scale_up.float())


def test_odd_dim_rejected():
    try:
        _split_interleaved_gate_up(torch.zeros(7, 3), dim=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError on odd gate/up dim")


def test_mapper_regexes_target_the_fused_gate_up_keys():
    """Guard: the de-interleave sites match exactly the fused-w13 checkpoint keys
    (and NOT w2 / down keys), so no fused gate/up tensor loads un-fixed and no
    down-projection is wrongly permuted. (shared_w13 loads RAW and is split in
    InklingSharedExperts.forward, so it has no mapper regex.)"""
    assert _DENSE_W13_RE.search("layers.0.mlp.w13_dn.weight")
    assert not _DENSE_W13_RE.search("layers.0.mlp.w2_md.weight")
    m = _EXPERT_RE.search("layers.3.mlp.experts.w13_weight")
    assert m and m.group(2) == "w13_weight"
    m = _EXPERT_RE.search("layers.3.mlp.experts.w13_weight.scale")
    assert m and m.group(2) == "w13_weight"  # block scale also split
    m = _EXPERT_RE.search("layers.3.mlp.experts.w2_weight")
    assert m and m.group(2) == "w2_weight"  # down proj -> NOT split


if __name__ == "__main__":
    test_split_recovers_interleaved_gate_up()
    test_split_returns_views_not_copies()
    test_split_is_row_permutation_safe_for_packed_and_scale()
    test_odd_dim_rejected()
    test_mapper_regexes_target_the_fused_gate_up_keys()
    print("INKLING_DEINTERLEAVE_UNIT_OK")
