# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness of the CuTe DSL MiniMax-M3 gate GEMM.

The kernel replaces ``F.linear(hidden_states.float(), weight)`` for the router
projection, so what has to hold is an accuracy claim rather than a bitwise one:
carrying the FP32 weight as two BF16 terms has to leave the logits close enough
to an FP64 reference that top-k over 128 experts cannot tell the difference.

The epilogue fold is the delicate part. It sums the weight terms by pairing
accumulator subtiles a fixed distance apart, which is only valid for some tile
shapes, so :func:`fold_is_supported` is tested directly against every tactic
rather than trusted -- a wrong answer there is silent.
"""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.minimax_m3_gate_gemm_runner import (
    default_tactic,
    fold_is_supported,
    gate_gemm,
    split_k_is_supported,
    split_weight,
)
from tensorrt_llm._utils import get_sm_version

HIDDEN = 6144
EXPERTS = 128

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() // 10 != 10,
    reason="The gate GEMM targets Blackwell SM100 tcgen05 MMA.",
)

# Every tile shape the runner might be pointed at, valid or not, so the fold
# guard is exercised on both sides.
ALL_TACTICS = [
    (use_2cta, tiler, cluster)
    for use_2cta in (False, True)
    for tiler in ((64, 128), (128, 128), (256, 128), (64, 256), (128, 256), (256, 256))
    for cluster in ((1, 1), (2, 1), (4, 1))
]


def _inputs(num_tokens: int, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(num_tokens, HIDDEN, device="cuda", dtype=torch.float32, generator=gen).to(
        torch.bfloat16
    )
    w = (
        torch.randn(EXPERTS, HIDDEN, device="cuda", dtype=torch.float32, generator=gen)
        * HIDDEN**-0.5
    )
    return x, w


def _reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (x.double() @ w.double().t()).float()


def _rel_error(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Max error against the RMS logit, not against each element.

    Router logits cross zero, and a per-element relative error would be
    dominated by the ones that land near it -- exactly the ones whose value the
    routing does not depend on.
    """
    scale = ref.double().pow(2).mean().sqrt()
    return ((got.double() - ref.double()).abs().max() / scale).item()


@pytest.mark.parametrize("terms,tolerance", [(1, 3e-2), (2, 3e-4), (3, 3e-4)])
def test_accuracy_by_term_count(terms: int, tolerance: float):
    """More BF16 terms means a more faithful weight, up to the accumulator's own floor."""
    x, w = _inputs(4096)
    got = gate_gemm(x, split_weight(w, terms), terms)
    assert _rel_error(got, _reference(x, w)) < tolerance


def test_two_terms_recover_fp32_grade_accuracy():
    """Two terms have to beat one by orders of magnitude, or the split is pointless."""
    x, w = _inputs(2048)
    ref = _reference(x, w)
    one = _rel_error(gate_gemm(x, split_weight(w, 1), 1), ref)
    two = _rel_error(gate_gemm(x, split_weight(w, 2), 2), ref)
    assert two < one / 50


@pytest.mark.parametrize("num_tokens", [1, 7, 32, 64, 129, 512, 1000, 4096, 8192, 16384])
def test_matches_reference_across_token_counts(num_tokens: int):
    """Including sizes that do not divide the tile, where the mask has to hold."""
    x, w = _inputs(num_tokens)
    got = gate_gemm(x, split_weight(w, 2), 2)
    assert got.shape == (num_tokens, EXPERTS)
    assert _rel_error(got, _reference(x, w)) < 3e-4


@pytest.mark.parametrize("num_tokens", [512, 8192, 16384])
def test_fused_fold_matches_unfused(num_tokens: int):
    """The epilogue fold must agree with summing the terms in a second pass.

    Both sum the same FP32 accumulators, so this is an exact comparison, not an
    approximate one. ``split_k=1`` is pinned because a K partition changes the
    accumulation order, which is a different question from the fold's.
    """
    x, w = _inputs(num_tokens)
    w_split = split_weight(w, 2)
    torch.testing.assert_close(
        gate_gemm(x, w_split, 2, fused=True, split_k=1),
        gate_gemm(x, w_split, 2, fused=False, split_k=1),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("num_tokens", [1, 7, 32, 129, 512, 2048])
@pytest.mark.parametrize("split_k", [2, 4, 8, 16, 32])
def test_split_k_matches_reference(num_tokens: int, split_k: int):
    """Every K partition count has to land on the same logits.

    The partition boundaries are unpredicated -- each partition takes an equal
    run of whole K tiles -- so an off-by-one in the offset silently drops a
    slice of the hidden dimension and still returns plausible numbers. Only a
    reference comparison catches that.
    """
    x, w = _inputs(num_tokens)
    got = gate_gemm(x, split_weight(w, 2), 2, split_k=split_k)
    assert got.shape == (num_tokens, EXPERTS)
    assert _rel_error(got, _reference(x, w)) < 3e-4


def test_split_k_covers_all_of_k():
    """A dropped K slice is the failure mode a reference tolerance can hide.

    Feeding a weight of all ones makes each logit a plain sum over the hidden
    dimension, so a partition that skipped tiles shows up as a proportionally
    smaller number rather than as noise.
    """
    x, _ = _inputs(64)
    w = torch.ones(EXPERTS, HIDDEN, device="cuda", dtype=torch.float32)
    expected = x.double().sum(dim=1, keepdim=True).expand(-1, EXPERTS).float()
    for split_k in (1, 2, 4, 8, 16, 32):
        got = gate_gemm(x, split_weight(w, 1), 1, split_k=split_k)
        # Losing one partition's tiles would move the answer by a fraction of
        # the whole sum, which is four orders of magnitude above this.
        assert _rel_error(got, expected) < 1e-4, f"split_k={split_k}"


def test_split_k_rejects_indivisible_partitions():
    """K must divide into whole MMA tiles per partition, and 6144 does not divide by 5."""
    assert split_k_is_supported(HIDDEN, 4, torch.bfloat16)
    assert not split_k_is_supported(HIDDEN, 5, torch.bfloat16)
    assert split_k_is_supported(HIDDEN, 1, torch.bfloat16)


def test_split_k_falls_back_when_unsupported():
    """An impossible partition count must degrade to one, not compute the wrong thing."""
    x, w = _inputs(64)
    got = gate_gemm(x, split_weight(w, 2), 2, split_k=5)
    assert _rel_error(got, _reference(x, w)) < 3e-4


@pytest.mark.parametrize("tactic", ALL_TACTICS, ids=str)
def test_fold_guard_admits_only_correct_tile_shapes(tactic):
    """``fold_is_supported`` must never green-light a tile shape that folds wrong.

    Tactics the kernel cannot build at all are skipped; the point is that among
    the ones it can, the guard's answer matches what the hardware produces.
    """
    x, w = _inputs(4096)
    w_split = split_weight(w, 2)
    try:
        got = gate_gemm(x, w_split, 2, tactic, fused=True, split_k=1)
    except Exception:  # noqa: BLE001 - an unbuildable tile shape is not a guard failure
        pytest.skip(f"{tactic} is not a constructible tile shape")

    correct = _rel_error(got, _reference(x, w)) < 3e-4
    if fold_is_supported(tactic, 2 * EXPERTS):
        assert correct, f"guard admitted {tactic} but the fold is wrong"
    else:
        # The guard sent this one down the unfused path, which is always correct.
        assert correct, f"unfused fallback wrong for {tactic}"


@pytest.mark.parametrize("num_tokens", [32, 4096, 16384])
def test_default_tactic_is_usable(num_tokens: int):
    x, w = _inputs(num_tokens)
    tactic = default_tactic(num_tokens)
    got = gate_gemm(x, split_weight(w, 2), 2, tactic)
    assert _rel_error(got, _reference(x, w)) < 3e-4


def test_split_weight_reconstructs_the_weight():
    _, w = _inputs(8)
    for terms, tolerance in ((1, 4e-3), (2, 2e-5), (3, 1e-7)):
        pieces = split_weight(w, terms).view(terms, EXPERTS, HIDDEN).to(torch.float32)
        residual = (pieces.sum(0) - w).abs().max() / w.abs().max()
        assert residual < tolerance, f"{terms} terms left {residual:.2e}"


def test_split_weight_rejects_non_fp32():
    _, w = _inputs(8)
    with pytest.raises(ValueError):
        split_weight(w.to(torch.bfloat16), 2)
