# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the glm_verify_head_accept entry."""

import functools
import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.spec.verify_head_accept import (  # noqa: E501
    glm_verify_head_accept as op,
)
from tensorrt_llm._torch.speculative.interface import (
    _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
    _FORCE_ACCEPT_RNG_POOL_SIZE,
    _FORCE_ACCEPT_RNG_SLOT_STRIDE,
    SpecWorkerBase,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the GLM-5.2 tp16 target's ops are certified on sm_107"
)

VOCAB = 154880
STRIDES = (
    _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
    _FORCE_ACCEPT_RNG_SLOT_STRIDE,
    _FORCE_ACCEPT_RNG_POOL_SIZE,
)


def _spec_worker_rng_state(device):
    """The pool and counter the one-model spec worker builds (fixed seed, identical on every rank)."""
    holder = SimpleNamespace(_force_accept_rng_pool=None, _force_accept_rng_counter=None)
    SpecWorkerBase._ensure_force_accept_rng_state(holder, device)
    return holder._force_accept_rng_pool, holder._force_accept_rng_counter


def _build(spec, seed):
    """One cell's operands.

    Target tokens, draft tokens with `matches[g]` leading tokens equal to request g's targets, the spec worker's
    pool, a counter at a seed-dependent value and, with `rows`, a sentinel-filled accepted_tokens.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    num_contexts, num_gens, draft_len = spec["num_contexts"], spec["num_gens"], spec["draft_len"]
    num_targets = num_contexts + num_gens * (draft_len + 1)
    target = torch.randint(0, VOCAB - 1, (num_targets,), generator=g, device="cuda")
    gen_target = target[num_contexts:].view(num_gens, draft_len + 1)
    draft = torch.randint(0, VOCAB, (num_gens, draft_len), generator=g, device="cuda")
    for request, matched in enumerate(spec["matches"]):
        draft[request, :matched] = gen_target[request, :matched]
        if matched < draft_len:
            # Targets stay below VOCAB - 1, so this is a valid token that differs from the target.
            draft[request, matched] = gen_target[request, matched] + 1
    pool, counter = _spec_worker_rng_state(torch.device("cuda"))
    counter.fill_(seed % 997)
    rows = None
    if spec["rows"]:
        rows = torch.full(
            (num_contexts + num_gens, draft_len + 1), -7, dtype=torch.int32, device="cuda"
        )
    return (
        target.to(spec["target_dtype"]).contiguous(),
        draft.to(torch.int32).contiguous(),
        rows,
        pool,
        counter,
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    """Integer outputs: the op and the ATen chain agree exactly."""
    spec = cell.spec
    target, draft, rows, pool, counter = _build(spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    mode, base, maximum, frac = spec["force"]
    rows_before = rows.clone() if rows is not None else None
    counter_before = counter.clone()
    certifying(op)
    contexts = spec["num_contexts"]
    counts = op(target, draft, contexts, rows, pool, counter, mode, base, maximum, frac, *STRIDES)
    ref_counts, ref_rows, ref_counter = op.reference(
        target,
        draft,
        spec["num_contexts"],
        rows_before,
        pool,
        counter_before,
        mode,
        base,
        maximum,
        frac,
        *STRIDES,
    )
    torch.testing.assert_close(counts, ref_counts, rtol=0, atol=0)
    if rows is not None:
        torch.testing.assert_close(rows, ref_rows, rtol=0, atol=0)
    if ref_counter is not None:
        torch.testing.assert_close(counter, ref_counter, rtol=0, atol=0)
    else:
        torch.testing.assert_close(counter, counter_before, rtol=0, atol=0)


@pytest.mark.parametrize("forced", [2.91, 2.5, 6.1, 3.0])
def test_forced_count_matches_the_spec_worker(forced) -> None:
    """The op's forced counts equal the spec worker's over consecutive steps.

    `SpecWorkerBase._apply_force_accepted_tokens` runs on its own counter, the op on a copy of it with the same
    pool and the arguments the target derives from the forced value as that method does.
    """
    draft_len, num_gens = 7, 4
    worker = SimpleNamespace(
        force_num_accepted_tokens=forced,
        _force_accept_rng_pool=None,
        _force_accept_rng_counter=None,
    )
    worker._ensure_force_accept_rng_state = functools.partial(
        SpecWorkerBase._ensure_force_accept_rng_state, worker
    )
    worker._ensure_force_accept_rng_state(torch.device("cuda"))
    op_counter = worker._force_accept_rng_counter.clone()
    int_part = int(forced)
    frac_part = forced - int_part
    max_total = draft_len + 1
    base_total = min(int_part + 1, max_total)
    mode = 2 if frac_part > 0.0 and base_total < max_total else 1
    target, draft, _, _, _ = _build(
        dict(
            num_contexts=0,
            num_gens=num_gens,
            draft_len=draft_len,
            target_dtype=torch.int32,
            matches=(0,) * num_gens,
            rows=False,
        ),
        seed=17,
    )
    for _ in range(64):
        strict = torch.ones(num_gens, dtype=torch.int32, device="cuda")
        expected = SpecWorkerBase._apply_force_accepted_tokens(worker, strict, 0, draft_len)
        counts = op(
            target,
            draft,
            0,
            None,
            worker._force_accept_rng_pool,
            op_counter,
            mode,
            base_total,
            max_total,
            frac_part if mode == 2 else 0.0,
            *STRIDES,
        )
        torch.testing.assert_close(counts, expected, rtol=0, atol=0)
    torch.testing.assert_close(op_counter, worker._force_accept_rng_counter, rtol=0, atol=0)


def test_cuda_graph_replays_advance_the_counter() -> None:
    """Inside the decode graph every replay draws the next pool entries, as the eager calls would."""
    spec = op.CELLS[0].spec
    target, draft, _, pool, counter = _build(spec, seed=5)
    force = spec["force"]
    # One eager call first, as the engine's warm-up does before capture.
    op(target, draft, 0, None, pool, counter, *force, *STRIDES)
    expected_counter = counter.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_counts = op(target, draft, 0, None, pool, counter, *force, *STRIDES)
    for _ in range(8):
        expected, _, expected_counter = op.reference(
            target, draft, 0, None, pool, expected_counter, *force, *STRIDES
        )
        graph.replay()
        torch.testing.assert_close(graph_counts, expected, rtol=0, atol=0)
        torch.testing.assert_close(counter, expected_counter, rtol=0, atol=0)


def test_guard_refuses_a_wider_rows_buffer() -> None:
    """Rows of the maximum draft length on a step that verifies fewer.

    The op's size check accepts them, and the rows are written with the wrong row stride.
    """
    target, draft, _, pool, counter = _build(op.CELLS[0].spec, seed=3)
    wide = torch.zeros(1, 12, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="exactly"):
        with validating(op):
            op(target, draft, 0, wide, pool, counter, 0, 0, 0, 0.0, *STRIDES)


def test_guard_refuses_a_forced_count_past_the_draft_window() -> None:
    target, draft, _, pool, counter = _build(op.CELLS[0].spec, seed=4)
    with pytest.raises(AssertionError, match="force_max == draft_len"):
        with validating(op):
            op(target, draft, 0, None, pool, counter, 1, 9, 9, 0.0, *STRIDES)
