# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the synthetic forced acceptance rate in 1-model spec dec.

These tests exercise ``SpecWorkerBase._apply_force_accepted_tokens``, which
honors the ``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`` environment
variable. The implementation must be:

  - Faithful to fractional rates: integer part always accepts, fractional
    part acts as the per-iteration probability of accepting one extra draft
    token.
  - Tensor-parallel deterministic: identical seed/counter state across
    ranks so all ranks agree on per-request accepted counts (otherwise
    downstream collectives expecting matching shapes deadlock).
  - CUDA-graph compatible: pure-tensor ops with statically-known shapes,
    captureable and replayable.
"""

import os
from typing import Optional
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.speculative.interface import (
    FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR,
    SpecWorkerBase,
    get_force_num_accepted_tokens,
    get_force_num_accepted_tokens_float,
)


class _StubSpecWorker(SpecWorkerBase):
    """Concrete ``SpecWorkerBase`` that stubs out the only abstract API.

    Used purely to drive ``_apply_force_accepted_tokens`` in isolation.
    """

    @property
    def max_draft_len(self) -> int:
        return 8


def _make_worker(value: Optional[float] = None) -> _StubSpecWorker:
    worker = _StubSpecWorker()
    if value is not None:
        worker.force_num_accepted_tokens = value
    return worker


def _make_input(batch_size: int, device: str = "cuda") -> torch.Tensor:
    # Initial counts: 1 (target token, no draft accepts yet). The forced
    # override only ever rewrites entries from ``num_contexts:`` onward.
    return torch.ones(batch_size, dtype=torch.int32, device=device)


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for this test.")


# ---------------- env-var parsing helpers -----------------------------------


@pytest.mark.parametrize(
    "env_value, expected",
    [
        ("0", 0.0),
        ("3", 3.0),
        ("2.6", 2.6),
        ("0.5", 0.5),
        ("not-a-number", 0.0),
    ],
)
def test_get_force_num_accepted_tokens_float(env_value, expected):
    with patch.dict(os.environ, {FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR: env_value}):
        assert get_force_num_accepted_tokens_float() == pytest.approx(expected)


@pytest.mark.parametrize(
    "env_value, expected",
    [
        ("0", 0),
        ("3", 3),
        ("2.6", 0),
    ],
)
def test_get_force_num_accepted_tokens_int_unchanged(env_value, expected):
    """The int helper must keep its original behavior (used by 2-model)."""
    with patch.dict(os.environ, {FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR: env_value}):
        assert get_force_num_accepted_tokens() == expected


# ---------------- _apply_force_accepted_tokens semantics --------------------


def test_zero_value_is_noop():
    _require_cuda()
    worker = _make_worker(0.0)
    base = _make_input(batch_size=4)
    before = base.clone()
    out = worker._apply_force_accepted_tokens(base, num_contexts=0, runtime_draft_len=4)
    assert torch.equal(out, before)
    # No RNG state should have been touched in the early-exit path.
    assert worker._force_accept_rng_pool is None
    assert worker._force_accept_rng_counter is None


@pytest.mark.parametrize("value", [1.0, 2.0, 4.0])
def test_integer_value_matches_legacy_behavior(value):
    _require_cuda()
    worker = _make_worker(value)
    out = worker._apply_force_accepted_tokens(
        _make_input(batch_size=8), num_contexts=0, runtime_draft_len=4
    )
    expected = min(int(value) + 1, 4 + 1)
    assert torch.all(out == expected)
    # Pure-integer path must not allocate the RNG pool.
    assert worker._force_accept_rng_pool is None


def test_integer_value_caps_at_runtime_draft_len_plus_one():
    _require_cuda()
    worker = _make_worker(10.0)
    out = worker._apply_force_accepted_tokens(
        _make_input(batch_size=4), num_contexts=0, runtime_draft_len=2
    )
    # 10 draft tokens requested but only 2 available → all 3 (= 2 + target).
    assert torch.all(out == 3)


def test_fractional_only_emits_two_values():
    _require_cuda()
    worker = _make_worker(2.6)
    seen = set()
    for _ in range(50):
        out = worker._apply_force_accepted_tokens(
            _make_input(batch_size=64), num_contexts=0, runtime_draft_len=4
        )
        seen.update(out.unique().tolist())
    # Either 2 draft + target = 3, or 3 draft + target = 4.
    assert seen == {3, 4}


def test_fractional_distribution_matches_target_probability():
    """Average over many draws should converge to the target fraction."""
    _require_cuda()
    worker = _make_worker(2.6)
    target_frac = 0.6
    n_iters = 200
    batch = 64
    extra_count = 0
    total = 0
    for _ in range(n_iters):
        out = worker._apply_force_accepted_tokens(
            _make_input(batch_size=batch), num_contexts=0, runtime_draft_len=4
        )
        extra_count += int((out == 4).sum().item())
        total += batch
    measured = extra_count / total
    # Pool-based RNG is deterministic; the empirical mean over ~13k draws
    # is tight against 0.6.
    assert abs(measured - target_frac) < 0.03, (
        f"measured fraction {measured:.4f} differs from target {target_frac}"
    )


def test_fractional_capped_when_no_room():
    """If ``int_part + 1`` already saturates the cap, no extra is granted."""
    _require_cuda()
    worker = _make_worker(9.5)
    out = worker._apply_force_accepted_tokens(
        _make_input(batch_size=8), num_contexts=0, runtime_draft_len=2
    )
    # max_total = runtime_draft_len + 1 = 3, base_total = min(10, 3) = 3.
    # frac_part > 0 but ``base_total < max_total`` is False → all 3, no RNG.
    assert torch.all(out == 3)
    assert worker._force_accept_rng_pool is None


def test_num_contexts_offset_is_respected():
    """Context (prefill) rows must be left untouched by the override."""
    _require_cuda()
    worker = _make_worker(2.6)
    batch_size = 8
    num_contexts = 3
    base = _make_input(batch_size=batch_size)
    sentinel = torch.iinfo(base.dtype).max
    base[:num_contexts] = sentinel
    out = worker._apply_force_accepted_tokens(base, num_contexts=num_contexts, runtime_draft_len=4)
    assert torch.all(out[:num_contexts] == sentinel)
    assert torch.all((out[num_contexts:] == 3) | (out[num_contexts:] == 4))


# ---------------- TP determinism --------------------------------------------


def test_tp_determinism_across_independent_workers():
    """
    Two independently-instantiated workers (simulating two TP ranks) must
    produce bit-identical accepted-token counts when fed the same call
    sequence. This is the property whose absence used to hang TP>1.
    """
    _require_cuda()
    rank0 = _make_worker(2.6)
    rank1 = _make_worker(2.6)
    # Use a non-power-of-two batch to make accidental shape coincidences
    # less likely to mask divergence.
    for _ in range(32):
        out0 = rank0._apply_force_accepted_tokens(
            _make_input(batch_size=33), num_contexts=0, runtime_draft_len=4
        )
        out1 = rank1._apply_force_accepted_tokens(
            _make_input(batch_size=33), num_contexts=0, runtime_draft_len=4
        )
        assert torch.equal(out0, out1)


def test_tp_determinism_survives_default_generator_drift():
    """
    Even if the default CUDA generator state diverges between ranks (e.g.
    because some other component consumed random numbers), the synthetic
    AR draws must still match. This is the regression guard for the
    original bug, where ``torch.rand`` against the default generator gave
    each rank a different sequence.
    """
    _require_cuda()
    rank0 = _make_worker(2.6)
    rank1 = _make_worker(2.6)

    def _advance_default_generator(seed: int):
        # Seed the default CUDA generator on the same device differently
        # for each "rank" and consume some randoms, simulating drift.
        torch.cuda.manual_seed(seed)
        _ = torch.rand(128, device="cuda")

    _advance_default_generator(seed=11)
    out0 = rank0._apply_force_accepted_tokens(
        _make_input(batch_size=33), num_contexts=0, runtime_draft_len=4
    )
    _advance_default_generator(seed=999)
    out1 = rank1._apply_force_accepted_tokens(
        _make_input(batch_size=33), num_contexts=0, runtime_draft_len=4
    )
    assert torch.equal(out0, out1)


# ---------------- CUDA graph capture/replay ---------------------------------


def test_cuda_graph_capture_and_replay_match_eager():
    """
    Capturing the override into a CUDA graph and replaying it must produce
    the exact same per-replay output sequence as running it eagerly.
    """
    _require_cuda()

    n_iters = 8
    batch_size = 16
    runtime_draft_len = 4
    value = 2.6

    # 1) Eager reference.
    eager_worker = _make_worker(value)
    eager_outputs = []
    for _ in range(n_iters):
        out = eager_worker._apply_force_accepted_tokens(
            _make_input(batch_size=batch_size), num_contexts=0, runtime_draft_len=runtime_draft_len
        )
        eager_outputs.append(out.clone())

    # 2) Captured-graph run, aligned to the eager reference.
    graph_worker = _make_worker(value)
    static_input = _make_input(batch_size=batch_size)
    # Eager warmup: forces lazy RNG state allocation OUTSIDE capture.
    graph_worker._apply_force_accepted_tokens(
        static_input, num_contexts=0, runtime_draft_len=runtime_draft_len
    )
    # Realign the device-side counter so the captured graph's first replay
    # advances 0 → 1, matching the eager loop's first iteration.
    graph_worker._force_accept_rng_counter.zero_()

    static_input.fill_(1)
    static_output = torch.empty_like(static_input)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_worker._apply_force_accepted_tokens(
            static_input, num_contexts=0, runtime_draft_len=runtime_draft_len
        )
        # ``_apply_force_accepted_tokens`` mutates ``static_input`` in place;
        # mirror it into ``static_output`` so we can snapshot per replay.
        static_output.copy_(static_input)

    graph_outputs = []
    for _ in range(n_iters):
        # Reset input each replay so the override semantics match eager
        # (which also starts from all-1s every iteration).
        static_input.fill_(1)
        graph.replay()
        torch.cuda.synchronize()
        graph_outputs.append(static_output.clone())

    for i, (eager, graphed) in enumerate(zip(eager_outputs, graph_outputs)):
        assert torch.equal(eager, graphed), (
            f"Iteration {i}: eager={eager.tolist()} vs graphed={graphed.tolist()}"
        )


def test_cuda_graph_replay_advances_rng_state():
    """
    Across replays, the captured counter must advance, producing different
    (but deterministic) draws. A constant output across replays would mean
    the counter increment was not captured.
    """
    _require_cuda()

    worker = _make_worker(0.5)
    batch_size = 64
    runtime_draft_len = 4

    static_input = _make_input(batch_size=batch_size)
    worker._apply_force_accepted_tokens(
        static_input, num_contexts=0, runtime_draft_len=runtime_draft_len
    )
    worker._force_accept_rng_counter.zero_()

    static_input.fill_(1)
    static_output = torch.empty_like(static_input)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        worker._apply_force_accepted_tokens(
            static_input, num_contexts=0, runtime_draft_len=runtime_draft_len
        )
        static_output.copy_(static_input)

    snapshots = []
    for _ in range(8):
        static_input.fill_(1)
        graph.replay()
        torch.cuda.synchronize()
        snapshots.append(static_output.clone())

    # At least two replays must differ; otherwise the captured counter is
    # not advancing (the original bug-class).
    assert any(not torch.equal(snapshots[0], s) for s in snapshots[1:]), (
        "Captured graph produced identical outputs on every replay — "
        "RNG counter is not advancing inside the captured graph."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
