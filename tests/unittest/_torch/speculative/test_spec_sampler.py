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
"""Unit tests for the one-model spec-decode sampler/acceptance framework:
forced acceptance (``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS``),
rejection-sampling buffer allocation with its fail-closed guard, and the
group-synchronized ``is_all_greedy_sample`` override."""

import os
import types
from typing import Optional
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.speculative.interface import (
    FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR,
    SpecMetadata,
    SpecWorkerBase,
    get_force_num_accepted_tokens,
    get_force_num_accepted_tokens_float,
)

_cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="rejection buffers are CUDA tensors"
)


class _StubSpecWorker(SpecWorkerBase):
    """Concrete ``SpecWorkerBase`` that stubs out the abstract API."""

    @property
    def max_draft_len(self) -> int:
        return 8

    def _forward_impl(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError


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
    "env_value, expected_float, expected_int",
    [
        ("0", 0.0, 0),
        ("3", 3.0, 3),
        ("2.6", 2.6, 0),
        ("0.5", 0.5, 0),
        ("not-a-number", 0.0, 0),
    ],
)
def test_get_force_num_accepted_tokens_float(env_value, expected_float, expected_int):
    """The float helper parses fractional rates; the int helper (used by the
    2-model path) must keep its original behavior on the same values."""
    with patch.dict(os.environ, {FORCE_NUM_ACCEPTED_TOKENS_ENV_VAR: env_value}):
        assert get_force_num_accepted_tokens_float() == pytest.approx(expected_float)
        assert get_force_num_accepted_tokens() == expected_int


# ---------------- _apply_force_accepted_tokens semantics --------------------


def test_zero_value_is_noop():
    """A zero forced rate must leave inputs and RNG state untouched."""
    _require_cuda()
    worker = _make_worker(0.0)
    base = _make_input(batch_size=4)
    before = base.clone()
    out = worker._apply_force_accepted_tokens(base, num_contexts=0, runtime_draft_len=4)
    assert torch.equal(out, before)
    # No RNG state should have been touched in the early-exit path.
    assert worker._force_accept_rng_pool is None
    assert worker._force_accept_rng_counter is None


@pytest.mark.parametrize(
    "value, runtime_draft_len",
    [
        (1.0, 4),
        (2.0, 4),
        (4.0, 4),
        # 10 draft tokens requested but only 2 available -> capped at 3.
        (10.0, 2),
        # Fractional but no room: max_total = 3, base_total = min(10, 3) = 3,
        # so ``base_total < max_total`` is False -> all 3, no RNG draw.
        (9.5, 2),
    ],
)
def test_integer_value_matches_legacy_behavior(value, runtime_draft_len):
    """Integer (or capped) rates take the deterministic path with no RNG pool."""
    _require_cuda()
    worker = _make_worker(value)
    out = worker._apply_force_accepted_tokens(
        _make_input(batch_size=8), num_contexts=0, runtime_draft_len=runtime_draft_len
    )
    expected = min(int(value) + 1, runtime_draft_len + 1)
    assert torch.all(out == expected)
    # The deterministic (non-random) path must not allocate the RNG pool.
    assert worker._force_accept_rng_pool is None


def test_fractional_distribution_matches_target_probability():
    """Fractional draws emit only base/base+1 and average to the target rate."""
    _require_cuda()
    worker = _make_worker(2.6)
    target_frac = 0.6
    n_iters = 200
    batch = 64
    seen = set()
    extra_count = 0
    total = 0
    for _ in range(n_iters):
        out = worker._apply_force_accepted_tokens(
            _make_input(batch_size=batch), num_contexts=0, runtime_draft_len=4
        )
        seen.update(out.unique().tolist())
        extra_count += int((out == 4).sum().item())
        total += batch
    # Either 2 draft + target = 3, or 3 draft + target = 4 -- never anything else.
    assert seen == {3, 4}
    measured = extra_count / total
    assert abs(measured - target_frac) < 0.03, (
        f"measured fraction {measured:.4f} differs from target {target_frac}"
    )


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
    """Two independently-instantiated workers (simulating two TP ranks) must
    produce bit-identical accepted counts for the same call sequence."""
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
    """Synthetic AR draws must still match across ranks even when the default
    CUDA generator state has diverged between them."""
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
    """Capturing the override into a CUDA graph and replaying it must produce
    the exact same per-replay output sequence as running it eagerly."""
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
    """Across replays the captured RNG counter must advance, producing
    different (but deterministic) draws rather than a constant output."""
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


class _FailingSpecWorker(_StubSpecWorker):
    """Stub whose forward saves spec-dec metadata state and then fails."""

    def _forward_impl(self, *args: object, **kwargs: object) -> None:
        attn_metadata = kwargs["attn_metadata"]
        attn_metadata.prepare_for_spec_dec("_seq_lens", "_seq_lens_cuda")
        raise RuntimeError("simulated draft failure")


def test_forward_restores_spec_dec_state_on_failure() -> None:
    """A failure between prepare_for_spec_dec and restore must not leak saved
    attn-metadata state; SpecWorkerBase.forward restores it in its cleanup."""
    _require_cuda()
    attn_metadata = AttentionMetadata(max_num_requests=2, max_num_tokens=16)
    attn_metadata.seq_lens = torch.ones(2, dtype=torch.int32)
    worker = _FailingSpecWorker()
    for _ in range(2):
        # The second iteration would trip the pairing assert inside
        # prepare_for_spec_dec if the first failure had leaked saved state.
        with pytest.raises(RuntimeError, match="simulated draft failure"):
            worker(attn_metadata=attn_metadata, spec_metadata=None)
        assert not attn_metadata.has_spec_dec_saved_state


# ---------------- rejection slot buffers + fail-closed guard ----------------

R, K, V = 8, 4, 32  # max_num_requests, max_draft_len, vocab_size


def _alloc_meta(**over):
    base = dict(
        use_rejection_sampling=True,
        draft_probs=None,
        vocab_size=V,
        draft_vocab_size=V,
        max_num_requests=R,
        max_draft_len=K,
        num_seq_slots=0,
        batch_slot_ids=None,
        full_draft_probs=None,
        draft_probs_vocab_size=0,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


@_cuda_only
def test_prepare_buffers_allocates_when_enabled():
    """When rejection sampling is enabled, slot buffers are allocated;
    full_draft_probs appears only when draft and target vocabs differ."""
    m = _alloc_meta()
    SpecMetadata.prepare_rejection_sampling_buffers(m)
    assert m.draft_probs is not None and tuple(m.draft_probs.shape) == (R + 1, K, V)
    assert m.batch_slot_ids is not None and m.batch_slot_ids.shape[0] == R
    assert m.batch_slot_ids.dtype == torch.long
    assert m.full_draft_probs is None
    assert m.draft_probs_vocab_size == V

    # Distinct draft vocab: full_draft_probs (d2t-expanded) is allocated.
    m = _alloc_meta(draft_vocab_size=V - 1)
    SpecMetadata.prepare_rejection_sampling_buffers(m)
    assert m.full_draft_probs is not None and tuple(m.full_draft_probs.shape) == (R + 1, K, V)


@_cuda_only
def test_prepare_buffers_span_seq_slot_pool():
    """Slot-indexed buffers must span the full seq-slot pool (which can exceed
    max_num_requests) plus a dummy scratch row placed on the last row."""
    pool = 2 * R
    m = _alloc_meta(num_seq_slots=pool)
    SpecMetadata.prepare_rejection_sampling_buffers(m)
    assert tuple(m.draft_probs.shape) == (pool + 1, K, V)
    assert m.dummy_slot_row == pool


@_cuda_only
def test_prepare_buffers_noop_when_disabled():
    """With rejection sampling disabled, no buffers are allocated."""
    m = _alloc_meta(use_rejection_sampling=False)
    SpecMetadata.prepare_rejection_sampling_buffers(m)
    assert m.draft_probs is None
    assert m.batch_slot_ids is None
    assert m.full_draft_probs is None


def _valid_state():
    return types.SimpleNamespace(
        draft_probs=torch.empty((R, K, V), device="cuda"),
        batch_slot_ids=torch.arange(R, device="cuda", dtype=torch.long),
        is_ragged_verify=False,
    )


# num_contexts=0, num_gens=4
_NUM_CTX = 0
_BATCH = 4
_NUM_GENS = _BATCH - _NUM_CTX


def _good_args():
    draft_tokens = torch.zeros((_NUM_GENS, K), dtype=torch.int, device="cuda")
    logits = torch.zeros((_NUM_CTX + _NUM_GENS * (K + 1), V), device="cuda")
    return draft_tokens, K, V, _NUM_CTX, _BATCH, logits


_DUMMY = object()  # _rejection_buffers_valid does not use self


def _call(meta, draft_tokens, draft_len, stored_vocab, num_contexts, batch_size, logits):
    return SpecWorkerBase._rejection_buffers_valid(
        _DUMMY, draft_tokens, draft_len, stored_vocab, num_contexts, batch_size, logits, meta
    )


@_cuda_only
def test_guard_true_on_valid_state():
    """The guard accepts a well-formed buffer/argument combination."""
    assert _call(_valid_state(), *_good_args()) is True


# Each builder returns (meta, *_call args) for one malformed state that the
# fail-closed guard must reject.
def _case_draft_probs_missing():
    m = _valid_state()
    m.draft_probs = None
    return (m, *_good_args())


def _case_batch_slot_ids_missing():
    m = _valid_state()
    m.batch_slot_ids = None
    return (m, *_good_args())


def _case_stored_vocab_nonpositive():
    dt, dl, _sv, nc, bs, lg = _good_args()
    return (_valid_state(), dt, dl, 0, nc, bs, lg)


def _case_stored_vocab_exceeds_buffer():
    dt, dl, _sv, nc, bs, lg = _good_args()
    return (_valid_state(), dt, dl, V + 1, nc, bs, lg)


def _case_draft_len_exceeds_buffer():
    # draft_probs has K steps; ask for K+1.
    draft_tokens = torch.zeros((_NUM_GENS, K + 1), dtype=torch.int, device="cuda")
    logits = torch.zeros((_NUM_CTX + _NUM_GENS * (K + 2), V), device="cuda")
    return (_valid_state(), draft_tokens, K + 1, V, _NUM_CTX, _BATCH, logits)


def _case_draft_tokens_wrong_rows():
    dt = torch.zeros((_NUM_GENS + 1, K), dtype=torch.int, device="cuda")
    _, dl, sv, nc, bs, lg = _good_args()
    return (_valid_state(), dt, dl, sv, nc, bs, lg)


def _case_too_few_logits_rows():
    dt, dl, sv, nc, bs, _lg = _good_args()
    too_few = torch.zeros((nc + _NUM_GENS, V), device="cuda")  # missing +1 each
    return (_valid_state(), dt, dl, sv, nc, bs, too_few)


def _case_batch_slot_ids_too_short():
    m = _valid_state()
    m.batch_slot_ids = torch.arange(_BATCH - 1, device="cuda", dtype=torch.long)
    return (m, *_good_args())


@_cuda_only
@pytest.mark.parametrize(
    "build_case",
    [
        _case_draft_probs_missing,
        _case_batch_slot_ids_missing,
        _case_stored_vocab_nonpositive,
        _case_stored_vocab_exceeds_buffer,
        _case_draft_len_exceeds_buffer,
        _case_draft_tokens_wrong_rows,
        _case_too_few_logits_rows,
        _case_batch_slot_ids_too_short,
    ],
    ids=lambda f: f.__name__.removeprefix("_case_"),
)
def test_guard_false_on_malformed_state(build_case):
    """The fail-closed guard rejects every malformed buffer/argument state."""
    assert _call(*build_case()) is False


# Fail-closed acceptance dispatch: _accept_draft_tokens() must route to
# strict/base acceptance when the buffers are malformed and to the rejection
# method when the state is valid; the acceptance methods are stubbed to
# record which path ran while the real guard logic is exercised.


class _Worker(SpecWorkerBase):
    @property
    def max_draft_len(self) -> int:
        return K

    def _forward_impl(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError


def _dispatch_meta(**over):
    base = dict(
        use_rejection_sampling=True,
        is_all_greedy_sample=False,
        is_ragged_verify=False,
        draft_probs_vocab_size=V,
        draft_probs_last_dim=V,
        batch_slot_ids=torch.arange(R, device="cuda", dtype=torch.long),
        draft_probs=torch.zeros((R, K, V), device="cuda"),
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def _make_dispatch_worker():
    w = _Worker()
    calls = {"base": 0, "rejection": 0}

    def _base(logits, draft_tokens, num_contexts, batch_size, spec_metadata):
        calls["base"] += 1
        return ("base", None)

    def _rej(logits, draft_tokens, draft_probs, num_contexts, batch_size, spec_metadata):
        calls["rejection"] += 1
        return ("rejection", None)

    w._sample_and_accept_draft_tokens_base = _base
    w._sample_and_accept_draft_tokens_rejection = _rej
    return w, calls


@_cuda_only
@pytest.mark.parametrize(
    "meta_over, expected_path",
    [
        # Malformed buffers (draft_probs None): _rejection_buffers_valid must
        # return False, so acceptance fails closed to base and the rejection
        # kernel is skipped.
        (dict(draft_probs=None), "base"),
        # Valid buffers: the rejection path runs.
        (dict(), "rejection"),
        # All-greedy batch: rejection is bypassed regardless of buffers.
        (dict(is_all_greedy_sample=True), "base"),
    ],
    ids=[
        "fails_closed_on_malformed_buffers",
        "routes_to_rejection_on_valid_state",
        "base_when_all_greedy",
    ],
)
def test_accept_dispatch(meta_over, expected_path):
    """Acceptance dispatch fails closed to base on bad state and routes to
    the rejection path only when the buffers are valid and not all-greedy."""
    w, calls = _make_dispatch_worker()
    meta = _dispatch_meta(**meta_over)
    draft_tokens = torch.zeros((_NUM_GENS, K), dtype=torch.int, device="cuda")
    logits = torch.zeros((_NUM_CTX + _NUM_GENS * (K + 1), V), device="cuda")
    out = w._accept_draft_tokens(logits, draft_tokens, _NUM_CTX, _BATCH, meta)
    assert out == (expected_path, None)
    other = "rejection" if expected_path == "base" else "base"
    assert calls[expected_path] == 1 and calls[other] == 0


# ---------------- group-synchronized is_all_greedy_sample (CPU) -------------


def _fake_request(temperature=None, top_k=None, top_p=None, slot=0):
    return types.SimpleNamespace(
        sampling_config=types.SimpleNamespace(
            temperature=[temperature] if temperature is not None else None,
            top_k=[top_k] if top_k is not None else None,
            top_p=[top_p] if top_p is not None else None,
        ),
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_seq_slot=slot,
    )


def _fake_meta(group_all_greedy_sample=None, force_capture=False):
    return types.SimpleNamespace(
        runtime_draft_len=2,
        dummy_slot_row=0,
        group_all_greedy_sample=group_all_greedy_sample,
        _force_non_greedy_for_capture=force_capture,
    )


def _scan(meta, requests):
    return SpecMetadata._scan_one_model_sampling(meta, requests)


def test_local_value_used_when_no_group_sync():
    """Without a group-synced value, the scan uses the rank-local flag."""
    meta = _fake_meta(group_all_greedy_sample=None)
    _scan(meta, [_fake_request(), _fake_request()])
    assert meta.is_all_greedy_sample is True

    _scan(meta, [_fake_request(), _fake_request(temperature=0.8)])
    assert meta.is_all_greedy_sample is False


def test_group_override_pulls_greedy_rank_onto_advanced_path():
    """The group AND overrides the rank-local all-greedy flag, and composes
    with the warmup capture force without changing the outcome."""
    cases = [
        (False, False, False),
        (True, False, True),
        (False, True, False),
    ]
    for group_value, force_capture, expected in cases:
        meta = _fake_meta(group_all_greedy_sample=group_value,
                          force_capture=force_capture)
        _scan(meta, [_fake_request(), _fake_request()])
        assert meta.is_all_greedy_sample is expected, (group_value,
                                                       force_capture)


def test_group_override_survives_rescan():
    """The group override must keep applying on every rescan after the CUDA
    graph key is built."""
    meta = _fake_meta(group_all_greedy_sample=False)
    for _ in range(3):
        _scan(meta, [_fake_request()])
        assert meta.is_all_greedy_sample is False


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
