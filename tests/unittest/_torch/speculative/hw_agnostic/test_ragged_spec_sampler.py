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
"""Ragged one-model speculative sampling and penalty tests."""

import types
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.sampler import penalties as penalty_ops
from tensorrt_llm._torch.speculative import interface as interface_ops
from tensorrt_llm._torch.speculative import spec_sampler_base as sampler_ops
from tensorrt_llm._torch.speculative.interface import SpecMetadata, SpecWorkerBase
from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSampler

pytestmark = pytest.mark.cpu_only


class _StubSpecWorker(SpecWorkerBase):
    """Concrete ``SpecWorkerBase`` that stubs out the abstract API."""

    @property
    def max_draft_len(self) -> int:
        return 8

    def _forward_impl(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError


def _make_worker(value: Optional[float] = None) -> _StubSpecWorker:
    # These tests cover CPU layout/acceptance, not native backend initialization.
    worker = _StubSpecWorker.__new__(_StubSpecWorker)
    torch.nn.Module.__init__(worker)
    worker.force_num_accepted_tokens = 0.0 if value is None else value
    worker._d2t = None
    return worker


def _penalty_mapping_meta(
    slot_ids: list[int], verify_lens: Optional[list[int]] = None
) -> types.SimpleNamespace:
    metadata = types.SimpleNamespace(
        batch_slot_ids=torch.tensor(slot_ids, dtype=torch.int64),
        is_ragged_verify=verify_lens is not None,
        verify_lens=None,
        qo_indptr=None,
        total_verify_tokens=None,
    )
    if verify_lens is not None:
        lens = torch.tensor(verify_lens, dtype=torch.int32)
        metadata.verify_lens = lens
        metadata.qo_indptr = torch.cat(
            [torch.zeros(1, dtype=torch.int32), torch.cumsum(lens, dim=0)]
        )
        metadata.total_verify_tokens = sum(verify_lens)
    return metadata


def test_occurrence_penalty_uniform_row_mapping_is_unchanged():
    """An all-full ragged layout must reproduce the existing uniform mapping."""
    draft_len = 4
    draft_tokens = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]])
    uniform = _penalty_mapping_meta([3, 7, 9])
    ragged = _penalty_mapping_meta([3, 7, 9], [draft_len + 1, draft_len + 1])
    num_rows = 1 + 2 * (draft_len + 1)

    uniform_mapping = penalty_ops.build_row_mapping(
        uniform,
        num_contexts=1,
        batch_size=3,
        draft_len=draft_len,
        draft_tokens=draft_tokens,
        device=torch.device("cpu"),
        num_logit_rows=num_rows,
    )
    ragged_mapping = penalty_ops.build_row_mapping(
        ragged,
        num_contexts=1,
        batch_size=3,
        draft_len=draft_len,
        draft_tokens=draft_tokens,
        device=torch.device("cpu"),
        num_logit_rows=num_rows,
    )

    assert uniform_mapping is not None and ragged_mapping is not None
    for uniform_tensor, ragged_tensor in zip(uniform_mapping, ragged_mapping):
        assert torch.equal(uniform_tensor, ragged_tensor)


def test_occurrence_penalty_ragged_row_mapping_uses_each_request_window():
    """Packed rows must stay with their request and see only its earlier drafts."""
    draft_tokens = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]])
    metadata = _penalty_mapping_meta([3, 7, 9], [2, 4])

    mapping = penalty_ops.build_row_mapping(
        metadata,
        num_contexts=1,
        batch_size=3,
        draft_len=4,
        draft_tokens=draft_tokens,
        device=torch.device("cpu"),
        num_logit_rows=7,
    )

    assert mapping is not None
    row_slots, intra_tokens, intra_valid = mapping
    assert row_slots.tolist() == [3, 7, 7, 9, 9, 9, 9]
    assert intra_tokens.tolist() == [
        [0, 0, 0, 0],
        [11, 12, 13, 14],
        [11, 12, 13, 14],
        [21, 22, 23, 24],
        [21, 22, 23, 24],
        [21, 22, 23, 24],
        [21, 22, 23, 24],
    ]
    assert intra_valid.tolist() == [
        [False, False, False, False],
        [False, False, False, False],
        [True, False, False, False],
        [False, False, False, False],
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
    ]


def test_occurrence_penalty_applies_to_ragged_packed_rows(monkeypatch):
    """The worker must run, rather than silently skip, the ragged penalty pass."""
    metadata = _penalty_mapping_meta([3, 7, 9], [2, 4])
    metadata.enable_penalty = True
    captured = {}

    def _apply(logits, spec_metadata, row_slots, intra_tokens, intra_valid):
        captured["row_slots"] = row_slots.clone()
        captured["intra_tokens"] = intra_tokens.clone()
        captured["intra_valid"] = intra_valid.clone()
        logits[:, 0].copy_(row_slots.to(logits.dtype))

    monkeypatch.setattr(penalty_ops, "apply_penalties", _apply)
    logits = torch.zeros((7, 3))
    draft_tokens = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]])

    penalized = _make_worker()._apply_occurrence_penalties(
        logits, draft_tokens, num_contexts=1, batch_size=3, spec_metadata=metadata
    )

    assert torch.equal(logits, torch.zeros_like(logits))
    assert penalized[:, 0].tolist() == [3.0, 7.0, 7.0, 9.0, 9.0, 9.0, 9.0]
    assert captured["row_slots"].numel() == logits.shape[0]
    assert captured["intra_valid"].sum().item() == 7


def test_legacy_ragged_without_executed_windows_is_rejected(cpu_sampler):
    sampler, _ = cpu_sampler
    scheduled, outputs, *_ = _sampling_step()
    with pytest.raises(ValueError, match="ragged producers"):
        sampler.sample_async(scheduled, outputs, [])


def test_uniform_verify_window_keeps_runtime_draft_length(cpu_sampler):
    sampler, _ = cpu_sampler
    scheduled, outputs, _, first, _ = _sampling_step()
    outputs["native_uniform_verify"] = True
    state = sampler.sample_async(scheduled, outputs, [])
    sampler.update_requests(state)
    assert state.host.verify_lens is None
    assert first.py_num_draft_tokens_verified == 3
    assert first.py_rewind_len == 2


@pytest.mark.parametrize("force_accept", [0.0, 1.0, 3.0])
def test_ragged_strict_acceptance_stops_at_each_request_window(force_accept):
    worker = _make_worker(force_accept)
    worker._sample_tokens_for_batch = lambda *args: torch.tensor(
        [1, 100, 2, 8, 4, 101], dtype=torch.int64
    )
    metadata = _penalty_mapping_meta([0, 1], [2, 4])
    metadata.is_cuda_graph = False
    draft_tokens = torch.tensor([[1, 9, 9], [2, 3, 4]], dtype=torch.int64)

    _, accepted = worker._sample_and_accept_draft_tokens_base(
        logits=torch.zeros((6, 8)),
        draft_tokens=draft_tokens,
        num_contexts=0,
        batch_size=2,
        spec_metadata=metadata,
    )

    assert accepted.tolist() == ([2, 4] if force_accept == 3.0 else [2, 2])


@pytest.mark.parametrize(
    "windows, packed_drafts, expected",
    [
        ([1, 3], [21, 22], [[0, 0], [21, 22]]),
        ([1, 1], [], [[0, 0], [0, 0]]),
    ],
)
def test_packed_drafts_allow_anchor_only_requests(windows, packed_drafts, expected):
    metadata = _penalty_mapping_meta([0, 1], windows)
    # The trailing capacity is stale and must not become another request's draft.
    metadata.draft_tokens = torch.tensor([*packed_drafts, 99, 99], dtype=torch.int32)

    padded = interface_ops._padded_gen_draft_tokens(metadata, num_gens=2, runtime_draft_len=2)

    assert padded.dtype == torch.int32
    assert padded.tolist() == expected


def test_ragged_rejection_guard_uses_packed_logit_count():
    metadata = types.SimpleNamespace(
        draft_probs=torch.empty((2, 3, 8)),
        batch_slot_ids=torch.arange(2, dtype=torch.long),
        is_ragged_verify=True,
        verify_lens=torch.tensor([2, 4], dtype=torch.int32),
        qo_indptr=torch.tensor([0, 2, 6], dtype=torch.int32),
        total_verify_tokens=6,
    )
    draft_tokens = torch.zeros((2, 3), dtype=torch.int64)
    valid_logits = torch.zeros((6, 8))
    short_logits = torch.zeros((5, 8))

    assert SpecWorkerBase._rejection_buffers_valid(
        object(), draft_tokens, 3, 8, 0, 2, valid_logits, metadata
    )
    assert not SpecWorkerBase._rejection_buffers_valid(
        object(), draft_tokens, 3, 8, 0, 2, short_logits, metadata
    )


@pytest.mark.parametrize("packed_verify", [False, True])
def test_padding_mask_distinguishes_target_windows_from_draft_rows(packed_verify):
    metadata = _penalty_mapping_meta([8, 3, 9], [2, 4])
    metadata.dummy_slot_row = 9
    logits = torch.tensor([[1.0, 2.0], [float("nan"), float("inf")]])
    expected = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    if packed_verify:
        logits = logits.repeat_interleave(torch.tensor([2, 4]), dim=0)
        expected = expected.repeat_interleave(torch.tensor([2, 4]), dim=0)
    result = SpecWorkerBase._zero_padding_rows(
        logits,
        metadata,
        1,
        3,
        1 if not packed_verify else 4,
        packed_verify=packed_verify,
    )
    assert torch.equal(result, expected)


def test_ragged_padding_rejects_incomplete_layout():
    metadata = _penalty_mapping_meta([3, 9], [2, 4])
    metadata.dummy_slot_row = 9
    metadata.total_verify_tokens = 5
    with pytest.raises(ValueError, match="verification layout"):
        SpecWorkerBase._zero_padding_rows(torch.zeros(6, 8), metadata, 0, 2, 4, packed_verify=True)


@pytest.mark.parametrize("ragged", [False, True])
def test_sampling_parameter_rows_follow_metadata_not_stale_request_windows(ragged):
    config = types.SimpleNamespace(temperature=0.7, top_k=4, top_p=0.8, min_p=0.1)
    requests = [
        types.SimpleNamespace(
            sampling_config=config, state=state, py_seq_slot=slot, py_verify_len=window
        )
        for slot, (state, window) in enumerate(
            [
                (LlmRequestState.CONTEXT_INIT, 5),
                (LlmRequestState.GENERATION_IN_PROGRESS, 1),
                (LlmRequestState.GENERATION_IN_PROGRESS, 3),
            ]
        )
    ]
    metadata = types.SimpleNamespace(
        runtime_draft_len=5,
        is_ragged_verify=ragged,
        dummy_slot_row=9,
        group_all_greedy_sample=None,
    )
    normalized, slots = SpecMetadata._scan_one_model_sampling(metadata, requests)
    assert [row[-1] for row in normalized] == ([1, 2, 4] if ragged else [1, 6, 6])
    assert [row[3] for row in normalized] == [0.1] * 3
    assert slots == [0, 1, 2]


@pytest.mark.parametrize(
    "native, host, device",
    [
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (1, False, False),
        (False, 1, False),
    ],
)
def test_conflicting_window_authorities_are_rejected(cpu_sampler, native, host, device):
    sampler, _ = cpu_sampler
    scheduled, outputs, *_ = _sampling_step()
    outputs["native_uniform_verify"] = native
    outputs["host_policy_windows_snapshot"] = host
    if device:
        outputs["verify_lens"] = torch.tensor([1, 1, 3, 4, 1], dtype=torch.int32)
        outputs["verify_lens_in_output_order"] = True
    with pytest.raises(ValueError):
        sampler.sample_async(scheduled, outputs, [])


def test_native_uniform_marker_ignores_host_policy_window(cpu_sampler):
    sampler, _ = cpu_sampler
    scheduled, outputs, _, first, _ = _sampling_step()
    outputs["native_uniform_verify"] = True
    state = sampler.sample_async(scheduled, outputs, [])
    first.py_verify_len = 1
    sampler.update_requests(state)
    assert state.host.verify_lens is None
    assert first.py_num_draft_tokens_verified == 3


def test_device_window_wins_over_host_shape_split(cpu_sampler):
    sampler, _ = cpu_sampler
    scheduled, outputs, _, first, _ = _sampling_step()
    first.py_verify_len = 5
    outputs["verify_lens"] = torch.tensor([1, 1, 4, 3, 1], dtype=torch.int32)
    outputs["verify_lens_in_output_order"] = True
    state = sampler.sample_async(scheduled, outputs, [])
    first.py_verify_len = 1
    sampler.update_requests(state)
    assert first.py_num_draft_tokens_verified == 3
    assert first.py_rewind_len == 2


def test_legacy_host_snapshot_marker_is_rejected_before_request_updates(cpu_sampler):
    sampler, transfers = cpu_sampler
    scheduled, outputs, context, first, last = _sampling_step()
    outputs["host_policy_windows_snapshot"] = True
    with pytest.raises(ValueError, match="legacy ragged producers"):
        sampler.sample_async(scheduled, outputs, [])
    assert transfers == []
    assert context.tokens == first.tokens == last.tokens == []


@pytest.mark.parametrize("ragged", [False, True])
def test_rejection_preserves_sampling_params_and_caps_acceptance(monkeypatch, ragged):
    """Exercise packing around the backend boundary, without a CUDA kernel."""
    worker = _make_worker(0.0)
    widths = [2, 4] if ragged else [4, 4]
    metadata = _penalty_mapping_meta([4, 2, 9], widths if ragged else None)
    metadata.dummy_slot_row = 9
    rows = 1 + sum(widths)
    metadata.temperatures = torch.arange(rows, dtype=torch.float32) + 0.5
    metadata.top_ks = torch.arange(rows, dtype=torch.int32) + 1
    metadata.top_ps = torch.full((rows,), 0.9)
    metadata.min_ps = torch.arange(rows, dtype=torch.float32) / 100
    metadata.advanced_sampling_mode = object()
    logits = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8)
    logits[1 + widths[0] :] = float("nan")
    calls = {}

    def compute_probs(mode, scores, temperatures, top_ks, top_ps, min_ps):
        assert mode is metadata.advanced_sampling_mode
        assert torch.equal(temperatures, metadata.temperatures[1:])
        assert torch.equal(top_ks, metadata.top_ks[1:])
        assert torch.equal(top_ps, metadata.top_ps[1:])
        assert torch.equal(min_ps, metadata.min_ps[1:])
        assert torch.equal(scores[: widths[0]], logits[1 : 1 + widths[0]])
        assert torch.equal(scores[widths[0] :], torch.zeros(widths[1], 8))
        return torch.softmax(scores, dim=-1)

    def rejection(**kwargs):
        calls.update(kwargs)
        return torch.zeros(2, 4, dtype=torch.int32), torch.tensor([4, 4])

    worker._sample_tokens_for_batch = lambda *args: torch.tensor([7])
    worker._compute_probs_from_logits = compute_probs
    worker._rng_state_per_request = lambda *args: (None, None)
    monkeypatch.setattr(interface_ops, "rejection_sampling_one_model", rejection)
    draft_probs = torch.full((2, 3, 8), 1 / 8)
    original_probs = draft_probs.clone()
    _, accepted = worker._sample_and_accept_draft_tokens_rejection(
        logits,
        torch.zeros(2, 3, dtype=torch.long),
        draft_probs,
        1,
        3,
        metadata,
    )
    assert accepted.tolist() == [1, *widths]
    assert torch.equal(draft_probs, original_probs)
    if not ragged:
        assert calls["draft_probs"] is draft_probs
    target_probs = calls["target_probs"]
    assert target_probs.shape == (2, 4, 8)
    assert torch.allclose(target_probs.sum(-1), torch.ones(2, 4))
    if ragged:
        assert target_probs[0, 2:, 0].tolist() == [1, 1]
        assert target_probs[0, 2:, 1:].count_nonzero() == 0


def _reference_rejection(
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    accept_uniforms: torch.Tensor,
    recovery_uniforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU probability-law oracle, not a native kernel or RNG implementation."""
    batch, width = draft_token_ids.shape
    output = torch.full((batch, width + 1), -1, dtype=torch.int32)
    lengths = torch.full((batch,), width + 1, dtype=torch.int32)
    active = torch.ones(batch, dtype=torch.bool)
    for pos in range(width + 1):
        q = target_probs[:, pos]
        if pos < width:
            p = draft_probs[:, pos]
            token = draft_token_ids[:, pos].long()
            q_token = q.gather(1, token[:, None]).squeeze(1)
            p_token = p.gather(1, token[:, None]).squeeze(1)
            accept = active & (accept_uniforms * p_token < q_token)
            output[:, pos] = torch.where(accept, token.int(), output[:, pos])
            residual = (q - p).clamp_min(0)
        else:
            accept = torch.zeros_like(active)
            residual = q
        cdf = (residual / residual.sum(-1, keepdim=True).clamp_min(1e-30)).cumsum(-1)
        recovered = (recovery_uniforms[:, None] >= cdf).sum(-1).int()
        rejected = active & ~accept
        output[:, pos] = torch.where(rejected, recovered, output[:, pos])
        lengths = torch.where(rejected, pos + 1, lengths)
        active = accept
    return output, lengths


@pytest.mark.parametrize(
    "windows, vocab_mode, preallocated, ragged",
    [
        ([1], "full", False, True),
        ([1, 2, 3, 4], "full", False, True),
        ([1, 2, 3, 4], "prefix", False, True),
        ([1, 2, 3, 4], "d2t", False, True),
        ([1, 2, 3, 4], "d2t", True, True),
        ([4, 4, 4, 4], "full", False, True),
        ([4, 4, 4, 4], "full", False, False),
    ],
)
def test_rejection_retains_request_local_bonus_distribution(
    monkeypatch, windows, vocab_mode, preallocated, ragged
):
    """All real drafts accept; the retained local bonus must still sample q."""
    width, vocab = 3, 4
    # Exact quadrature for these dyadic probabilities, not random sampling.
    grid = (torch.arange(16, dtype=torch.float64) + 0.5) / 16
    trials, requests = grid.numel() ** 2, len(windows)
    num_gens = trials * requests
    accept_uniforms = grid.repeat_interleave(16 * requests)
    recovery_uniforms = grid.repeat(16).repeat_interleave(requests)
    bonuses = torch.tensor(
        [[0.75, 0.25, 0, 0], [0.5, 0.25, 0.25, 0], [0, 0.25, 0.25, 0.5], [1, 0, 0, 0]]
    )[:requests]
    slots = torch.tensor([4, 1, 3, 2][:requests]).repeat(trials)
    stored_vocab = vocab if vocab_mode == "full" else 2
    slot_probs = torch.zeros(6, width + 2, stored_vocab)
    slot_probs[..., 0] = torch.arange(1, 7)[:, None] / 8
    slot_probs[..., 1] = 1 - slot_probs[..., 0]
    expanded = torch.zeros(num_gens, width, vocab)
    indices = torch.tensor([1, 3]) if vocab_mode == "d2t" else torch.arange(stored_vocab)
    expanded[..., indices] = slot_probs[slots, :width]
    worker = _make_worker()
    if vocab_mode == "d2t":
        # Neither draft-vocabulary ID maps to target token 0.
        worker._d2t = torch.tensor([1, 2])
    worker._sample_tokens_for_batch = lambda *args: torch.tensor([3])
    worker._compute_probs_from_logits = lambda mode, scores, *params: scores.softmax(-1)
    worker._rng_state_per_request = lambda *args: (None, None)
    calls = {}

    def rejection(**kwargs):
        calls.update(kwargs)
        return _reference_rejection(
            kwargs["draft_probs"],
            kwargs["draft_token_ids"],
            kwargs["target_probs"],
            accept_uniforms,
            recovery_uniforms,
        )

    monkeypatch.setattr(interface_ops, "rejection_sampling_one_model", rejection)
    scratch = torch.zeros(num_gens + 2, width + 2, vocab) if preallocated else None
    # Reuse expansion storage with different windows in the same token bucket.
    for local_windows in [windows, windows[::-1]]:
        lens = torch.tensor(local_windows * trials, dtype=torch.int32)
        metadata = _penalty_mapping_meta([5, *slots.tolist()], lens.tolist() if ragged else None)
        metadata.dummy_slot_row = 9
        metadata.runtime_draft_len = width
        metadata.enable_penalty = False
        metadata.use_rejection_sampling = True
        metadata.is_all_greedy_sample = False
        metadata.draft_probs = slot_probs
        metadata.draft_probs_vocab_size = metadata.draft_probs_last_dim = stored_vocab
        metadata.full_draft_probs = scratch
        metadata.d2t_target_indices = None
        packed = torch.cat(
            [
                torch.cat([expanded[i, : length - 1], bonuses[i % requests, None]])
                for i, length in enumerate(lens.tolist())
            ]
        )
        logits = torch.cat([torch.zeros(1, vocab), packed.log()])
        rows = logits.shape[0]
        metadata.temperatures = torch.ones(rows)
        metadata.top_ks = torch.full((rows,), vocab, dtype=torch.int32)
        metadata.top_ps = torch.ones(rows)
        metadata.min_ps = torch.zeros(rows)
        metadata.advanced_sampling_mode = None
        metadata.draft_tokens = torch.cat(
            [torch.ones(int((lens - 1).sum()), dtype=torch.int32), torch.tensor([99, 99])]
        )
        before = [t.clone() for t in (slot_probs, metadata.draft_tokens, logits, lens)]

        def no_host_read(*args, **kwargs):
            raise AssertionError("device data must not be read by the host")

        with monkeypatch.context() as guard:
            for name in ("item", "tolist", "__bool__", "numpy", "cpu"):
                guard.setattr(torch.Tensor, name, no_host_read)
            drafts = (
                interface_ops._padded_gen_draft_tokens(metadata, num_gens, width)
                if ragged
                else torch.ones(num_gens, width, dtype=torch.int32)
            )
            drafts_before = drafts.clone()
            output, lengths = worker._accept_draft_tokens(logits, drafts, 1, num_gens + 1, metadata)

        assert output[0, 0] == 3 and lengths[0] == 1
        assert torch.equal(lengths[1:], lens)
        real = torch.arange(width)[None, :] < (lens - 1)[:, None]
        assert torch.all(output[1:, :width][real] == 1)
        bonus = output[1:].gather(1, (lens.long() - 1)[:, None]).reshape(trials, requests)
        for request in range(requests):
            observed = torch.bincount(bonus[:, request].long(), minlength=vocab) / trials
            torch.testing.assert_close(observed, bonuses[request], rtol=0, atol=0)
        passed_probs = calls["draft_probs"]
        torch.testing.assert_close(passed_probs[real], expanded[real], rtol=0, atol=0)
        if ragged:
            expected_padding = torch.zeros_like(passed_probs[~real])
            expected_padding[:, 0] = 1
            torch.testing.assert_close(passed_probs[~real], expected_padding, rtol=0, atol=0)
        else:
            torch.testing.assert_close(passed_probs, expanded, rtol=0, atol=0)
        for original, actual in zip(before, (slot_probs, metadata.draft_tokens, logits, lens)):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
        assert torch.equal(drafts, drafts_before)
        assert torch.equal(calls["draft_token_ids"], drafts_before)
        if scratch is not None:
            # Synthetic mass at an unmapped ID must never survive in reusable storage.
            torch.testing.assert_close(scratch[:num_gens, :width], expanded, rtol=0, atol=0)
            assert not scratch[:, width:].count_nonzero()
            assert not scratch[num_gens:].count_nonzero()
            assert passed_probs.untyped_storage().data_ptr() != scratch.untyped_storage().data_ptr()


@pytest.fixture
def cpu_sampler(monkeypatch):
    """Replace only CUDA transfers/events and the unrelated stop-criteria API."""
    original_to = torch.Tensor.to

    def to_cpu(tensor, *args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs = {**kwargs, "device": "cpu"}
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to_cpu)
    monkeypatch.setattr(sampler_ops, "handle_stop_criteria", lambda *args, **kwargs: False)
    sampler = SpecSampler.__new__(SpecSampler)
    sampler.draft_len = 5
    sampler.max_accepted_path_len = 6
    sampler.max_seq_len = 128
    sampler.store = types.SimpleNamespace(
        new_tokens=torch.zeros(6, 5, 1, dtype=torch.int32),
        next_new_tokens=torch.zeros(6, 5, 1, dtype=torch.int32),
        next_draft_tokens=torch.zeros(5, 5, dtype=torch.int32),
        new_tokens_lens=torch.zeros(5, dtype=torch.int32),
    )
    transfers = []

    def copy_to_host(tensor):
        transfers.append(("copy", tuple(tensor.shape)))
        return tensor.clone()

    def record_event():
        transfers.append(("record", None))
        return types.SimpleNamespace(synchronize=lambda: transfers.append(("sync", None)))

    sampler._copy_to_host = copy_to_host
    sampler._record_sampler_event = record_event
    return sampler, transfers


def _sampling_step():
    def request(request_id, slot, context=False):
        result = types.SimpleNamespace(
            py_request_id=request_id,
            py_seq_slot=slot,
            state=LlmRequestState.CONTEXT_INIT
            if context
            else LlmRequestState.GENERATION_IN_PROGRESS,
            py_draft_tokens=[] if context else [10, 11, 12],
            py_draft_tokens_effective_len=None,
            py_verify_len=None,
            py_decoding_iter=0,
            tokens=[],
        )
        result.add_new_token = lambda token, beam_idx: result.tokens.append(token)
        return result

    context, first, complete = request(7, 2, True), request(8, 3), request(9, 0)
    first.py_verify_len = 2
    scheduled = types.SimpleNamespace(
        context_requests_chunking=[object()],
        context_requests_last_chunk=[context],
        generation_requests=[first, complete],
    )
    # Include both a skipped context chunk and an unsampled trailing graph row.
    outputs = dict(
        new_tokens=torch.arange(20, dtype=torch.int32).reshape(5, 4),
        new_tokens_lens=torch.tensor([1, 1, 2, 2, 4], dtype=torch.int32),
        next_draft_tokens=torch.arange(15, dtype=torch.int32).reshape(5, 3),
        next_new_tokens=torch.arange(20, dtype=torch.int32).reshape(5, 4),
    )
    return scheduled, outputs, context, first, complete


@pytest.mark.parametrize(
    "authority, expected_verified, expected_rewind",
    [("device", 1, 0), ("uniform", 3, 2)],
)
def test_step_windows_survive_request_and_output_reuse(
    cpu_sampler, authority, expected_verified, expected_rewind
):
    sampler, transfers = cpu_sampler
    scheduled, outputs, context, first, complete = _sampling_step()
    if authority == "device":
        outputs["verify_lens"] = torch.tensor([99, 1, 2, 4, 99], dtype=torch.int32)
        outputs["verify_lens_in_output_order"] = True
    elif authority == "uniform":
        outputs["native_uniform_verify"] = True
    state = sampler.sample_async(scheduled, outputs, [])
    assert state.requests == [context, first, complete]
    assert state.draft_lens == [0, 3, 3]
    assert state.runtime_draft_len == 3
    assert len(transfers) == (5 if authority == "device" else 4)
    assert transfers[-1] == ("record", None)
    if authority == "device":
        assert state.host.verify_lens.tolist() == [1, 2, 4]
        outputs["verify_lens"].fill_(6)
    else:
        assert state.host.verify_lens is None
    # A following iteration mutates both the live request and slot-indexed store.
    first.py_verify_len = 5
    first.py_draft_tokens = [99] * 5
    sampler.store.new_tokens.fill_(99)
    complete.state = LlmRequestState.GENERATION_COMPLETE
    sampler.update_requests(state)
    assert transfers[-1] == ("sync", None)
    assert first.tokens == [8, 9]
    assert first.py_num_accepted_draft_tokens == 1
    assert first.py_num_draft_tokens_verified == expected_verified
    assert first.py_rewind_len == expected_rewind
    assert first.py_draft_tokens == [6, 7, 8]
    assert first.py_decoding_iter == 1
    assert context.py_num_draft_tokens_verified == 0
    assert context.py_rewind_len == (0 if authority == "device" else 3)
    assert complete.tokens == []
    assert complete.py_decoding_iter == 0


@pytest.mark.parametrize(
    "invalid",
    [
        [1, 2, 3, 4],
        torch.ones(4, 1),
        torch.ones(3, dtype=torch.int32),
        torch.ones(4),
        torch.ones(4, dtype=torch.bool),
    ],
)
def test_device_windows_require_request_aligned_integer_vector(cpu_sampler, invalid):
    sampler, transfers = cpu_sampler
    scheduled, outputs, *_ = _sampling_step()
    outputs["verify_lens"] = invalid
    outputs["verify_lens_in_output_order"] = True
    with pytest.raises(ValueError, match="verify_lens"):
        sampler.sample_async(scheduled, outputs, [])
    assert transfers == []


@pytest.mark.parametrize(
    "token_window, num_new_tokens", [(1, 1), (2, 1), (2, 2), (4, 1), (4, 2), (4, 4)]
)
def test_executed_device_window_is_not_host_plan_or_acceptance(
    cpu_sampler, token_window, num_new_tokens
):
    sampler, _ = cpu_sampler
    scheduled, outputs, context, first, _ = _sampling_step()
    # A device redistribution need not preserve the host's draft shape split.
    first.py_verify_len = 0
    outputs["verify_lens"] = torch.tensor([99, 1, token_window, 4, 99], dtype=torch.int32)
    outputs["verify_lens_in_output_order"] = True
    outputs["new_tokens_lens"][2] = num_new_tokens
    state = sampler.sample_async(scheduled, outputs, [])
    first.py_verify_len = 5
    first.py_draft_tokens = [99] * 5
    sampler.update_requests(state)

    assert not hasattr(state, "verify_lens_snapshot")
    assert first.py_num_accepted_draft_tokens == num_new_tokens - 1
    assert first.py_num_draft_tokens_verified == token_window - 1
    assert first.py_rewind_len == token_window - num_new_tokens
    assert len(first.tokens) == num_new_tokens
    assert context.py_num_draft_tokens_verified == 0
    assert context.py_rewind_len == 0


def test_two_pending_steps_keep_separate_window_and_token_copies(cpu_sampler):
    sampler, transfers = cpu_sampler
    scheduled, outputs, _, first, _ = _sampling_step()
    outputs["verify_lens"] = torch.tensor([99, 1, 4, 4, 99], dtype=torch.int32)
    outputs["verify_lens_in_output_order"] = True
    earlier = sampler.sample_async(scheduled, outputs, [])

    # The next step reuses both worker outputs and the slot-indexed sampler store.
    outputs["verify_lens"][2] = 2
    outputs["new_tokens_lens"][2] = 1
    outputs["new_tokens"][2, 0] = 77
    later = sampler.sample_async(scheduled, outputs, [])
    assert earlier.host.verify_lens.data_ptr() != later.host.verify_lens.data_ptr()
    outputs["verify_lens"].fill_(99)
    outputs["new_tokens"].fill_(99)
    first.py_verify_len = 5

    sampler.update_requests(earlier)
    assert first.tokens == [8, 9]
    assert first.py_num_draft_tokens_verified == 3
    assert first.py_rewind_len == 2
    sampler.update_requests(later)
    assert first.tokens == [8, 9, 77]
    assert first.py_num_accepted_draft_tokens == 0
    assert first.py_num_draft_tokens_verified == 1
    assert first.py_rewind_len == 1
    assert sum(event == "sync" for event, _ in transfers) == 2


def test_completed_request_slot_reuse_does_not_rebind_pending_windows(cpu_sampler):
    sampler, _ = cpu_sampler
    scheduled, outputs, _, first, complete = _sampling_step()
    outputs["verify_lens"] = torch.tensor([99, 1, 2, 4, 99], dtype=torch.int32)
    outputs["verify_lens_in_output_order"] = True
    earlier = sampler.sample_async(scheduled, outputs, [])
    complete.state = LlmRequestState.GENERATION_COMPLETE

    next_scheduled, next_outputs, _, _, replacement = _sampling_step()
    replacement.py_request_id = 10
    assert replacement.py_seq_slot == complete.py_seq_slot
    next_outputs["verify_lens"] = torch.tensor([99, 1, 4, 2, 99], dtype=torch.int32)
    next_outputs["verify_lens_in_output_order"] = True
    later = sampler.sample_async(next_scheduled, next_outputs, [])
    sampler.update_requests(earlier)
    assert first.py_num_draft_tokens_verified == 1
    assert complete.tokens == []
    assert replacement.tokens == []
    sampler.update_requests(later)
    assert replacement.py_num_draft_tokens_verified == 1
    assert replacement.py_rewind_len == 0


@pytest.mark.parametrize(
    "output_row, token_window, num_new_tokens, effective_len",
    [
        (2, 0, 2, None),
        (2, -1, 2, None),
        (2, 1, 2, None),
        (2, 2, 0, None),
        (2, 2, -1, None),
        (3, 1, 2, None),
        (2, 4, 3, 1),
        (1, 4, 2, None),
        (3, 8, 7, None),
    ],
)
def test_invalid_executed_windows_fail_before_request_updates(
    cpu_sampler, output_row, token_window, num_new_tokens, effective_len
):
    sampler, _ = cpu_sampler
    scheduled, outputs, context, first, last = _sampling_step()
    first.py_draft_tokens_effective_len = effective_len
    outputs["verify_lens"] = torch.tensor([99, 1, 3, 4, 99], dtype=torch.int32)
    outputs["verify_lens_in_output_order"] = True
    outputs["verify_lens"][output_row] = token_window
    outputs["new_tokens_lens"][output_row] = num_new_tokens
    state = sampler.sample_async(scheduled, outputs, [])

    with pytest.raises(ValueError, match="executed verify_lens"):
        sampler.update_requests(state)

    assert context.tokens == first.tokens == last.tokens == []
    assert context.py_decoding_iter == first.py_decoding_iter == last.py_decoding_iter == 0


def test_invalid_executed_window_for_completed_request_is_not_consumed(cpu_sampler):
    sampler, _ = cpu_sampler
    scheduled, outputs, _, first, complete = _sampling_step()
    outputs["verify_lens"] = torch.tensor([99, 1, 3, -1, 99], dtype=torch.int32)
    outputs["verify_lens_in_output_order"] = True
    state = sampler.sample_async(scheduled, outputs, [])
    complete.state = LlmRequestState.GENERATION_COMPLETE
    sampler.update_requests(state)

    assert first.py_num_draft_tokens_verified == 2
    assert first.py_rewind_len == 1
    assert complete.tokens == []


@pytest.mark.parametrize("contexts,generations", [(0, 3), (2, 3), (3, 0), (0, 0)])
def test_publisher_uses_explicit_row_counts_without_host_sync(monkeypatch, contexts, generations):
    from tensorrt_llm._torch.speculative.dspark import _publish_policy_window_output

    batch = contexts + generations
    outputs = {
        name: torch.zeros((batch, 4), dtype=torch.int32)
        for name in ("new_tokens", "next_new_tokens", "next_draft_tokens")
    }
    outputs["new_tokens_lens"] = torch.ones(batch, dtype=torch.int32)
    executed = torch.arange(1, generations + 1, dtype=torch.int32)
    storage = torch.full((batch + 3,), -1, dtype=torch.int32)
    pointer = storage.data_ptr()

    def forbidden(*args, **kwargs):
        raise AssertionError("publisher must not allocate or read device values on host")

    with monkeypatch.context() as guard:
        for name in ("item", "tolist", "__bool__", "numpy", "cpu", "clone", "to"):
            guard.setattr(torch.Tensor, name, forbidden)
        for name in ("empty", "zeros", "ones", "full", "tensor", "cat", "stack"):
            guard.setattr(torch, name, forbidden)
        _publish_policy_window_output(
            outputs,
            executed,
            num_contexts=contexts,
            num_generations=generations,
            output_verify_lens=storage,
        )
    assert outputs["verify_lens_in_output_order"] is True
    assert outputs["verify_lens"].tolist() == [1] * contexts + list(range(1, generations + 1))
    assert storage.data_ptr() == pointer
    assert storage[batch:].tolist() == [-1] * 3
    if batch:
        assert outputs["verify_lens"].data_ptr() == pointer


@pytest.mark.parametrize(
    "invalid", ["capacity", "short", "dtype", "dimensions", "storage", "missing_storage", "rows"]
)
def test_publisher_rejects_ambiguous_capacity_and_bad_shapes(invalid):
    from tensorrt_llm._torch.speculative.dspark import _publish_policy_window_output

    _, outputs, *_ = _sampling_step()
    executed = torch.tensor([3, 4, 1], dtype=torch.int32)
    storage = torch.empty(8, dtype=torch.int32)
    if invalid == "capacity":
        executed = torch.tensor([3, 4, 1, 99, 99], dtype=torch.int32)
    elif invalid == "short":
        executed = executed[:2]
    elif invalid == "dtype":
        executed = executed.long()
    elif invalid == "dimensions":
        executed = executed[:, None]
    elif invalid == "storage":
        storage = storage[:4]
    elif invalid == "missing_storage":
        storage = None
    else:
        outputs["next_new_tokens"] = outputs["next_new_tokens"][:4]
    with pytest.raises(ValueError):
        _publish_policy_window_output(
            outputs, executed, num_contexts=2, num_generations=3, output_verify_lens=storage
        )


def test_publisher_consumer_mixed_skipped_sparse_padding_and_stale_plan(cpu_sampler):
    from tensorrt_llm._torch.speculative.dspark import _publish_policy_window_output

    sampler, _ = cpu_sampler
    scheduled, outputs, context, first, second = _sampling_step()
    first.py_verify_len, second.py_verify_len = 0, 1
    storage = torch.empty(8, dtype=torch.int32)
    _publish_policy_window_output(
        outputs,
        torch.tensor([3, 4, 1], dtype=torch.int32),
        num_contexts=2,
        num_generations=3,
        output_verify_lens=storage,
    )
    assert outputs["verify_lens"].tolist() == [1, 1, 3, 4, 1]
    state = sampler.sample_async(scheduled, outputs, [])
    assert [r.py_seq_slot for r in state.requests] == [2, 3, 0]
    assert state.host.verify_lens.tolist() == [1, 3, 4]
    assert state.draft_lens == [0, 3, 3]
    first.py_verify_len, second.py_verify_len = 5, 0
    storage.fill_(99)
    sampler.update_requests(state)
    assert [r.py_rewind_len for r in state.requests] == [0, 1, 2]
    assert [r.py_num_draft_tokens_verified for r in state.requests] == [0, 2, 3]
    assert [r.py_num_accepted_draft_tokens for r in state.requests] == [0, 1, 1]
    assert [r.tokens for r in state.requests] == [[4], [8, 9], [12, 13]]


def test_publisher_two_pending_steps_reuse_storage_not_host_ownership(cpu_sampler):
    from tensorrt_llm._torch.speculative.dspark import _publish_policy_window_output

    sampler, _ = cpu_sampler
    storage = torch.empty(8, dtype=torch.int32)
    states = []
    for generation_windows in ([3, 4, 1], [4, 2, 1]):
        scheduled, outputs, *_ = _sampling_step()
        _publish_policy_window_output(
            outputs,
            torch.tensor(generation_windows, dtype=torch.int32),
            num_contexts=2,
            num_generations=3,
            output_verify_lens=storage,
        )
        states.append(sampler.sample_async(scheduled, outputs, []))
    storage.fill_(99)
    assert states[0].host.verify_lens.tolist() == [1, 3, 4]
    assert states[1].host.verify_lens.tolist() == [1, 4, 2]
    assert states[0].host.verify_lens.data_ptr() != states[1].host.verify_lens.data_ptr()
    for state in states:
        sampler.update_requests(state)
    assert [r.py_rewind_len for r in states[0].requests] == [0, 1, 2]
    assert [r.py_rewind_len for r in states[1].requests] == [0, 2, 0]


@pytest.mark.parametrize(
    "stale_marker", ["host_policy_windows_snapshot", "verify_lens_in_output_order"]
)
def test_publisher_native_fallback_clears_stale_ragged_authority(cpu_sampler, stale_marker):
    from tensorrt_llm._torch.speculative.dspark import _publish_policy_window_output

    sampler, _ = cpu_sampler
    scheduled, outputs, _, first, _ = _sampling_step()
    outputs[stale_marker] = True
    outputs["verify_lens"] = torch.full((5,), 99, dtype=torch.int32)
    _publish_policy_window_output(
        outputs, None, num_contexts=2, num_generations=3, output_verify_lens=None
    )
    assert outputs["native_uniform_verify"] is True
    assert "verify_lens" not in outputs and "verify_lens_in_output_order" not in outputs
    assert "host_policy_windows_snapshot" not in outputs
    state = sampler.sample_async(scheduled, outputs, [])
    assert state.host.verify_lens is None
    sampler.update_requests(state)
    assert first.py_rewind_len == 2
    assert first.py_num_draft_tokens_verified == 3


@pytest.mark.parametrize("return_confidence", [False, True])
@pytest.mark.parametrize("executed_window", [None, 2])
def test_dspark_forward_publishes_execution_not_confidence_flag(return_confidence, executed_window):
    from tensorrt_llm._torch.speculative.dspark import DSv4DSparkWorker

    worker = DSv4DSparkWorker.__new__(DSv4DSparkWorker)
    worker.spec_config = types.SimpleNamespace(max_draft_len=1)
    worker.return_confidence = return_confidence
    worker.guided_decoder = None
    worker._output_verify_lens = torch.empty(4, dtype=torch.int32)
    worker._lazy_init = lambda *_args: None
    worker._execute_guided_decoder_if_present = lambda *_args: None
    worker.sample_and_accept_draft_tokens = lambda *_args: (
        torch.tensor([[11, 12]]),
        torch.tensor([2]),
    )
    worker._draft_gen_block_batched = lambda *_args, **_kwargs: torch.zeros((1, 1, 8))
    worker.sample_draft_tokens = lambda *_args, **_kwargs: torch.tensor([[13]])
    worker.write_context_onehot_draft_probs = lambda *_args: None
    worker._prepare_next_new_tokens = lambda *_args: torch.tensor([[12, 13]])
    metadata = types.SimpleNamespace(
        batch_indices_cuda=torch.tensor([0]),
        draft_probs_last_dim=8,
        verify_lens=None
        if executed_window is None
        else torch.tensor([executed_window], dtype=torch.int32),
    )

    outputs = worker._forward_impl(
        input_ids=torch.tensor([11, 12]),
        position_ids=torch.tensor([[0, 1]]),
        hidden_states=None,
        logits=torch.zeros((2, 8)),
        attn_metadata=types.SimpleNamespace(num_seqs=1, num_contexts=0),
        spec_metadata=metadata,
        draft_model=types.SimpleNamespace(block_size=1),
    )

    if executed_window is None:
        assert outputs["native_uniform_verify"] is True
        assert "verify_lens" not in outputs
    else:
        assert outputs["verify_lens"].tolist() == [executed_window]
        assert outputs["verify_lens_in_output_order"] is True
        assert "native_uniform_verify" not in outputs


@pytest.mark.parametrize("deferred", [False, True])
def test_publisher_exact_copy_owns_two_pending_steps(cpu_sampler, monkeypatch, deferred):
    from concurrent.futures import Future

    from tensorrt_llm._torch.pyexecutor.sampler.sampler_features import AsyncWorkerMixin
    from tensorrt_llm._torch.speculative.dspark import _publish_policy_window_output

    class Event:
        def record(self):
            self.recorded = True

        def synchronize(self):
            assert self.recorded

    class DeferredCopies:
        def __init__(self):
            self.pending = []

        def submit(self, function, *args):
            future = Future()
            self.pending.append((future, function, args))
            return future

        def finish(self):
            for future, function, args in self.pending:
                future.set_result(function(*args))

    monkeypatch.setattr(torch.cuda, "Event", Event)
    original_empty_like = torch.empty_like
    monkeypatch.setattr(
        torch,
        "empty_like",
        lambda *args, **kwargs: original_empty_like(*args, **{**kwargs, "pin_memory": False}),
    )
    sampler, _ = cpu_sampler
    copier = AsyncWorkerMixin()
    copies = DeferredCopies()
    copier._async_worker = copies if deferred else None
    copier._async_worker_futures = []
    sampler._copy_to_host = copier._copy_to_host
    sampler._record_sampler_event = copier._record_sampler_event
    storage = torch.empty(8, dtype=torch.int32)
    states = []
    for windows in ([3, 4, 1], [4, 2, 1]):
        scheduled, outputs, *_ = _sampling_step()
        _publish_policy_window_output(
            outputs,
            torch.tensor(windows, dtype=torch.int32),
            num_contexts=2,
            num_generations=3,
            output_verify_lens=storage,
        )
        states.append(sampler.sample_async(scheduled, outputs, []))
    storage.fill_(99)
    copies.finish()
    for state in states:
        sampler.update_requests(state)
    assert [r.py_rewind_len for r in states[0].requests] == [0, 1, 2]
    assert [r.py_rewind_len for r in states[1].requests] == [0, 2, 0]
    assert states[0].host.verify_lens.data_ptr() != states[1].host.verify_lens.data_ptr()
    assert copier._async_worker_futures == []
    if deferred:
        assert len(states[0].sampler_event.worker_futures) == 4
        assert len(states[1].sampler_event.worker_futures) == 4
        assert states[0].sampler_event.worker_futures is not states[1].sampler_event.worker_futures


@pytest.mark.parametrize("native", [False, True])
def test_marked_execution_never_reads_planning_property(cpu_sampler, native):
    class RequestProxy:
        def __init__(self, request):
            self.request = request

        def __getattr__(self, name):
            if name == "py_verify_len":
                raise AssertionError("completed execution must not read pre-forward planning")
            return getattr(self.request, name)

    sampler, _ = cpu_sampler
    scheduled, outputs, *_ = _sampling_step()
    scheduled.context_requests_last_chunk = [
        RequestProxy(r) for r in scheduled.context_requests_last_chunk
    ]
    scheduled.generation_requests = [RequestProxy(r) for r in scheduled.generation_requests]
    if native:
        outputs["native_uniform_verify"] = True
    else:
        outputs["verify_lens"] = torch.tensor([1, 1, 3, 4, 1], dtype=torch.int32)
        outputs["verify_lens_in_output_order"] = True
    sampler.sample_async(scheduled, outputs, [])


@pytest.mark.parametrize("explicit_native", [False, True])
def test_native_uniform_without_optional_ragged_attribute(cpu_sampler, explicit_native):
    sampler, _ = cpu_sampler
    scheduled, outputs, *_ = _sampling_step()
    for request in scheduled.context_requests_last_chunk + scheduled.generation_requests:
        del request.py_verify_len
    if explicit_native:
        outputs["native_uniform_verify"] = True

    state = sampler.sample_async(scheduled, outputs, [])
    sampler.update_requests(state)

    assert state.host.verify_lens is None
    assert [request.tokens for request in state.requests] == [[4], [8, 9], [12, 13]]
    assert [request.py_num_draft_tokens_verified for request in state.requests] == [0, 3, 3]
    assert [request.py_rewind_len for request in state.requests] == [3, 2, 2]


@pytest.mark.parametrize("ragged_generation", [0, 1])
def test_legacy_ragged_rejected_with_missing_optional_attributes(cpu_sampler, ragged_generation):
    sampler, _ = cpu_sampler
    scheduled, outputs, *_ = _sampling_step()
    for request in scheduled.context_requests_last_chunk + scheduled.generation_requests:
        del request.py_verify_len
    scheduled.generation_requests[ragged_generation].py_verify_len = 2

    with pytest.raises(ValueError, match="ragged producers"):
        sampler.sample_async(scheduled, outputs, [])
