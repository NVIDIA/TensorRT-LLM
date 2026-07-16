# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler.ops.penalties import (
    apply_batched_occurrence_penalties_triton,
    update_occurrence_workspace,
)
from tensorrt_llm._torch.pyexecutor.sampler.sampler import TorchSampler, _OccurrencePenaltyHandler


def _col(values):
    return torch.tensor(values, dtype=torch.float32, device="cuda").view(-1, 1)


def _dense_penalty_reference(logits, counts, presence, rep, pre, freq, temp):
    """Dense post-temperature reference for ``apply_occurrence_penalties_triton``.

    Follows the TorchSampler order: repetition where the token is present anywhere
    (``counts > 0`` or the prefix bitmap), then presence + frequency where counted
    (``counts > 0``), followed by temperature division in the sampling strategy.
    ``rep/pre/freq/temp`` are per-row ``[A, 1]`` tensors.
    """
    penalized = logits.float()
    present = counts > 0
    if presence is not None:
        present = present | (presence > 0)
    penalized = torch.where(
        present,
        torch.where(penalized < 0, penalized * rep, penalized / rep),
        penalized,
    )
    counts_f = counts.to(torch.float32)
    sub = torch.where(counts > 0, pre + freq * counts_f, penalized.new_zeros(()))
    return (penalized - sub) / temp


@pytest.mark.parametrize(
    "name,rep,pre,freq,temp,use_prefix",
    [
        # repetition only, exercises the sign branch (>1, <1) at temp=1
        ("repetition", [1.3, 2.0, 0.7], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], False),
        # presence only
        ("presence", [1.0, 1.0], [0.5, 1.5], [0.0, 0.0], [1.0, 1.0], False),
        # frequency only (counts > 1 -> proportional)
        ("frequency", [1.0, 1.0], [0.0, 0.0], [0.4, 0.9], [1.0, 1.0], False),
        # combined with temperature != 1 (exercises penalty-before-temperature order)
        (
            "combined_temp",
            [1.2, 0.8, 1.5],
            [0.3, 0.0, 0.7],
            [0.2, 0.5, 0.0],
            [0.7, 1.3, 2.0],
            False,
        ),
        # ignored-prompt-prefix bitmap affects repetition only, not presence/frequency
        ("prefix", [1.4, 1.1, 0.9], [0.4, 0.6, 0.2], [0.3, 0.1, 0.5], [1.0, 0.8, 1.6], True),
    ],
)
def test_regular_triton_matches_dense_logits_reference(name, rep, pre, freq, temp, use_prefix):
    # vocab=5000 is not a multiple of BLOCK_SIZE (1024), exercising the tail mask.
    A, V = len(rep), 5000
    gen = torch.Generator(device="cuda").manual_seed(sum(name.encode()))
    logits = torch.randn(A, V, device="cuda", generator=gen) * 5.0
    counts = torch.randint(0, 4, (A, V), dtype=torch.int32, device="cuda", generator=gen)
    presence = (
        torch.randint(0, 2, (A, V), dtype=torch.int32, device="cuda", generator=gen)
        if use_prefix
        else None
    )
    rep_t, pre_t, freq_t, temp_t = _col(rep), _col(pre), _col(freq), _col(temp)
    slots = torch.arange(A, dtype=torch.int64, device="cuda")

    got = logits.clone()
    apply_batched_occurrence_penalties_triton(
        got,
        counts,
        presence,
        torch.ones(A, dtype=torch.bool, device="cuda"),
        torch.zeros(A, dtype=torch.bool, device="cuda"),
        torch.zeros(1, A, 1, dtype=torch.int32, device="cuda"),
        slots,
        torch.arange(A, dtype=torch.int32, device="cuda"),
        torch.ones(A, dtype=torch.int32, device="cuda"),
        None,
        rep_t.squeeze(1),
        pre_t.squeeze(1),
        freq_t.squeeze(1),
        max_num_steps=1,
    )
    ref = _dense_penalty_reference(logits, counts, presence, rep_t, pre_t, freq_t, temp_t)
    # the kernel is pre-temperature-division; divide by temp to compare to the final value.
    torch.testing.assert_close(got / temp_t, ref, rtol=1e-4, atol=1e-4)


def test_triton_indirect_indexing_bf16():
    # Permuted request offsets and sequence slots penalize a subset of logits rows, with
    # repeated slot mappings. Other rows must stay untouched. bfloat16 also covers the
    # fp32-compute -> bf16-store cast path.
    gen = torch.Generator(device="cuda").manual_seed(3)
    num_slots, num_rows, vocab = 5, 10, 3000
    logits = (torch.randn(num_rows, vocab, device="cuda", generator=gen) * 3).to(torch.bfloat16)
    orig = logits.clone()
    counts = torch.randint(
        0, 4, (num_slots, vocab), dtype=torch.int32, device="cuda", generator=gen
    )
    rep = torch.empty(num_slots, device="cuda").uniform_(0.7, 1.6, generator=gen)
    pre = torch.empty(num_slots, device="cuda").uniform_(0.0, 0.6, generator=gen)
    freq = torch.empty(num_slots, device="cuda").uniform_(0.0, 0.4, generator=gen)
    temp = torch.empty(num_slots, device="cuda").uniform_(0.6, 1.4, generator=gen)
    # Explicitly exercise permuted rows and repeated slot mappings.
    active_rows = torch.tensor([8, 1, 6, 3, 9, 0, 5], dtype=torch.int64, device="cuda")
    row_slots = torch.tensor([4, 1, 4, 0, 2, 1, 3], dtype=torch.int64, device="cuda")

    active = torch.ones(num_slots, dtype=torch.bool, device="cuda")
    active[1] = False
    apply_batched_occurrence_penalties_triton(
        logits,
        counts,
        None,
        active,
        torch.zeros(num_slots, dtype=torch.bool, device="cuda"),
        torch.zeros(1, num_slots, 1, dtype=torch.int32, device="cuda"),
        row_slots,
        active_rows.to(torch.int32),
        torch.ones(active_rows.numel(), dtype=torch.int32, device="cuda"),
        None,
        rep,
        pre,
        freq,
        max_num_steps=1,
    )

    active_row_mask = active[row_slots]
    active_slots = row_slots[active_row_mask]
    ref = _dense_penalty_reference(
        orig[active_rows[active_row_mask]],
        counts[active_slots],
        None,
        rep[active_slots].view(-1, 1),
        pre[active_slots].view(-1, 1),
        freq[active_slots].view(-1, 1),
        temp[active_slots].view(-1, 1),
    )
    expected = orig[active_rows].clone()
    active_temperature = temp[active_slots].view(-1, 1)
    # Recover the pre-temperature kernel output, then match its fp32-compute -> bf16-store
    # boundary. This keeps the tolerance about Triton math, not bf16 rounding.
    expected[active_row_mask] = (ref * active_temperature).to(torch.bfloat16)
    torch.testing.assert_close(logits[active_rows], expected, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(
        logits[active_rows[~active_row_mask]],
        orig[active_rows[~active_row_mask]],
        rtol=0,
        atol=0,
    )
    untouched = torch.ones(num_rows, dtype=torch.bool, device="cuda")
    untouched[active_rows] = False
    torch.testing.assert_close(logits[untouched], orig[untouched], rtol=0, atol=0)


def test_update_occurrence_workspace():
    counts = torch.zeros(3, 16, dtype=torch.int32, device="cuda")
    presence = torch.zeros(3, 16, dtype=torch.int32, device="cuda")
    long = lambda xs: torch.tensor(xs, dtype=torch.int64, device="cuda")  # noqa: E731

    update_occurrence_workspace(
        counts, presence, long([0, 0, 1]), long([2, 2, 5]), long([0]), long([7])
    )
    assert counts[0, 2] == 2  # repeated (slot, token) accumulates
    assert counts[1, 5] == 1
    assert presence[0, 7] == 1
    assert counts[0, 7] == 0  # prefix token is not counted

    # presence_prefix=None and empty prefix updates must not raise.
    update_occurrence_workspace(counts, None, long([0]), long([2]), long([]), long([]))
    assert counts[0, 2] == 3


def _make_handler_request(
    *,
    slot: int,
    tokens: list[int],
    prompt_ignore_length: int = 0,
    beam_width: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        sampling_config=SimpleNamespace(
            repetition_penalty=[1.2],
            presence_penalty=[0.4],
            frequency_penalty=[0.3],
            temperature=[1.0],
            prompt_ignore_length=[prompt_ignore_length],
            beam_width=beam_width,
            beam_width_array=None,
        ),
        py_orig_prompt_len=len(tokens),
        py_seq_slot=slot,
        py_return_log_probs=False,
        get_tokens=lambda _beam_idx: tokens,
    )


def _apply_handler(
    handler: _OccurrencePenaltyHandler,
    request: SimpleNamespace,
    logits: torch.Tensor,
    num_steps: int,
    new_tokens: torch.Tensor,
) -> None:
    handler.apply(
        logits,
        [request],
        [num_steps],
        new_tokens=new_tokens,
        seq_slots=torch.tensor([request.py_seq_slot], dtype=torch.int64, device="cuda"),
        request_offsets=torch.zeros(1, dtype=torch.int32),
        request_num_steps=torch.tensor([num_steps], dtype=torch.int32),
    )


def test_regular_handler_syncs_speculative_history_in_logits():
    vocab = 16
    slot = 2
    handler = _OccurrencePenaltyHandler(
        max_num_sequences=3,
        device="cuda",
    )
    history = [3]
    request = _make_handler_request(slot=slot, tokens=history)
    handler.prepare_for_new_request(request, slot=slot)
    new_tokens = torch.zeros(3, 3, 1, dtype=torch.int32, device="cuda")

    _apply_handler(handler, request, torch.zeros(1, vocab, device="cuda"), 1, new_tokens)
    history.append(5)
    new_tokens[0, slot, 0] = 5
    _apply_handler(handler, request, torch.zeros(1, vocab, device="cuda"), 1, new_tokens)

    history.append(6)
    _apply_handler(handler, request, torch.zeros(3, vocab, device="cuda"), 3, new_tokens)

    history.extend([7, 8, 7])
    logits = torch.linspace(-4.0, 4.0, steps=3 * vocab, device="cuda").view(3, vocab)
    original = logits.clone()
    _apply_handler(handler, request, logits, 3, new_tokens)

    expected_counts = torch.bincount(torch.tensor(history, device="cuda"), minlength=vocab).to(
        torch.int32
    )[None]
    expected = _dense_penalty_reference(
        original,
        expected_counts.expand(3, -1),
        None,
        torch.full((3, 1), 1.2, device="cuda"),
        torch.full((3, 1), 0.4, device="cuda"),
        torch.full((3, 1), 0.3, device="cuda"),
        torch.ones(3, 1, device="cuda"),
    )
    torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)


def test_regular_handler_slot_reuse_does_not_leak_penalties():
    vocab = 16
    handler = _OccurrencePenaltyHandler(
        max_num_sequences=1,
        device="cuda",
    )
    new_tokens = torch.zeros(1, 1, 1, dtype=torch.int32, device="cuda")

    first = _make_handler_request(slot=0, tokens=[3, 3], prompt_ignore_length=1)
    handler.prepare_for_new_request(first, slot=0)
    _apply_handler(handler, first, torch.zeros(1, vocab, device="cuda"), 1, new_tokens)

    second = _make_handler_request(slot=0, tokens=[5])
    handler.prepare_for_new_request(second, slot=0)
    logits = torch.linspace(-2.0, 2.0, steps=vocab, device="cuda").view(1, vocab)
    original = logits.clone()
    _apply_handler(handler, second, logits, 1, new_tokens)

    expected_counts = torch.zeros(1, vocab, dtype=torch.int32, device="cuda")
    expected_counts[0, 5] = 1
    expected = _dense_penalty_reference(
        original,
        expected_counts,
        None,
        torch.full((1, 1), 1.2, device="cuda"),
        torch.full((1, 1), 0.4, device="cuda"),
        torch.full((1, 1), 0.3, device="cuda"),
        torch.ones(1, 1, device="cuda"),
    )
    torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)


def test_beam_penalties_warn_and_leave_logits_unchanged(monkeypatch):
    request = _make_handler_request(slot=0, tokens=[3, 3], beam_width=2)
    warnings = []
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.sampler.sampler.logger.warning",
        warnings.append,
    )

    sampler = object.__new__(TorchSampler)
    sampler.max_beam_width = 2
    sampler.validate_request(request)
    assert warnings == [
        "TorchSampler does not support repetition, presence, or frequency "
        "penalties with beam search; these penalties will be ignored."
    ]

    handler = _OccurrencePenaltyHandler(max_num_sequences=1, device="cuda")
    handler.prepare_for_new_request(request, slot=0)
    logits = torch.randn(2, 16, device="cuda")
    expected = logits.clone()
    handler.apply(
        logits,
        [request],
        [1],
        new_tokens=torch.zeros(1, 1, 2, dtype=torch.int32, device="cuda"),
        seq_slots=torch.zeros(1, dtype=torch.int64, device="cuda"),
        request_offsets=torch.zeros(1, dtype=torch.int32),
        request_num_steps=torch.ones(1, dtype=torch.int32),
    )
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
