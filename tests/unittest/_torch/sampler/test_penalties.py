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

from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import (
    apply_batched_occurrence_penalties,
    update_occurrence_workspace,
)
from tensorrt_llm._torch.pyexecutor.sampler.sampler import _OccurrencePenaltyHandler


@pytest.fixture(autouse=True)
def _dynamo_recompile_headroom():
    """Recompile headroom for the fullgraph=True penalty op.

    These cases sweep tensor shapes/dtypes, so the op legitimately builds one graph per shape
    -- more than the default recompile_limit (8). A served model has fixed shapes; raising the
    limit only here avoids tripping fullgraph's hard-fail without touching production.
    """
    import torch._dynamo

    with torch._dynamo.config.patch(recompile_limit=128):
        yield


def _col(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device="cuda").view(-1, 1)


def _dense_penalty_reference(
    logits: torch.Tensor,
    counts: torch.Tensor,
    presence: torch.Tensor | None,
    rep: torch.Tensor,
    pre: torch.Tensor,
    freq: torch.Tensor,
    temp: torch.Tensor,
) -> torch.Tensor:
    """Dense post-temperature reference for ``apply_batched_occurrence_penalties``.

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


def _dense_presence_prefix(counts: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
    prefix = torch.zeros(
        presence.size(0),
        presence.size(1),
        dtype=torch.bool,
        device=presence.device,
    )
    prefix_slots, prefix_tokens = torch.nonzero(presence, as_tuple=True)
    empty = torch.empty(0, dtype=torch.int64, device=presence.device)
    update_occurrence_workspace(
        counts,
        prefix,
        empty,
        empty,
        prefix_slots,
        prefix_tokens,
    )
    return prefix


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
@pytest.mark.parametrize("num_steps", [1, 3], ids=["regular", "speculative"])
def test_penalties_match_dense_logits_reference(
    name: str,
    rep: list[float],
    pre: list[float],
    freq: list[float],
    temp: list[float],
    use_prefix: bool,
    num_steps: int,
) -> None:
    # vocab=5000 is not a multiple of BLOCK_SIZE (1024), exercising the tail mask.
    A, V = len(rep), 5000
    gen = torch.Generator(device="cuda").manual_seed(sum(name.encode()) + num_steps)
    logits = torch.randn(A * num_steps, V, device="cuda", generator=gen) * 5.0
    counts = torch.randint(0, 4, (A, V), dtype=torch.int32, device="cuda", generator=gen)
    presence = (
        torch.randint(0, 2, (A, V), dtype=torch.int32, device="cuda", generator=gen)
        if use_prefix
        else None
    )
    presence_prefix = _dense_presence_prefix(counts, presence) if presence is not None else None
    rep_t, pre_t, freq_t, temp_t = _col(rep), _col(pre), _col(freq), _col(temp)
    slots = torch.arange(A, dtype=torch.int64, device="cuda")
    row_slots = slots.repeat_interleave(num_steps)

    got = logits.clone()
    apply_batched_occurrence_penalties(
        got,
        counts,
        presence_prefix,
        torch.ones(A, dtype=torch.bool, device="cuda"),
        torch.zeros(A, dtype=torch.bool, device="cuda"),
        torch.zeros(1, A, 1, dtype=torch.int32, device="cuda"),
        slots,
        torch.arange(0, A * num_steps, num_steps, dtype=torch.int32, device="cuda"),
        torch.full((A,), num_steps, dtype=torch.int32, device="cuda"),
        rep_t.squeeze(1),
        pre_t.squeeze(1),
        freq_t.squeeze(1),
    )
    row_presence = presence[row_slots] if presence is not None else None
    ref = _dense_penalty_reference(
        logits,
        counts[row_slots],
        row_presence,
        rep_t[row_slots],
        pre_t[row_slots],
        freq_t[row_slots],
        temp_t[row_slots],
    )
    # the kernel is pre-temperature-division; divide by temp to compare to the final value.
    torch.testing.assert_close(got / temp_t[row_slots], ref, rtol=1e-4, atol=1e-4)


def test_penalties_indirect_indexing_bf16() -> None:
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
    apply_batched_occurrence_penalties(
        logits,
        counts,
        None,
        active,
        torch.zeros(num_slots, dtype=torch.bool, device="cuda"),
        torch.zeros(1, num_slots, 1, dtype=torch.int32, device="cuda"),
        row_slots,
        active_rows.to(torch.int32),
        torch.ones(active_rows.numel(), dtype=torch.int32, device="cuda"),
        rep,
        pre,
        freq,
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


def test_prefix_marking_matches_dense_logits_reference() -> None:
    vocab = 70
    counts = torch.zeros(1, vocab, dtype=torch.int32, device="cuda")
    presence_prefix = torch.zeros(1, vocab, dtype=torch.bool, device="cuda")

    counted_tokens = torch.tensor([31, 31, 45], dtype=torch.int64, device="cuda")
    prefix_tokens = torch.tensor([0, 31, 31, 32, 63, 69], dtype=torch.int64, device="cuda")
    counted_slots = torch.zeros_like(counted_tokens)
    prefix_slots = torch.zeros_like(prefix_tokens)
    update_occurrence_workspace(
        counts,
        presence_prefix,
        counted_slots,
        counted_tokens,
        prefix_slots,
        prefix_tokens,
    )

    logits = torch.linspace(-7.0, 7.0, vocab, device="cuda").view(1, -1)
    original = logits.clone()
    apply_batched_occurrence_penalties(
        logits,
        counts,
        presence_prefix,
        torch.ones(1, dtype=torch.bool, device="cuda"),
        torch.zeros(1, dtype=torch.bool, device="cuda"),
        torch.zeros(1, 1, 1, dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int64, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        torch.ones(1, dtype=torch.int32, device="cuda"),
        torch.tensor([1.2], device="cuda"),
        torch.tensor([0.4], device="cuda"),
        torch.tensor([0.3], device="cuda"),
    )

    dense_prefix = torch.zeros_like(counts)
    dense_prefix[0, torch.unique(prefix_tokens)] = 1
    expected = _dense_penalty_reference(
        original,
        counts,
        dense_prefix,
        torch.tensor([[1.2]], device="cuda"),
        torch.tensor([[0.4]], device="cuda"),
        torch.tensor([[0.3]], device="cuda"),
        torch.ones(1, 1, device="cuda"),
    )
    assert presence_prefix.shape == (1, vocab)
    torch.testing.assert_close(logits, expected, rtol=1e-4, atol=1e-4)


def test_penalty_op_does_not_latch_pending_token() -> None:
    """The penalty op must not write ``has_previous_token``.

    The op reads the flag to decide whether to fold the pending ``new_tokens`` token
    into ``counts``; it must never write it (the host re-arms the flag after the op).
    Here the flag is False with a stale token far up the vocab: nothing may be folded,
    the flag must stay False, and the logits must be untouched.
    """
    vocab = 3000
    stale_token = 2500  # a stale pending token far up the vocab
    has_previous_token = torch.zeros(1, dtype=torch.bool, device="cuda")
    new_tokens = torch.zeros(1, 1, 1, dtype=torch.int32, device="cuda")
    new_tokens[0, 0, 0] = stale_token
    counts = torch.zeros(1, vocab, dtype=torch.int32, device="cuda")
    logits = torch.linspace(-4.0, 4.0, steps=vocab, device="cuda").view(1, vocab)
    original = logits.clone()

    apply_batched_occurrence_penalties(
        logits,
        counts,
        None,
        torch.ones(1, dtype=torch.bool, device="cuda"),
        has_previous_token,
        new_tokens,
        torch.zeros(1, dtype=torch.int64, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        torch.ones(1, dtype=torch.int32, device="cuda"),
        torch.tensor([1.5], device="cuda"),
        torch.tensor([0.5], device="cuda"),
        torch.tensor([0.4], device="cuda"),
    )

    # Deterministic: the penalty op must leave the latch untouched (host re-arms it).
    assert not bool(has_previous_token.item())
    # With has_previous_token False and counts all zero, no penalty may be applied; the
    # stale token in particular must not be folded (would perturb logits[2500]).
    torch.testing.assert_close(logits, original, rtol=0, atol=0)


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
        new_tokens=new_tokens,
        seq_slots=torch.tensor([request.py_seq_slot], dtype=torch.int64, device="cuda"),
        request_offsets=torch.zeros(1, dtype=torch.int32),
        request_num_steps=torch.tensor([num_steps], dtype=torch.int32),
    )


def test_handler_tracks_overlap_and_commits_speculative_tail() -> None:
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

    # The first apply initializes the prompt and marks the first sampled token as
    # pending. The request's host history need not be updated before the next apply.
    _apply_handler(handler, request, torch.zeros(1, vocab, device="cuda"), 1, new_tokens)
    new_tokens[0, slot, 0] = 5
    overlap_logits = torch.linspace(-2.0, 2.0, vocab, device="cuda").view(1, vocab)
    overlap_original = overlap_logits.clone()
    _apply_handler(handler, request, overlap_logits, 1, new_tokens)
    overlap_counts = torch.bincount(torch.tensor([3, 5], device="cuda"), minlength=vocab).to(
        torch.int32
    )[None]
    overlap_expected = _dense_penalty_reference(
        overlap_original,
        overlap_counts,
        None,
        torch.full((1, 1), 1.2, device="cuda"),
        torch.full((1, 1), 0.4, device="cuda"),
        torch.full((1, 1), 0.3, device="cuda"),
        torch.ones(1, 1, device="cuda"),
    )
    torch.testing.assert_close(overlap_logits, overlap_expected, rtol=1e-4, atol=1e-4)

    # The next invocation is speculative. All rows use the same confirmed history;
    # the current draft window remains tentative until acceptance is resolved.
    history.extend([5, 6])
    new_tokens[0, slot, 0] = 6
    spec_logits = torch.linspace(-3.0, 3.0, steps=3 * vocab, device="cuda").view(3, vocab)
    spec_original = spec_logits.clone()
    _apply_handler(handler, request, spec_logits, 3, new_tokens)
    spec_counts = torch.bincount(torch.tensor(history, device="cuda"), minlength=vocab).to(
        torch.int32
    )[None]
    spec_expected = _dense_penalty_reference(
        spec_original,
        spec_counts.expand(3, -1),
        None,
        torch.full((3, 1), 1.2, device="cuda"),
        torch.full((3, 1), 0.4, device="cuda"),
        torch.full((3, 1), 0.3, device="cuda"),
        torch.ones(3, 1, device="cuda"),
    )
    torch.testing.assert_close(spec_logits, spec_expected, rtol=1e-4, atol=1e-4)

    # Sampler-side acceptance commits the complete finalized sequence. Deliberately
    # leave a different raw target token in the device buffer, as rejection sampling
    # can do; clearing the pending flag must prevent it from entering the workspace.
    history.extend([7, 8, 7])
    new_tokens[0, slot, 0] = 4
    handler.update_token_counts([(slot, [7, 8, 7])])
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


def test_regular_handler_slot_reuse_does_not_leak_penalties() -> None:
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
