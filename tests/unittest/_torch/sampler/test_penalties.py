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

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import Fusions
from tensorrt_llm._torch.pyexecutor.sampler.penalties import PenaltyHandler

apply_batched_occurrence_penalties = Fusions.apply_batched_occurrence_penalties
update_occurrence_workspace = Fusions.update_occurrence_workspace


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
    (``counts > 0`` or the prefix mask), then presence + frequency where counted
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
        # ignored-prompt-prefix mask affects repetition only, not presence/frequency
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
    # vocab=5000 is deliberately not a round power of two.
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
    # Recover the pre-temperature op output, then match its fp32-compute -> bf16-store
    # boundary. This keeps the tolerance about the op's fp32 math, not bf16 rounding.
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
        py_is_draft=False,
    )


def _admit(handler: PenaltyHandler, request: SimpleNamespace, slot: int) -> None:
    """Admit one request, mirroring TorchSampler.setup_sampler_step.

    ``prepare_for_new_request`` only accumulates on the host; the device buffers are
    written by the batched ``update_for_new_requests`` flush at the end of the step.
    """
    handler.prepare_for_new_request(request, slot=slot)
    handler.update_for_new_requests(
        new_seq_slots_cuda_long=torch.tensor([slot], dtype=torch.int64, device="cuda")
    )


def _apply_handler(
    handler: PenaltyHandler,
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
    handler = PenaltyHandler(
        max_num_sequences=3,
        max_beam_width=1,
        device="cuda",
    )
    history = [3]
    request = _make_handler_request(slot=slot, tokens=history)
    _admit(handler, request, slot)
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
    handler = PenaltyHandler(
        max_num_sequences=1,
        max_beam_width=1,
        device="cuda",
    )
    new_tokens = torch.zeros(1, 1, 1, dtype=torch.int32, device="cuda")

    first = _make_handler_request(slot=0, tokens=[3, 3], prompt_ignore_length=1)
    _admit(handler, first, 0)
    _apply_handler(handler, first, torch.zeros(1, vocab, device="cuda"), 1, new_tokens)

    second = _make_handler_request(slot=0, tokens=[5])
    _admit(handler, second, 0)
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


BEAM_PAD = -1
BEAM_VOCAB = 11
BEAM_SLOTS = 5
BEAM_WIDTH = 4


@dataclass(frozen=True)
class _BeamStep:
    """One decoding step of a scenario; every field carries one row per slot.

    Rows are in ascending slot order, matching the ``seq_slots`` handed to the op.
    """

    predecessor_beams: list[list[int]]
    """Parent beam of each beam, as the previous step's beam search would have written."""
    sampled_tokens: list[list[int]]
    """Token each beam sampled last step; ``BEAM_PAD`` where the parent had finished."""
    armed: list[bool]
    """Whether the slot sampled last step -- the ``has_previous_token`` latch."""
    num_beams: list[int]
    """Row-layout beam width; beams at or past it are re-parented but never folded.

    The sampler passes the static admission width here, so in production this only
    drops the beams of a narrower request sharing the engine; a beam left behind by a
    growing ``beam_width_array`` is dropped by its ``BEAM_PAD`` instead. The op's
    contract is the same either way, so the scenarios exercise both gates."""


@dataclass(frozen=True)
class _BeamCountScenario:
    name: str
    prompts: dict[int, list[int]]
    """Prompt tokens per slot, seeded onto every beam of that slot."""
    steps: list[_BeamStep]


_BEAM_COUNT_SCENARIOS = [
    # Every beam continues itself, so the histories just diverge.
    _BeamCountScenario(
        name="divergence",
        prompts={1: [2, 2, 5], 3: [7]},
        steps=[
            _BeamStep(
                predecessor_beams=[[0, 1, 2, 3], [0, 1, 2, 3]],
                sampled_tokens=[[1, 2, 3, 4], [5, 6, 7, 8]],
                armed=[True, True],
                num_beams=[4, 4],
            ),
            _BeamStep(
                predecessor_beams=[[0, 1, 2, 3], [0, 1, 2, 3]],
                sampled_tokens=[[2, 2, 2, 2], [7, 7, 7, 7]],
                armed=[True, True],
                num_beams=[4, 4],
            ),
        ],
    ),
    # Several beams share one parent and another parent is dropped entirely -- the
    # case a per-slot count row cannot represent.
    _BeamCountScenario(
        name="reconvergence",
        prompts={0: [3]},
        steps=[
            _BeamStep([[0, 0, 0, 0]], [[1, 2, 3, 4]], [True], [4]),
            _BeamStep([[2, 2, 0, 2]], [[5, 6, 7, 8]], [True], [4]),
            _BeamStep([[1, 1, 1, 1]], [[9, 9, 9, 9]], [True], [4]),
        ],
    ),
    # Beam width grows 1 -> BEAM_WIDTH: the context step is unarmed, then the first
    # generation step re-parents every beam onto beam 0 and folds its own token.
    _BeamCountScenario(
        name="vbws-growth",
        prompts={2: [4, 4]},
        steps=[
            _BeamStep([[0, 0, 0, 0]], [[6, BEAM_PAD, BEAM_PAD, BEAM_PAD]], [False], [1]),
            _BeamStep([[0, 0, 0, 0]], [[6, 7, 8, 9]], [True], [4]),
            _BeamStep([[0, 1, 2, 3]], [[1, 1, 1, 1]], [True], [4]),
        ],
    ),
    # Intermediate growth 2 -> 4: the previous step's map holds two valid entries and two
    # stale ones in the same slot, and the new beams inherit from beam 1, whose history
    # differs from beam 0's -- so a stale row that was not re-parented is detectable.
    _BeamCountScenario(
        name="vbws-growth-partial",
        prompts={0: [3]},
        steps=[
            _BeamStep([[0, 0, 0, 0]], [[1, 2, BEAM_PAD, BEAM_PAD]], [True], [2]),
            _BeamStep([[0, 1, 1, 1]], [[5, 6, 7, 8]], [True], [4]),
            _BeamStep([[0, 1, 2, 3]], [[9, 9, 9, 9]], [True], [4]),
        ],
    ),
    # A beam whose predecessor already finished emits BEAM_SEARCH_PAD_TOKEN.
    _BeamCountScenario(
        name="pad-token",
        prompts={4: [0]},
        steps=[
            _BeamStep([[0, 0, 0, 0]], [[1, 2, BEAM_PAD, BEAM_PAD]], [True], [4]),
            _BeamStep([[0, 1, 2, 3]], [[BEAM_PAD, 3, BEAM_PAD, 4]], [True], [4]),
        ],
    ),
    # A slot scheduled without sampling keeps a stale predecessor map; the
    # has_previous_token latch must make the whole step a no-op.
    _BeamCountScenario(
        name="unarmed-identity",
        prompts={1: [8, 8]},
        steps=[
            _BeamStep([[0, 0, 0, 0]], [[1, 2, 3, 4]], [True], [4]),
            _BeamStep([[3, 3, 3, 3]], [[5, 5, 5, 5]], [False], [4]),
            _BeamStep([[0, 1, 2, 3]], [[6, 6, 6, 6]], [True], [4]),
        ],
    ),
    # Beams at or past the request's layout width are re-parented but never folded --
    # a narrower request sharing a wider engine.
    _BeamCountScenario(
        name="narrow-request",
        prompts={0: [1]},
        steps=[
            _BeamStep([[0, 0, 0, 0]], [[2, 3, 4, 5]], [True], [4]),
            _BeamStep([[1, 3, 0, 0]], [[6, 7, 0, 0]], [True], [2]),
        ],
    ),
    # Multimodal placeholder ids land outside the vocab and must be dropped.
    _BeamCountScenario(
        name="out-of-range-token",
        prompts={3: [2]},
        steps=[_BeamStep([[0, 0, 0, 0]], [[BEAM_VOCAB, BEAM_VOCAB + 7, 1, -5]], [True], [4])],
    ),
]


def _beam_counts_reference(scenario: _BeamCountScenario) -> torch.Tensor:
    """Host replay of the per-beam occurrence counts.

    Mirrors the op's contract: on an armed slot every beam first takes over the
    history of ``predecessor_beams[beam]``, then the beams below the layout width
    append the token they sampled. Unarmed slots are left alone.
    """
    slots = sorted(scenario.prompts)
    history = {
        (slot, beam): list(prompt)
        for slot, prompt in scenario.prompts.items()
        for beam in range(BEAM_WIDTH)
    }
    for step in scenario.steps:
        advanced = dict(history)
        for index, slot in enumerate(slots):
            if not step.armed[index]:
                continue
            for beam in range(BEAM_WIDTH):
                inherited = list(history[(slot, step.predecessor_beams[index][beam])])
                token = step.sampled_tokens[index][beam]
                if beam < step.num_beams[index] and 0 <= token < BEAM_VOCAB:
                    inherited.append(token)
                advanced[(slot, beam)] = inherited
        history = advanced

    expected = torch.zeros((BEAM_SLOTS * BEAM_WIDTH, BEAM_VOCAB), dtype=torch.int32, device="cuda")
    for (slot, beam), tokens in history.items():
        for token in tokens:
            expected[slot * BEAM_WIDTH + beam, token] += 1
    return expected


@pytest.mark.parametrize(
    "scenario", _BEAM_COUNT_SCENARIOS, ids=[s.name for s in _BEAM_COUNT_SCENARIOS]
)
def test_beam_occurrence_counts_follow_each_beam_history(scenario: _BeamCountScenario) -> None:
    """``update_beam_occurrence_counts`` keeps one true history per beam.

    Each beam inherits its parent's counts and appends its own token, the torch
    counterpart of the C++ ``batchApplyPenalty`` workspace copy along ``parentIds``.
    """
    counts = torch.zeros((BEAM_SLOTS * BEAM_WIDTH, BEAM_VOCAB), dtype=torch.int32, device="cuda")
    active = torch.zeros(BEAM_SLOTS, dtype=torch.bool, device="cuda")
    has_previous_token = torch.zeros(BEAM_SLOTS, dtype=torch.bool, device="cuda")
    predecessor_beams = torch.zeros((BEAM_SLOTS, BEAM_WIDTH), dtype=torch.int32, device="cuda")
    new_tokens = torch.zeros(1, BEAM_SLOTS, BEAM_WIDTH, dtype=torch.int32, device="cuda")
    slots = sorted(scenario.prompts)
    seq_slots = torch.tensor(slots, dtype=torch.int64, device="cuda")

    # Seed every beam from the prompt, as PenaltyHandler._initialize_workspace does.
    for slot, prompt in scenario.prompts.items():
        active[slot] = True
        for beam in range(BEAM_WIDTH):
            for token in prompt:
                counts[slot * BEAM_WIDTH + beam, token] += 1

    for step in scenario.steps:
        for index, slot in enumerate(slots):
            predecessor_beams[slot] = torch.tensor(
                step.predecessor_beams[index], dtype=torch.int32, device="cuda"
            )
            new_tokens[0, slot] = torch.tensor(
                step.sampled_tokens[index], dtype=torch.int32, device="cuda"
            )
            has_previous_token[slot] = step.armed[index]
        Fusions.update_beam_occurrence_counts(
            counts,
            active,
            has_previous_token,
            torch.ones(BEAM_SLOTS, dtype=torch.bool, device="cuda"),
            new_tokens,
            predecessor_beams,
            seq_slots,
            torch.tensor(step.num_beams, dtype=torch.int32, device="cuda"),
            BEAM_WIDTH,
        )

    torch.testing.assert_close(counts, _beam_counts_reference(scenario), rtol=0, atol=0)


def test_single_beam_slot_sharing_a_beam_engine_is_routed_correctly() -> None:
    """A width-1 request on a beam engine must not be treated as a beam request.

    Beam width is per request, so once TRTLLM-14792 lifts the equal-width restriction a
    batch can mix the two. The single-beam slot must never be re-parented -- nothing
    writes its ``predecessor_beams`` row, so believing it would corrupt its history --
    while both kinds are penalized by the same packed pass, each against its own row.
    """
    max_beam_width, vocab, num_slots = 4, 32, 2
    beam_slot, plain_slot = 0, 1

    counts = torch.zeros((num_slots * max_beam_width, vocab), dtype=torch.int32, device="cuda")
    active = torch.ones(num_slots, dtype=torch.bool, device="cuda")
    armed = torch.ones(num_slots, dtype=torch.bool, device="cuda")
    is_beam = torch.tensor([True, False], dtype=torch.bool, device="cuda")
    seq_slots = torch.tensor([beam_slot, plain_slot], dtype=torch.int64, device="cuda")

    # Give the plain slot a hostile predecessor map: if it were believed, beam 0 would
    # inherit beam 3's counts.
    predecessor_beams = torch.zeros((num_slots, max_beam_width), dtype=torch.int32, device="cuda")
    predecessor_beams[plain_slot, :] = 3
    counts[plain_slot * max_beam_width + 3, 9] = 7  # poison beam 3 of the plain slot

    new_tokens = torch.zeros(1, num_slots, max_beam_width, dtype=torch.int32, device="cuda")
    new_tokens[0, beam_slot] = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
    new_tokens[0, plain_slot] = torch.tensor([5, 6, 7, 8], dtype=torch.int32, device="cuda")

    Fusions.update_beam_occurrence_counts(
        counts,
        active,
        armed,
        is_beam,
        new_tokens,
        predecessor_beams,
        seq_slots,
        torch.tensor([max_beam_width, 1], dtype=torch.int32, device="cuda"),
        max_beam_width,
    )

    plain_base = plain_slot * max_beam_width
    assert int(counts[plain_base, 9].item()) == 0, "plain slot must not inherit beam 3"
    assert int(counts[plain_base, 5].item()) == 1, "plain slot folds only its own token"
    for token in (6, 7, 8):
        assert int(counts[plain_base, token].item()) == 0, "beams 1..3 must not fold"
    # The beam slot still behaves as before: every beam folds its own token.
    for beam, token in enumerate((1, 2, 3, 4)):
        assert int(counts[beam_slot * max_beam_width + beam, token].item()) == 1

    # The packed pass penalizes both kinds; the plain slot must read its beam-0 row and
    # not, say, beam 3's poisoned counts.
    rows = max_beam_width + 1  # beam slot: 4 rows; plain slot: 1
    logits = torch.linspace(-2.0, 2.0, steps=rows * vocab, device="cuda").view(rows, vocab)
    original = logits.clone()
    rep = torch.full((num_slots,), 2.0, device="cuda")
    zero = torch.zeros(num_slots, device="cuda")
    apply_batched_occurrence_penalties(
        logits,
        counts,
        None,
        active,
        armed,
        new_tokens,
        seq_slots,
        torch.tensor([0, max_beam_width], dtype=torch.int32, device="cuda"),
        torch.tensor([1, 1], dtype=torch.int32, device="cuda"),
        rep,
        zero,
        zero,
        torch.tensor([max_beam_width, 1], dtype=torch.int32, device="cuda"),
        max_beam_width,
        False,  # the fold already happened above
    )

    def penalized(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x < 0, x * 2.0, x / 2.0)

    plain_row = max_beam_width
    # Token 5 is the plain slot's only counted token -> repetition branch.
    torch.testing.assert_close(
        logits[plain_row, 5], penalized(original[plain_row, 5]), rtol=1e-5, atol=1e-5
    )
    # Token 9 is only in the poisoned beam-3 row, which the plain slot must not read.
    torch.testing.assert_close(logits[plain_row, 9], original[plain_row, 9], rtol=0, atol=0)
    # Each beam row of the beam slot is penalized against its own token.
    for beam, token in enumerate((1, 2, 3, 4)):
        torch.testing.assert_close(
            logits[beam, token], penalized(original[beam, token]), rtol=1e-5, atol=1e-5
        )
        other = 1 + ((beam + 1) % 4)  # a token belonging to a different beam
        if other != token:
            torch.testing.assert_close(logits[beam, other], original[beam, other], rtol=0, atol=0)
