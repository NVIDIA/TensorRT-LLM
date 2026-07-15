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
    BeamPenaltyMetadata,
    apply_beam_occurrence_penalties_triton,
    apply_occurrence_penalties_triton,
    update_occurrence_workspace,
)
from tensorrt_llm._torch.pyexecutor.sampler.sampler import _OccurrencePenaltyHandler
from tensorrt_llm._torch.pyexecutor.sampler.sampling_utils import (
    BeamSearchMetadata,
    beam_search_sampling_batch,
)
from tensorrt_llm.bindings.executor import FinishReason


def _col(values):
    return torch.tensor(values, dtype=torch.float32, device="cuda").view(-1, 1)


def _dense_penalty_reference(logits, counts, presence, rep, pre, freq, temp):
    """Dense post-temperature reference for ``apply_occurrence_penalties_triton``.

    Follows the ``batchApplyPenalty`` order: temperature (``logit /= temp``), then
    repetition where the token is present anywhere (``counts > 0`` or the prefix bitmap),
    then presence + frequency where counted (``counts > 0``). ``rep/pre/freq/temp`` are
    per-row ``[A, 1]`` tensors. The kernel is pre-temperature-division, so callers compare
    ``kernel_out / temp`` against this.
    """
    lt = logits.float() / temp
    present = counts > 0
    if presence is not None:
        present = present | (presence > 0)
    lt = torch.where(present, torch.where(lt < 0, lt * rep, lt / rep), lt)
    counts_f = counts.to(torch.float32)
    sub = torch.where(counts > 0, pre + freq * counts_f, lt.new_zeros(()))
    return lt - sub


@pytest.mark.parametrize(
    "name,rep,pre,freq,temp,use_prefix",
    [
        # repetition only, exercises the sign branch (>1, <1) at temp=1
        ("repetition", [1.3, 2.0, 0.7], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], False),
        # presence only
        ("presence", [1.0, 1.0], [0.5, 1.5], [0.0, 0.0], [1.0, 1.0], False),
        # frequency only (counts > 1 -> proportional)
        ("frequency", [1.0, 1.0], [0.0, 0.0], [0.4, 0.9], [1.0, 1.0], False),
        # combined with temperature != 1 (exercises the temperature coupling)
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
def test_regular_triton_matches_cpp_logits_reference(name, rep, pre, freq, temp, use_prefix):
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
    active_rows = torch.arange(A, dtype=torch.int64, device="cuda")

    got = logits.clone()
    apply_occurrence_penalties_triton(
        got,
        active_rows,
        active_rows,
        counts,
        presence,
        rep_t.squeeze(1),
        pre_t.squeeze(1),
        freq_t.squeeze(1),
        temp_t.squeeze(1),
    )
    ref = _dense_penalty_reference(logits, counts, presence, rep_t, pre_t, freq_t, temp_t)
    # the kernel is pre-temperature-division; divide by temp to compare to the final value.
    torch.testing.assert_close(got / temp_t, ref, rtol=1e-4, atol=1e-4)


def test_triton_indirect_indexing_bf16():
    # Permuted active_rows / row_slots: penalize a subset of logits rows, each reading a
    # (possibly repeated) slot's counts/params; other rows must stay untouched. bfloat16
    # also covers the fp32-compute -> bf16-store cast path.
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

    apply_occurrence_penalties_triton(
        logits, active_rows, row_slots, counts, None, rep, pre, freq, temp
    )

    ref = _dense_penalty_reference(
        orig[active_rows],
        counts[row_slots],
        None,
        rep[row_slots].view(-1, 1),
        pre[row_slots].view(-1, 1),
        freq[row_slots].view(-1, 1),
        temp[row_slots].view(-1, 1),
    )
    got = logits[active_rows].float() / temp[row_slots].view(-1, 1)
    torch.testing.assert_close(got, ref, rtol=3e-2, atol=3e-2)
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


def test_inactive_request_only_clears_active_beam_transition(monkeypatch):
    handler = _OccurrencePenaltyHandler(
        max_num_sequences=1,
        max_beam_width=2,
        device="cpu",
    )
    request = SimpleNamespace(
        sampling_config=SimpleNamespace(
            repetition_penalty=None,
            presence_penalty=None,
            frequency_penalty=None,
        )
    )
    fills: list[tuple[torch.Tensor, int, bool | float | int]] = []
    monkeypatch.setattr(
        handler,
        "_fill_slot",
        lambda tensor, slot, value: fills.append((tensor, slot, value)),
    )

    handler.prepare_for_new_request(request, slot=0)
    assert fills == []

    handler._slots[0] = handler._SlotState(
        prompt_ignore_length=0,
        counted_len=0,
        use_beam_search=True,
    )
    handler.prepare_for_new_request(request, slot=0)
    assert len(fills) == 1
    tensor, slot, value = fills[0]
    assert tensor is handler.store.beam_active
    assert slot == 0
    assert value is False


def _make_beam_penalty_metadata(
    *,
    active: list[bool],
    prompt_lengths: list[int],
    repetition: list[float],
    presence: list[float],
    frequency: list[float],
    beam_width: int,
    vocab: int,
) -> BeamPenaltyMetadata:
    num_slots = len(active)
    workspace_a = torch.zeros(
        num_slots,
        beam_width,
        2,
        vocab,
        dtype=torch.int32,
        device="cuda",
    )
    return BeamPenaltyMetadata(
        workspace_a=workspace_a,
        workspace_b=torch.zeros_like(workspace_a),
        active=torch.tensor(active, dtype=torch.bool, device="cuda"),
        buffer_indices=torch.zeros(num_slots, dtype=torch.int32, device="cuda"),
        prompt_lengths=torch.tensor(prompt_lengths, dtype=torch.int32, device="cuda"),
        new_tokens=torch.zeros(1, num_slots, beam_width, dtype=torch.int32, device="cuda"),
        predecessor_beams=torch.zeros(num_slots, beam_width, dtype=torch.int32, device="cuda"),
        repetition=torch.tensor(repetition, dtype=torch.float32, device="cuda"),
        presence=torch.tensor(presence, dtype=torch.float32, device="cuda"),
        frequency=torch.tensor(frequency, dtype=torch.float32, device="cuda"),
    )


def test_beam_triton_parent_aware_double_buffer():
    """Beam penalties follow the previous step's parent mapping across buffer swaps."""
    vocab = 32
    beam_width = 2
    slot = 1
    metadata = _make_beam_penalty_metadata(
        active=[False, True],
        prompt_lengths=[0, 3],
        repetition=[1.0, 1.4],
        presence=[0.0, 0.6],
        frequency=[0.0, 0.25],
        beam_width=beam_width,
        vocab=vocab,
    )
    workspace_a = metadata.workspace_a
    workspace_b = metadata.workspace_b
    # Token 2 is in the ignored prompt prefix. Tokens 3 and 4 are counted.
    workspace_a[slot, 0, 0, 2] = 1
    workspace_a[slot, 0, 1, 3] = 2
    workspace_a[slot, 0, 1, 4] = 1

    def reference(log_probs, workspace):
        return _dense_penalty_reference(
            log_probs,
            workspace[:, 1],
            workspace[:, 0],
            metadata.repetition[slot].expand(log_probs.size(0), 1),
            metadata.presence[slot].expand(log_probs.size(0), 1),
            metadata.frequency[slot].expand(log_probs.size(0), 1),
            torch.ones(log_probs.size(0), 1, device="cuda"),
        )

    # The inactive row is already in generation while the active row is in context.
    # It must leave both logits and workspace untouched and must not toggle its buffer.
    seq_slots = torch.tensor([0, slot], dtype=torch.int64, device="cuda")
    seq_lens = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
    context = torch.log_softmax(torch.randn(2, vocab, device="cuda"), dim=-1)
    inactive_row = context[:1].clone()
    expected_context = reference(context[1:], workspace_a[slot, :1].clone())
    apply_beam_occurrence_penalties_triton(context, metadata, seq_slots, seq_lens, beam_width=1)
    torch.testing.assert_close(context[:1], inactive_row, rtol=0, atol=0)
    torch.testing.assert_close(context[1:], expected_context, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(workspace_a[0], torch.zeros_like(workspace_a[0]))
    torch.testing.assert_close(workspace_b[0], torch.zeros_like(workspace_b[0]))
    torch.testing.assert_close(metadata.buffer_indices, torch.zeros_like(metadata.buffer_indices))

    # First generation step: both beams descend from beam 0 and add different tokens.
    seq_slots = seq_slots[1:]
    metadata.new_tokens[0, slot] = torch.tensor([5, 6], dtype=torch.int32, device="cuda")
    metadata.predecessor_beams[slot] = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
    generation = torch.log_softmax(torch.randn(2, vocab, device="cuda"), dim=-1)
    expected_workspace_b = workspace_a[slot, :1].expand(2, -1, -1).clone()
    expected_workspace_b[0, 1, 5] += 1
    expected_workspace_b[1, 1, 6] += 1
    expected_generation = reference(generation, expected_workspace_b)
    seq_lens = torch.tensor([4], dtype=torch.int32, device="cuda")
    apply_beam_occurrence_penalties_triton(
        generation, metadata, seq_slots, seq_lens, beam_width=beam_width
    )
    torch.testing.assert_close(generation, expected_generation, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(workspace_b[slot], expected_workspace_b)
    assert metadata.buffer_indices[slot] == 1

    # Second generation step: swap the parents and return to workspace A.
    metadata.new_tokens[0, slot] = torch.tensor([7, 8], dtype=torch.int32, device="cuda")
    metadata.predecessor_beams[slot] = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    generation = torch.log_softmax(torch.randn(2, vocab, device="cuda"), dim=-1)
    expected_workspace_a = expected_workspace_b[[1, 0]].clone()
    expected_workspace_a[0, 1, 7] += 1
    expected_workspace_a[1, 1, 8] += 1
    expected_generation = reference(generation, expected_workspace_a)
    seq_lens.fill_(5)
    apply_beam_occurrence_penalties_triton(
        generation, metadata, seq_slots, seq_lens, beam_width=beam_width
    )
    torch.testing.assert_close(generation, expected_generation, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(workspace_a[slot], expected_workspace_a)
    assert metadata.buffer_indices[slot] == 0


@pytest.mark.parametrize(
    "name,rep,pre,freq,temperature",
    [
        ("repetition", 1.5, 0.0, 0.0, 1.0),
        ("repetition_reward", 0.7, 0.0, 0.0, 1.0),
        ("presence", 1.0, 0.8, 0.0, 1.0),
        ("frequency", 1.0, 0.0, 0.6, 1.0),
        ("combined_temperature", 1.4, 0.5, 0.25, 2.0),
    ],
)
def test_beam_penalty_matches_logits_reference(name, rep, pre, freq, temperature):
    """Validate beam scores against an independent penalty reference."""
    vocab = 16
    beam_width = 2
    logits = torch.full((1, vocab), -3.0, dtype=torch.float32, device="cuda")
    logits[0, [1, 4, 7, 8]] = torch.tensor([4.0, 3.5, 3.0, 2.8], dtype=torch.float32, device="cuda")
    penalty = _make_beam_penalty_metadata(
        active=[True],
        prompt_lengths=[3],
        repetition=[rep],
        presence=[pre],
        frequency=[freq],
        beam_width=beam_width,
        vocab=vocab,
    )
    workspace_a = penalty.workspace_a
    workspace_a[0, 0, 0, 1] = 1
    workspace_a[0, 0, 1, 4] = 2
    seq_slots = torch.zeros(1, dtype=torch.int64, device="cuda")
    seq_lens = penalty.prompt_lengths.clone()
    initial_cum_log_prob = 0.3
    cum_log_probs = torch.tensor([[initial_cum_log_prob, -0.4]], dtype=torch.float32, device="cuda")
    new_log_probs = torch.zeros_like(cum_log_probs)
    metadata = BeamSearchMetadata(
        cache_indirection=torch.zeros(1, beam_width, 8, dtype=torch.int32, device="cuda"),
        cache_indirection_buffer=torch.zeros(1, beam_width, 8, dtype=torch.int32, device="cuda"),
        cum_log_probs=cum_log_probs,
        new_log_probs=new_log_probs,
        seq_slots=seq_slots,
        seq_lens=seq_lens,
        finished_beams=torch.full(
            (1, beam_width),
            FinishReason.NOT_FINISHED.value,
            dtype=torch.int32,
            device="cuda",
        ),
        predecessor_beams=penalty.predecessor_beams,
        seq_offsets=torch.zeros(1, dtype=torch.int64, device="cuda"),
        beam_idx_arange=torch.arange(beam_width, dtype=torch.int32, device="cuda"),
        penalty=penalty,
    )

    tempered_logits = logits / temperature
    normalized = torch.log_softmax(tempered_logits, dim=-1)
    expected_scores = _dense_penalty_reference(
        normalized,
        workspace_a[0, :1, 1],
        workspace_a[0, :1, 0],
        penalty.repetition.view(1, 1),
        penalty.presence.view(1, 1),
        penalty.frequency.view(1, 1),
        torch.ones(1, 1, device="cuda"),
    )
    expected_new_log_probs, expected_tokens = torch.topk(expected_scores, k=beam_width, dim=-1)
    expected_cum_log_probs = expected_new_log_probs + initial_cum_log_prob
    check_temperature_order = name == "combined_temperature"

    next_tokens, softmax = beam_search_sampling_batch(
        logits,
        beam_width_in=1,
        beam_width_out=beam_width,
        beam_search_args=metadata,
        temperature=temperature,
        return_probs=check_temperature_order,
    )

    if check_temperature_order:
        assert softmax is not None
        torch.testing.assert_close(softmax, torch.softmax(tempered_logits, dim=-1).unsqueeze(0))
    torch.testing.assert_close(next_tokens, expected_tokens.to(torch.int32))
    torch.testing.assert_close(cum_log_probs, expected_cum_log_probs, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(new_log_probs, expected_new_log_probs, rtol=1e-5, atol=1e-5)
