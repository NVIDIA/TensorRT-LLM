# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-free beam-search scheduling, admission and output assembly checks."""

import functools
import types

import pytest
import torch
from beam_search_test_utils import make_beam_request, set_beam_iteration

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.sampler import BeamHistory
from tensorrt_llm._torch.pyexecutor.sampler.beam_search import (
    BEAM_SEARCH_PAD_TOKEN,
    BeamSearchEarlyStop,
    BeamSearchParams,
    BeamSearchStep,
    CBAGroupHost,
    _assemble_final_beams,
    _gather_beam_path,
    _prepare_beam_history_cba,
    finalize_beam,
)
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest as CppLlmRequest
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("params_as_tensors", [False, True])
@pytest.mark.parametrize(
    "length_penalty,diversity_rate",
    [(0.5, 0.0), (2.0, 2.0), (0.0, 0.5)],
)
def test_beam_candidate_topk_equivalence(length_penalty, diversity_rate, params_as_tensors):
    """The two-stage op must match naive full-matrix adjustment + flat topk
    for length penalty, diversity, and their combination."""
    from tensorrt_llm._torch.pyexecutor.sampler.beam_search import beam_candidate_topk

    torch.manual_seed(7)
    batch_size, beam_width_in, beam_width_out, vocab_size = 5, 4, 4, 1000
    logprobs = -torch.rand((batch_size, beam_width_in, vocab_size)) * 20.0
    # emulate a finished beam row: -inf everywhere except a frozen entry
    logprobs[0, 1, :] = float("-inf")
    logprobs[0, 1, 0] = -1.5
    cand_gen_lengths = torch.randint(1, 30, (batch_size, beam_width_in), dtype=torch.int32)

    def as_param(value):
        if value == 0.0:
            return None
        if params_as_tensors:
            return torch.full((batch_size,), value)
        return value

    sorted_logprobs, predecessor_beams, tokens = beam_candidate_topk(
        logprobs,
        beam_width_out=beam_width_out,
        length_penalty=as_param(length_penalty),
        cand_gen_lengths=cand_gen_lengths if length_penalty else None,
        diversity_rate=as_param(diversity_rate),
    )

    # Naive reference: adjust the full candidate matrix, flat topk.
    adjusted = logprobs
    if diversity_rate:
        adjusted = adjusted + diversity_rate * torch.arange(
            beam_width_in, dtype=logprobs.dtype
        ).view(1, -1, 1)
    if length_penalty:
        factor = cand_gen_lengths.float().pow(length_penalty)
        adjusted = adjusted / factor.unsqueeze(-1)
    _, ref_indices = torch.topk(
        adjusted.view(batch_size, -1), k=beam_width_out, sorted=True, dim=-1
    )
    ref_logprobs = logprobs.view(batch_size, -1).gather(1, ref_indices)

    torch.testing.assert_close(sorted_logprobs, ref_logprobs)
    torch.testing.assert_close(predecessor_beams, (ref_indices // vocab_size).to(torch.int32))
    torch.testing.assert_close(tokens, (ref_indices % vocab_size).to(torch.int32))


@pytest.mark.parametrize(
    "beam_width_array, expected",
    [
        # Index is (iteration - 1), clamped at both ends.
        ([2, 3, 4], [2, 2, 3, 4]),
        ([2, 2, 4], [2, 2, 2, 4]),
        ([4, 4, 4], [4, 4, 4, 4]),
    ],
    ids=["widening", "flat_then_widening", "constant"],
)
def test_vbws_beam_width_by_iter_follows_array(beam_width_array: list[int], expected: list[int]):
    """Python and C++ widths follow the schedule and clamp past its end."""
    request = make_beam_request(beam_width_array)
    # Explicit expected prefixes cover the context step and every width change;
    # subsequent steps must keep the last width in both implementations.
    expected_widths = expected + [expected[-1]] * 8
    for iteration, width in enumerate(expected_widths[:-1]):
        set_beam_iteration(request, iteration)
        next_width = expected_widths[iteration + 1]
        assert request.get_beam_width_by_iter() == width
        assert CppLlmRequest.get_beam_width_by_iter(request, False) == width
        assert request.get_beam_width_by_iter(for_next_iteration=True) == next_width
        assert CppLlmRequest.get_beam_width_by_iter(request, True) == next_width

    if len(set(beam_width_array)) == 1:
        fixed = make_beam_request(None, max_beam_width=beam_width_array[0])
        for iteration in range(len(expected_widths)):
            set_beam_iteration(request, iteration)
            set_beam_iteration(fixed, iteration)
            for next_iteration in (False, True):
                assert request.get_beam_width_by_iter(
                    for_next_iteration=next_iteration
                ) == fixed.get_beam_width_by_iter(for_next_iteration=next_iteration)


def test_vbws_width_follows_python_iter_when_response_handling_lags():
    """The width schedule tracks the counter the sampling loop advances.

    Under the overlap scheduler the sampler advances ``py_decoding_iter`` in
    ``_update_requests`` and ``_handle_responses`` copies it into
    ``decoding_iter`` later in the same iteration, so between the two the
    C++-backed counter trails by one step. Indexing the schedule with it made a
    widening run repeat a width: beam_width_array=[2, 3, 4] sampled
    1 -> 2, 2 -> 2, 2 -> 3 instead of 1 -> 2, 2 -> 3, 3 -> 4, ending one width
    short of what was asked for
    (test_beam_search_vbws_widening_terminal_step).
    """
    request = make_beam_request([2, 3, 4])
    # Iteration index 0..3 through a three-entry array, clamped past its end.
    expected_in = [2, 2, 3, 4]
    expected_out = [2, 3, 4, 4]
    for py_iteration in range(len(expected_in)):
        request.py_decoding_iter = py_iteration
        # Response handling has not caught up with the sampler yet.
        request.decoding_iter = max(py_iteration - 1, 0)
        assert request.get_beam_width_by_iter() == expected_in[py_iteration]
        assert request.get_beam_width_by_iter(for_next_iteration=True) == expected_out[py_iteration]


@pytest.mark.parametrize(
    "beam_width_array, accepted",
    [
        ([2, 3, 4], True),
        ([4, 4, 4], True),
        ([2, 2, 4], True),
        ([4, 3, 2], False),
        ([2, 4, 3], False),
    ],
)
def test_vbws_rejects_decreasing_beam_width_array(beam_width_array: list[int], accepted: bool):
    """Only non-decreasing beam_width_array values are supported.

    The documented VBWS semantics only cover widening (see getBeamWidthByIter
    in llmRequest.h), and both samplers depend on it: a step writes the leading
    beam_width_out rows of the beam state while finalize reads py_beam_width
    (the array maximum) of them, so a narrowing array would return beams whose
    ancestry and cumulative log-probs are left over from an earlier, wider
    step. Reject at admission instead of emitting silently stale beams.
    """

    request = make_beam_request(beam_width_array)
    # The request itself is still constructible; admission is what rejects it,
    # and py_beam_width is the array maximum either way.
    assert request.py_beam_width == max(beam_width_array)

    # Exercise the real beam admission checks with unrelated cache, transport,
    # token-range, sampler, model, and budget validation disabled in the
    # executor stub.
    executor = types.SimpleNamespace(
        _is_kv_manager_v2=True,
        max_beam_width=request.py_beam_width,
        kv_cache_manager=None,
        kv_cache_transceiver=None,
        _configure_csa2_remote_tail=lambda _request: None,
        _validate_token_id_range=lambda _request: None,
        sampler=types.SimpleNamespace(validate_request=lambda _request: None),
        model_engine=types.SimpleNamespace(model=None),
        _validate_request_budget=lambda _request: None,
        _validate_embedding_bias=lambda _request: None,
    )
    validate = functools.partial(
        PyExecutor._validate_request,
        executor,  # type: ignore[arg-type]  # minimal executor stub
    )

    if accepted:
        validate(request)
    else:
        with pytest.raises(ValueError, match="decreases"):
            validate(request)


def test_gather_beam_path_follows_cache_indirection():
    """Beam ancestry is reconstructed by following cache_indirection.

    Where C++ walks parent pointers back one step at a time, the Python side
    keeps cache_indirection as an already-flattened ancestry table and
    reconstructs a beam's tokens with a single gather. This pins that gather,
    which both finalization paths share.

    NB: this used to drive _prepare_beam_history end to end. That is now the
    CBA path, which merges the finished-candidate pool into the active beams
    and reorders the result by normalized score, so a per-beam expectation
    computed from cache_indirection alone no longer describes its output. The
    merge and ordering are covered by
    test_cba_finalize_merges_pool_and_orders_by_score below.
    """

    beam_width = 3
    num_generated_tokens = 5

    torch.manual_seed(42)
    # current_path[beam, t] is the token beam `beam` held at position t before
    # correction; cache_indirection[beam, t] names the beam it descended from
    # at that position.
    current_path = torch.randint(
        0, 100, (beam_width, num_generated_tokens), dtype=torch.int32, device="cpu"
    )
    cache_indirection = torch.randint(
        0, beam_width, (beam_width, num_generated_tokens), dtype=torch.int64, device="cpu"
    )

    corrected = _gather_beam_path(current_path=current_path, cache_indirection=cache_indirection)

    expected = torch.zeros_like(current_path)
    for beam in range(beam_width):
        for t in range(num_generated_tokens):
            expected[beam, t] = current_path[cache_indirection[beam, t], t]
    torch.testing.assert_close(corrected, expected)

    # An identity table must leave every beam untouched.
    identity = (
        torch.arange(beam_width, device="cpu", dtype=torch.int64)
        .view(-1, 1)
        .expand(-1, num_generated_tokens)
        .contiguous()
    )
    torch.testing.assert_close(
        _gather_beam_path(current_path=current_path, cache_indirection=identity), current_path
    )


def test_cba_finalize_merges_pool_and_orders_by_score():
    """CBA finalization ranks pool entries against the live beams.

    _prepare_beam_history_cba concatenates the finished-candidate pool with
    the active beams, orders the union by normalized score and emits the top
    num_beams. Pool entries carry their own (shorter) length and are padded to
    the output width; active beams contribute the full generated window. This
    pins that selection, which the plain ancestry gather in
    test_gather_beam_path_follows_cache_indirection does not cover.
    """
    num_beams = 3
    num_generated = 4
    pad = BEAM_SEARCH_PAD_TOKEN

    request = make_beam_request(None, max_beam_width=num_beams)
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    request.py_decoding_iter = num_generated
    request.decoding_iter = num_generated
    request.py_seq_slot = 0
    prompt_len = request.py_prompt_len
    # num_generated_tokens is derived from the request's token count, so give
    # it the generated tokens the beam state below describes. The last token
    # of the step is not added yet, hence num_generated - 1.
    request.set_generated_tokens([[0] * (num_generated - 1)] * num_beams)

    total = prompt_len + num_generated
    # Identity ancestry: each active beam keeps its own tokens, so the ordering
    # rather than the gather is what this test observes.
    cache_indirection = (
        torch.arange(num_beams, dtype=torch.int64)
        .view(-1, 1)
        .expand(-1, total)
        .contiguous()
        .unsqueeze(0)
    )
    original_tokens = torch.zeros((1, num_beams, total), dtype=torch.int32)
    original_tokens[0, :, prompt_len:] = torch.tensor(
        [[31, 32, 33, 34], [41, 42, 43, 44], [51, 52, 53, 54]], dtype=torch.int32
    )

    # Pool: entry 0 outranks every live beam, entry 1 sits between them,
    # entry 2 is an unused (-inf) slot that must never be selected.
    cba_tokens = torch.full((1, num_beams, total), pad, dtype=torch.int32)
    cba_tokens[0, 0, :2] = torch.tensor([11, 12], dtype=torch.int32)
    cba_tokens[0, 1, :3] = torch.tensor([21, 22, 23], dtype=torch.int32)

    cba_group = CBAGroupHost(
        pos={0: 0},
        token_base=0,
        should_stop=torch.tensor([True]),
        cache_indirection=cache_indirection,
        original_tokens=original_tokens,
        cum=torch.tensor([[7.0, 5.0, 3.0]]),
        cba_tokens=cba_tokens,
        cba_cum=torch.tensor([[90.0, 6.0, 0.0]]),
        cba_normed=torch.tensor([[90.0, 6.0, float("-inf")]]),
        cba_lengths=torch.tensor([[2, 3, 0]], dtype=torch.int32),
        original_log_probs=None,
        cba_log_probs=None,
    )

    builder = _prepare_beam_history_cba(request, cba_group=cba_group)
    assert builder is not None
    history = builder()
    assert history is not None

    # Ranking over the union: pool0=90 > active0=7 > pool1=6, so the second
    # pool entry and the two weaker live beams drop out.
    torch.testing.assert_close(
        history.tokens,
        torch.tensor(
            [
                [11, 12, pad, pad],  # pool entry, padded past its length
                [31, 32, 33, 34],  # best live beam, full window
                [21, 22, 23, pad],  # next pool entry
            ],
            dtype=torch.int32,
        ),
    )
    assert history.cum_logprobs is not None
    torch.testing.assert_close(history.cum_logprobs, torch.tensor([90.0, 7.0, 6.0]))


def test_cba_finalize_collects_beams_a_widening_step_produced():
    """CBA finalization reads the width the step produced, not its input width.

    A VBWS run whose last step widens (3 -> 4 here) leaves four live beams in
    the snapshot being finalized. Slicing that snapshot with the step's input
    width keeps only three of them, so the fourth output beam stays padded and
    reaches the caller as an empty sequence: with beam_width_array=[16, 32,
    100] and max_tokens=3 that is 84 of the 100 requested outputs.

    Both iteration counters are set to the same value, the state the
    non-overlap loop is in, so this pins the finalize width on its own rather
    than through the width schedule.
    """
    beam_width_array = [2, 3, 4]
    num_beams = max(beam_width_array)  # the requested output beams
    num_generated = len(beam_width_array)  # the run ends on the widening step
    pad = BEAM_SEARCH_PAD_TOKEN

    request = make_beam_request(beam_width_array, max_beam_width=num_beams)
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    # The last step is the third one: it starts from 3 beams and produces 4.
    request.py_decoding_iter = num_generated - 1
    request.decoding_iter = num_generated - 1
    request.py_seq_slot = 0
    prompt_len = request.py_prompt_len
    # num_generated_tokens is derived from the request's token count, so give it
    # the generated tokens the beam state below describes. The last token of
    # the step is not added yet, hence num_generated - 1.
    request.set_generated_tokens([[0] * (num_generated - 1)] * num_beams)
    assert request.get_beam_width_by_iter() == num_beams - 1
    assert request.get_beam_width_by_iter(for_next_iteration=True) == num_beams

    total = prompt_len + num_generated
    # Identity ancestry: the history must hold exactly the live beams, so what
    # this test observes is how many of them were collected.
    cache_indirection = (
        torch.arange(num_beams, dtype=torch.int64)
        .view(-1, 1)
        .expand(-1, total)
        .contiguous()
        .unsqueeze(0)
    )
    active_tokens = [[31, 32, 33], [41, 42, 43], [51, 52, 53], [61, 62, 63]]
    original_tokens = torch.zeros((1, num_beams, total), dtype=torch.int32)
    original_tokens[0, :, prompt_len:] = torch.tensor(active_tokens, dtype=torch.int32)

    # An empty pool: every output beam has to come from the live ones.
    cba_group = CBAGroupHost(
        pos={0: 0},
        token_base=0,
        should_stop=torch.tensor([True]),
        cache_indirection=cache_indirection,
        original_tokens=original_tokens,
        cum=torch.tensor([[4.0, 3.0, 2.0, 1.0]]),
        cba_tokens=torch.full((1, num_beams, total), pad, dtype=torch.int32),
        cba_cum=torch.zeros((1, num_beams)),
        cba_normed=torch.full((1, num_beams), float("-inf")),
        cba_lengths=torch.zeros((1, num_beams), dtype=torch.int32),
        original_log_probs=None,
        cba_log_probs=None,
    )

    builder = _prepare_beam_history_cba(request, cba_group=cba_group)
    assert builder is not None
    history = builder()
    assert history is not None

    torch.testing.assert_close(history.tokens, torch.tensor(active_tokens, dtype=torch.int32))
    assert (history.tokens != pad).all(), (
        "a finalized beam is still padded, so the beams the widening step "
        f"added were dropped: {history.tokens.tolist()}"
    )
    assert history.cum_logprobs is not None
    torch.testing.assert_close(history.cum_logprobs, torch.tensor([4.0, 3.0, 2.0, 1.0]))


def _reference_assemble_final_beams(
    order,
    all_normed,
    pool_width,
    width,
    num_generated_tokens,
    cba_tokens,
    cba_lengths,
    cba_cum,
    active_path,
    active_cum,
    cba_log_probs,
    active_lp_path,
):
    """Per-beam loop the vectorized _assemble_final_beams replaces."""
    num_beams = order.size(0)
    tokens = torch.full((num_beams, width), BEAM_SEARCH_PAD_TOKEN, dtype=torch.int32)
    cum_logprobs = torch.zeros((num_beams,), dtype=torch.float32)
    log_probs = None
    if cba_log_probs is not None:
        log_probs = torch.zeros((num_beams, width), dtype=torch.float32)
    for out_idx, merged_idx in enumerate(order.tolist()):
        if not torch.isfinite(all_normed[merged_idx]):
            continue
        if merged_idx < pool_width:
            entry_len = int(cba_lengths[merged_idx].item())
            tokens[out_idx, :entry_len] = cba_tokens[merged_idx, :entry_len]
            cum_logprobs[out_idx] = cba_cum[merged_idx]
            if log_probs is not None:
                log_probs[out_idx, :entry_len] = cba_log_probs[merged_idx, :entry_len]
        else:
            active_idx = merged_idx - pool_width
            tokens[out_idx, :num_generated_tokens] = active_path[active_idx]
            cum_logprobs[out_idx] = active_cum[active_idx]
            if log_probs is not None:
                log_probs[out_idx, :num_generated_tokens] = active_lp_path[active_idx]
    return tokens, cum_logprobs, log_probs


@pytest.mark.parametrize("with_log_probs", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_assemble_final_beams_matches_reference(seed, with_log_probs):
    """The vectorized final-beam assembly reproduces the per-beam loop.

    Random pools mix finished entries of varying length, unused (-inf) pool
    slots and live beams (some harvested to -inf), including the under-filled
    case where fewer finite candidates exist than output beams so some rows
    must stay padded.
    """
    gen = torch.Generator().manual_seed(seed)
    pool_width = int(torch.randint(1, 6, (1,), generator=gen))
    active_width = int(torch.randint(1, 6, (1,), generator=gen))
    num_generated = int(torch.randint(1, 5, (1,), generator=gen))
    snap_width = num_generated + int(torch.randint(0, 3, (1,), generator=gen))
    num_beams = int(torch.randint(1, pool_width + active_width + 1, (1,), generator=gen))

    cba_lengths = torch.randint(0, snap_width + 1, (pool_width,), generator=gen).to(torch.int32)
    cba_normed = torch.randn(pool_width, generator=gen)
    cba_normed[cba_lengths == 0] = float("-inf")
    cba_cum = torch.randn(pool_width, generator=gen)
    cba_tokens = torch.randint(0, 1000, (pool_width, snap_width), generator=gen).to(torch.int32)
    active_path = torch.randint(0, 1000, (active_width, num_generated), generator=gen).to(
        torch.int32
    )
    active_cum = torch.randn(active_width, generator=gen)
    active_cum[torch.rand(active_width, generator=gen) < 0.5] = float("-inf")
    all_normed = torch.cat([cba_normed, active_cum])
    order = torch.argsort(all_normed, descending=True)[:num_beams]
    width = max(num_generated, int(cba_lengths.max().item()))

    cba_log_probs = active_lp_path = None
    if with_log_probs:
        cba_log_probs = torch.randn(pool_width, snap_width, generator=gen)
        active_lp_path = torch.randn(active_width, num_generated, generator=gen)

    kwargs = dict(
        order=order,
        all_normed=all_normed,
        pool_width=pool_width,
        width=width,
        num_generated_tokens=num_generated,
        cba_tokens=cba_tokens,
        cba_lengths=cba_lengths,
        cba_cum=cba_cum,
        active_path=active_path,
        active_cum=active_cum,
        cba_log_probs=cba_log_probs,
        active_lp_path=active_lp_path,
    )
    tokens, cum, lps = _assemble_final_beams(**kwargs)  # type: ignore
    ref_tokens, ref_cum, ref_lps = _reference_assemble_final_beams(**kwargs)
    torch.testing.assert_close(tokens, ref_tokens)
    torch.testing.assert_close(cum, ref_cum)
    if with_log_probs:
        torch.testing.assert_close(lps, ref_lps)
    else:
        assert lps is None and ref_lps is None


def test_cba_finalize_reads_snapshot_from_token_base():
    """The group snapshot may start at the smallest prompt length.

    prepare_cba_group_host copies cache_indirection / original_tokens from
    ``token_base`` (the group's minimum prompt length) rather than position 0;
    the builder must offset its generated-region slice accordingly.
    """
    num_beams = 2
    num_generated = 3
    request = make_beam_request(None, max_beam_width=num_beams)
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    request.py_decoding_iter = num_generated
    request.decoding_iter = num_generated
    request.py_seq_slot = 0
    prompt_len = request.py_prompt_len
    request.set_generated_tokens([[0] * (num_generated - 1)] * num_beams)
    total = prompt_len + num_generated
    # Snapshot starting two positions before the prompt end: the generated
    # window sits at columns [2, 2 + num_generated).
    token_base = prompt_len - 2
    snap = total - token_base
    cache_indirection = (
        torch.arange(num_beams, dtype=torch.int64)
        .view(-1, 1)
        .expand(-1, snap)
        .contiguous()
        .unsqueeze(0)
    )
    original_tokens = torch.zeros((1, num_beams, snap), dtype=torch.int32)
    generated = torch.tensor([[31, 32, 33], [41, 42, 43]], dtype=torch.int32)
    original_tokens[0, :, 2:] = generated
    cba_group = CBAGroupHost(
        pos={0: 0},
        token_base=token_base,
        should_stop=torch.tensor([True]),
        cache_indirection=cache_indirection,
        original_tokens=original_tokens,
        cum=torch.tensor([[2.0, 1.0]]),
        cba_tokens=torch.full(
            (1, num_beams, num_generated), BEAM_SEARCH_PAD_TOKEN, dtype=torch.int32
        ),
        cba_cum=torch.zeros((1, num_beams)),
        cba_normed=torch.full((1, num_beams), float("-inf")),
        cba_lengths=torch.zeros((1, num_beams), dtype=torch.int32),
        original_log_probs=None,
        cba_log_probs=None,
    )
    builder = _prepare_beam_history_cba(request, cba_group=cba_group)
    assert builder is not None
    history = builder()
    assert history is not None
    torch.testing.assert_close(history.tokens, generated)
    torch.testing.assert_close(history.cum_logprobs, torch.tensor([2.0, 1.0]))


def _beam_params(
    *,
    early_stopping: BeamSearchEarlyStop,
    beam_width_in: int = 2,
    beam_width_out: int = 2,
    temperature: float = 1.0,
    length_penalty: float = 0.0,
    diversity_rate: float = 0.0,
) -> BeamSearchParams:
    return BeamSearchParams(
        beam_width_in=beam_width_in,
        beam_width_out=beam_width_out,
        row_stride=beam_width_in,
        temperature=temperature,
        length_penalty=length_penalty,
        diversity_rate=diversity_rate,
        early_stopping=early_stopping,
    )


def test_beam_step_from_params():
    for early_stopping in BeamSearchEarlyStop:
        step = BeamSearchStep.from_params(
            [_beam_params(early_stopping=early_stopping)], torch.device("cpu"), return_probs=False
        )
        assert step._beam_width_in == step._beam_width_out == 2
        assert step._early_stopping is early_stopping
        assert step._length_penalty is None
        assert step._diversity_rate is None
    step = BeamSearchStep.from_params(
        [
            _beam_params(
                early_stopping=BeamSearchEarlyStop.NEVER, length_penalty=1.5, diversity_rate=0.5
            )
        ],
        torch.device("cpu"),
        return_probs=False,
    )
    torch.testing.assert_close(step._length_penalty, torch.tensor([1.5]))
    torch.testing.assert_close(step._diversity_rate, torch.tensor([0.5]))


def test_beam_step_rejects_mixed_row_stride():
    params = _beam_params(early_stopping=BeamSearchEarlyStop.FALSE)
    with pytest.raises(ValueError):
        BeamSearchStep.from_params(
            [params, params._replace(row_stride=params.row_stride + 1)],
            torch.device("cpu"),
            return_probs=False,
        )


def test_finalize_beam_trims_padding():
    request = make_beam_request(None, max_beam_width=3)
    prompt = [1, 2, 3]
    tokens = torch.tensor([[10, 11, 12], [20, 21, 22], [30, 31, 32]], dtype=torch.int32)
    request.py_return_log_probs = False
    for padded in (False, True):
        if padded:
            tokens[0, 1:] = BEAM_SEARCH_PAD_TOKEN
        finalize_beam(request, BeamHistory(tokens=tokens))
        expected = [[10] if padded else [10, 11, 12], [20, 21, 22], [30, 31, 32]]
        assert request.get_tokens() == [prompt + row for row in expected]


@pytest.mark.parametrize("use_beam_search", [False, None])
def test_use_beam_search_disabled_rejects_multiple_returns(use_beam_search):
    params = dict(max_tokens=8, n=1, best_of=4, end_id=-1)
    if use_beam_search is not None:
        params["use_beam_search"] = use_beam_search
    with pytest.raises(
        ValueError, match="Greedy decoding in the LLM API does not allow multiple returns"
    ):
        SamplingParams(**params)
