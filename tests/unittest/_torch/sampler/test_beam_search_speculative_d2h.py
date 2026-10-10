# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end tests for the beam-history speculative D2H opt-in.

This file covers the code paths gated behind
`TorchLlmArgs.enable_speculative_beam_history_d2h`:

* parity vs the default (synchronous) beam-history D2H path,
* the predictor-miss fallback in
  `BeamSearchHandler._prepare_beam_history._builder`
  (the synchronous `.cpu()` issued when the host-side predictor
  decided the step is non-terminal but the beam still finalizes),
* the predictor-hit path that routes copies through the side stream,
* the two stream-ordering contracts the path relies on: the side-stream
  copier keeping its sources alive until it has read them, and the
  fallback builder awaiting its own non-blocking copies before reading
  them.

The dummy model from `test_beam_search_util` produces deterministic
outputs, so we can compare runs token-for-token without depending on
real model weights.
"""

import gc
import os
import pathlib as _pl
from contextlib import AbstractContextManager, contextmanager, nullcontext
from copy import deepcopy
from typing import Any, Generator, Iterable

import pytest
import torch
from pydantic import ValidationError
from test_beam_search_util import DummyConfigLoader, DummyWeightLoader

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.models.checkpoints import HfCheckpointLoader
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.sampler import SampleStateTorch, TorchSampler
from tensorrt_llm._torch.pyexecutor.sampler.beam_search import (
    BEAM_SEARCH_PAD_TOKEN,
    BeamSearchHandler,
    BeamSearchStore,
)
from tensorrt_llm._torch.pyexecutor.sampler.sampler_features import _SideStreamCopier
from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.llmapi import KvCacheConfig

# Long enough that host-side work issued after the sleep is enqueued
# while the GPU is still inside it: ~0.5-1 s on current parts.
_RACE_WINDOW_CYCLES = 1_000_000_000


@pytest.fixture(scope="module")
def input_prompts() -> list[list[int]]:
    return [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


@pytest.fixture(scope="module")
def fixed_params() -> dict[str, Any]:
    return {"max_tokens": 8, "max_beam_width": 2}


def _build_llm(
    fixed_params: dict[str, Any],
    input_prompts: list[list[int]],
    *,
    enable_speculative_beam_history_d2h: bool = False,
    sampler_force_async_worker: bool = False,
) -> LLM:
    return LLM(
        model=_pl.Path("dummy_path"),
        checkpoint_loader=HfCheckpointLoader(
            weight_loader=DummyWeightLoader(),
            config_loader=DummyConfigLoader(),
        ),
        max_batch_size=fixed_params["max_beam_width"] * len(input_prompts),
        kv_cache_config=KvCacheConfig(max_tokens=10000),  # pyright: ignore
        max_seq_len=32,
        max_beam_width=fixed_params["max_beam_width"],
        disable_overlap_scheduler=True,
        cuda_graph_config=None,
        sampler_force_async_worker=sampler_force_async_worker,
        enable_speculative_beam_history_d2h=enable_speculative_beam_history_d2h,
    )


def _make_sampling_params(
    fixed_params: dict[str, Any], stop_token_ids: list[int] | None
) -> SamplingParams:
    return SamplingParams(
        max_tokens=fixed_params["max_tokens"],
        n=fixed_params["max_beam_width"],
        best_of=fixed_params["max_beam_width"],
        use_beam_search=True,
        end_id=-1,
        stop_token_ids=stop_token_ids,
        include_stop_str_in_output=True,
        additional_model_outputs=["cache_indirection"],
    )


def _generate(
    llm: LLM, prompts: list[list[int]], sampling_params: SamplingParams
) -> list[GenerationResult]:
    outputs = llm.generate(deepcopy(prompts), sampling_params=deepcopy(sampling_params))
    assert isinstance(outputs, list)
    return outputs


def _assert_outputs_equal(
    actual: Iterable[GenerationResult], expected: Iterable[GenerationResult]
) -> None:
    """Assert two beam-search runs produce identical per-beam outputs.

    Compares token ids, finish reasons, and cumulative log probabilities
    for every beam of every prompt. Since the dummy model is fully
    deterministic and the speculative path only changes when the D2H
    copy is issued (not what is computed), parity must be exact.
    """
    actual_list = list(actual)
    expected_list = list(expected)
    assert len(actual_list) == len(expected_list)

    for prompt_idx, (got, exp) in enumerate(zip(actual_list, expected_list)):
        got_beams = list(got.outputs)
        exp_beams = list(exp.outputs)
        assert len(got_beams) == len(exp_beams), (
            f"prompt {prompt_idx}: beam count mismatch ({len(got_beams)} vs {len(exp_beams)})"
        )
        for beam_idx, (gb, eb) in enumerate(zip(got_beams, exp_beams)):
            assert gb.token_ids == eb.token_ids, (
                f"prompt {prompt_idx} beam {beam_idx}: token mismatch "
                f"({gb.token_ids} vs {eb.token_ids})"
            )
            assert gb.finish_reason == eb.finish_reason, (
                f"prompt {prompt_idx} beam {beam_idx}: finish_reason mismatch "
                f"({gb.finish_reason} vs {eb.finish_reason})"
            )
            # cum_logprob is computed identically on both paths; only the
            # D2H timing differs, so equality must be exact.
            assert gb.cumulative_logprob == eb.cumulative_logprob, (
                f"prompt {prompt_idx} beam {beam_idx}: cum_logprob mismatch "
                f"({gb.cumulative_logprob} vs {eb.cumulative_logprob})"
            )


def _run_with_env(
    fixed_params: dict[str, Any],
    input_prompts: list[list[int]],
    monkeypatch: pytest.MonkeyPatch,
    *,
    speculative: bool,
    stop_token_ids: list[int] | None,
    predictor_override: Any = None,
    sampler_force_async_worker: bool = False,
    sampler_method_patches: dict[str, Any] | None = None,
    handler_method_patches: dict[str, Any] | None = None,
) -> list[GenerationResult]:
    """Build a fresh LLM with the speculative flag configured, run beam search, tear down.

    Opt-in is via `TorchLlmArgs.enable_speculative_beam_history_d2h`.
    `predictor_override` patches
    `BeamSearchHandler.predict_is_likely_finishing`,
    `sampler_method_patches` patches arbitrary `TorchSampler` methods and
    `handler_method_patches` arbitrary `BeamSearchHandler` methods; any of
    them forces `TLLM_WORKER_USE_SINGLE_PROCESS=1` so class-level patches
    reach the sampler. The method patches are installed after a warmup
    generate, so they only observe the measured run.
    `sampler_force_async_worker` enables the AsyncWorkerMixin path.
    """
    method_patches = bool(sampler_method_patches) or bool(handler_method_patches)
    needs_single_process = predictor_override is not None or method_patches
    if needs_single_process:
        # Class-level patches do not cross process boundaries; force the
        # sampler to run in-process so the patch is observed.
        monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    if predictor_override is not None:
        monkeypatch.setattr(BeamSearchHandler, "predict_is_likely_finishing", predictor_override)

    gc.collect(2)
    llm = _build_llm(
        fixed_params,
        input_prompts,
        enable_speculative_beam_history_d2h=speculative,
        sampler_force_async_worker=sampler_force_async_worker,
    )
    try:
        with llm:
            sampling_params = _make_sampling_params(fixed_params, stop_token_ids)
            if method_patches:
                # Warmup before installing the hooks.
                _generate(llm, input_prompts, sampling_params)
            with monkeypatch.context() if method_patches else nullcontext() as p:
                for name, replacement in (sampler_method_patches or {}).items():
                    p.setattr(TorchSampler, name, replacement)
                for name, replacement in (handler_method_patches or {}).items():
                    p.setattr(BeamSearchHandler, name, replacement)
                # Run with the hooks installed.
                return _generate(llm, input_prompts, sampling_params)
    finally:
        del llm
        gc.collect(2)


# ---------------------------------------------------------------------------
# Parity: real predictor.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stop_token_ids",
    [None, [15]],
    ids=["no_stop_token", "stop_token_15"],
)
@pytest.mark.threadleak(enabled=False)
def test_speculative_d2h_parity_real_predictor(
    fixed_params: dict[str, Any],
    input_prompts: list[list[int]],
    stop_token_ids: list[int] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature-on output matches feature-off output token-for-token.

    Exercises the real host-side predictor (no override). With
    `stop_token_ids=None` every step is non-terminal until the length
    budget is hit, so the predictor mostly skips D2H. With
    `stop_token_ids=[15]` some prompts may hit the stop token early,
    triggering predictor misses that fall back to a synchronous
    `.cpu()` in `_builder`.
    """
    with monkeypatch.context() as mp_off:
        out_off = _run_with_env(
            fixed_params, input_prompts, mp_off, speculative=False, stop_token_ids=stop_token_ids
        )

    with monkeypatch.context() as mp_on:
        out_on = _run_with_env(
            fixed_params, input_prompts, mp_on, speculative=True, stop_token_ids=stop_token_ids
        )

    _assert_outputs_equal(out_on, out_off)


# ---------------------------------------------------------------------------
# Predictor-miss fallback: synchronous `.cpu()` in `_builder`.
# ---------------------------------------------------------------------------


def _pinned_predictor(value: bool) -> tuple[Any, dict[str, int]]:
    """Return a method that always reports `value`, plus a call counter.

    The counter lets tests assert the patch was actually exercised
    (i.e., the speculative code path was taken), guarding against silent
    regressions where `_prepare_beam_history` stops calling the predictor
    in the speculative branch.
    """
    state = {"calls": 0}

    def _pinned(self, request, *, num_generated_tokens, num_tokens):
        state["calls"] += 1
        return value

    return _pinned, state


@pytest.mark.threadleak(enabled=False)
def test_speculative_d2h_predictor_miss_fallback(
    fixed_params: dict[str, Any],
    input_prompts: list[list[int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force every step to be a predictor miss; verify outputs are correct.

    With the predictor pinned to `False`, the speculative branch of
    `_prepare_beam_history` never stages side-stream copies, so every
    call into `_builder` takes the synchronous `.cpu()` fallback. This
    guards the only path under the speculative branch that issues a
    host-blocking transfer.
    """
    _always_miss, miss_state = _pinned_predictor(False)

    with monkeypatch.context() as mp_off:
        out_off = _run_with_env(
            fixed_params, input_prompts, mp_off, speculative=False, stop_token_ids=[15]
        )

    with monkeypatch.context() as mp_on:
        out_on = _run_with_env(
            fixed_params,
            input_prompts,
            mp_on,
            speculative=True,
            stop_token_ids=[15],
            predictor_override=_always_miss,
        )

    assert miss_state["calls"] > 0, (
        "predictor patch was never invoked; the speculative path did not run "
        "(check that enable_speculative_beam_history_d2h is honored)"
    )
    _assert_outputs_equal(out_on, out_off)


@pytest.mark.threadleak(enabled=False)
def test_speculative_d2h_fallback_builder_runs_only_on_finishing_step(
    fixed_params: dict[str, Any],
    input_prompts: list[list[int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _always_miss, miss_state = _pinned_predictor(False)
    speculative_builder_orig = BeamSearchHandler._speculative_builder
    builder_results: list[bool] = []

    def _counting_speculative_builder(self, request):  # type: ignore[no-untyped-def]
        inner = speculative_builder_orig(self, request)

        def _builder():  # type: ignore[no-untyped-def]
            history = inner()
            builder_results.append(history is not None)
            return history

        return _builder

    with monkeypatch.context() as mp_off:
        out_off = _run_with_env(
            fixed_params, input_prompts, mp_off, speculative=False, stop_token_ids=None
        )

    with monkeypatch.context() as mp_on:
        out_on = _run_with_env(
            fixed_params,
            input_prompts,
            mp_on,
            speculative=True,
            stop_token_ids=None,
            predictor_override=_always_miss,
            handler_method_patches={"_speculative_builder": _counting_speculative_builder},
        )

    assert miss_state["calls"] > 0, (
        "predictor patch was never invoked; the speculative path did not run "
        "(check that enable_speculative_beam_history_d2h is honored)"
    )
    assert len(builder_results) == len(input_prompts), (
        "the fallback builder must run exactly once per request, on its final "
        f"step; it ran {len(builder_results)} times for {len(input_prompts)} requests"
    )
    assert all(builder_results), "every fallback builder invocation must finalize its request"
    _assert_outputs_equal(out_on, out_off)


# ---------------------------------------------------------------------------
# Predictor-hit: every step routes copies through the side stream.
# ---------------------------------------------------------------------------


@pytest.mark.threadleak(enabled=False)
def test_speculative_d2h_predictor_always_hit(
    fixed_params: dict[str, Any],
    input_prompts: list[list[int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force every step to be a predictor hit; verify outputs are correct.

    With the predictor pinned to `True`, every per-step beam history
    is copied off the device via the side-stream copier and the
    `_builder` fallback is never taken. Confirms that the side-stream
    path itself produces bit-exact parity with the default code path.
    """
    _always_hit, hit_state = _pinned_predictor(True)

    with monkeypatch.context() as mp_off:
        out_off = _run_with_env(
            fixed_params, input_prompts, mp_off, speculative=False, stop_token_ids=None
        )

    with monkeypatch.context() as mp_on:
        out_on = _run_with_env(
            fixed_params,
            input_prompts,
            mp_on,
            speculative=True,
            stop_token_ids=None,
            predictor_override=_always_hit,
        )

    assert hit_state["calls"] > 0, (
        "predictor patch was never invoked; the speculative path did not run "
        "(check that enable_speculative_beam_history_d2h is honored)"
    )
    _assert_outputs_equal(out_on, out_off)


@pytest.mark.threadleak(enabled=False)
def test_speculative_d2h_skips_only_when_no_request_can_finish(
    fixed_params: dict[str, Any],
    input_prompts: list[list[int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The skip decision is per group, not per request.

    The snapshot is one batched copy for the whole group, so it cannot be
    skipped for some requests and taken for others. A step is skipped only when
    every request is predicted non-terminal; a single possible finisher makes
    the group pay for the copy it would have cost anyway.

    Pin the predictor to answer True for exactly one request and False for the
    rest. The mixed verdict must land on "copy", and the outputs must still
    match the feature-off run -- a per-request reading of the same verdict
    would drop the histories of the requests that answered False.
    """
    seen_verdicts: list[bool] = []
    first_slot: dict[str, int | None] = {"slot": None}

    def _one_finisher(self, request, *, num_generated_tokens, num_tokens):
        # Latch onto whichever request is seen first and only ever claim that
        # one might finish.
        if first_slot["slot"] is None:
            first_slot["slot"] = request.py_seq_slot
        verdict = request.py_seq_slot == first_slot["slot"]
        seen_verdicts.append(verdict)
        return verdict

    with monkeypatch.context() as mp_off:
        out_off = _run_with_env(
            fixed_params, input_prompts, mp_off, speculative=False, stop_token_ids=None
        )

    with monkeypatch.context() as mp_on:
        out_on = _run_with_env(
            fixed_params,
            input_prompts,
            mp_on,
            speculative=True,
            stop_token_ids=None,
            predictor_override=_one_finisher,
        )

    assert seen_verdicts, "predictor patch was never invoked; the speculative path did not run"
    assert any(seen_verdicts) and not all(seen_verdicts), (
        "the test needs a step where the verdicts disagree, so that the group "
        f"decision is observable; got {set(seen_verdicts)}"
    )
    _assert_outputs_equal(out_on, out_off)


# ---------------------------------------------------------------------------
# Validator: speculative path must be rejected when sampler_force_async_worker
# is also set, since the speculative path bypasses the async D2H worker.
# ---------------------------------------------------------------------------


@pytest.mark.threadleak(enabled=False)
def test_speculative_d2h_rejects_async_worker_combo(
    fixed_params: dict[str, Any],
    input_prompts: list[list[int]],
) -> None:
    """`TorchLlmArgs.validate_speculative_beam_history_d2h` must raise when
    both `enable_speculative_beam_history_d2h=True` and
    `sampler_force_async_worker=True` are passed.

    The speculative path bypasses `_copy_to_host`, which AsyncWorkerMixin
    relies on, so the combination is rejected at config validation time.
    """
    with pytest.raises(ValidationError, match="enable_speculative_beam_history_d2h"):
        _build_llm(
            fixed_params,
            input_prompts,
            enable_speculative_beam_history_d2h=True,
            sampler_force_async_worker=True,
        )


# ---------------------------------------------------------------------------
# No-sync invariant: predictor-hit path on the side stream must not sync.
# ---------------------------------------------------------------------------


@contextmanager
def _assert_no_cuda_sync_locally() -> Generator[None, None, None]:
    """Local stand-in for utils.util.assert_no_cuda_sync.

    That helper parks a @hostfunc on the stream to block it, and only lowers
    its cancel flag *after* the assertion. The hostfunc holds the GIL while it
    spins, so anything that blocks before the cancel deadlocks the step: the
    main thread waits on the stream, the stream waits on the hostfunc, and the
    hostfunc waits for a cancel that the main thread never reaches. On the
    speculative path -- side-stream copier plus the executor's own threads --
    that race is live, and CI hangs here until the 3600s pytest timeout.

    Detect the same thing without parking anything on the stream:
    set_sync_debug_mode("error") makes torch raise on a synchronizing call,
    which is the actual property under test. It misses syncs from non-torch
    kernels, which the stream-blocking variant would catch; nothing in
    sample_async/update_requests issues those today.
    """
    if int(os.environ.get("CUDA_LAUNCH_BLOCKING", 0)):
        yield None
        return
    previous_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield None
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)


@pytest.mark.threadleak(enabled=False)
def test_speculative_d2h_predictor_hit_is_sync_free(
    fixed_params: dict[str, Any],
    input_prompts: list[list[int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Speculative + always-hit path must not introduce host-device syncs
    inside `sample_async` or inside `update_requests` after the sampler
    event has been awaited.

    Mirrors the sync check used by
    `tests/unittest/_torch/sampler/test_beam_search.py::validate_outputs`,
    but pins the predictor to always-hit so every step routes through
    the side-stream copier and never reaches the `.cpu()` fallback.
    """
    _always_hit, hit_state = _pinned_predictor(True)

    sample_async_orig = TorchSampler.sample_async
    update_requests_orig = TorchSampler.update_requests
    hook_state = {"sample_async_called": False, "update_requests_called": False}

    def _sample_async_hook(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        hook_state["sample_async_called"] = True
        with _assert_no_cuda_sync_locally():
            return sample_async_orig(self, *args, **kwargs)

    def _update_requests_hook(self, state: SampleStateTorch, *args, **kwargs):  # type: ignore[no-untyped-def]
        hook_state["update_requests_called"] = True
        # Sampler event awaits all device work (incl. side-stream copies)
        # and is the one expected sync; do it outside the guard below.
        sampler_event = state.sampler_event
        if sampler_event:
            sampler_event.synchronize()
        with _assert_no_cuda_sync_locally():
            state.sampler_event = None
            try:
                return update_requests_orig(self, state, *args, **kwargs)
            finally:
                state.sampler_event = sampler_event

    _ = _run_with_env(
        fixed_params,
        input_prompts,
        monkeypatch,
        speculative=True,
        stop_token_ids=None,
        predictor_override=_always_hit,
        sampler_method_patches={
            "sample_async": _sample_async_hook,
            "update_requests": _update_requests_hook,
        },
    )

    assert hit_state["calls"] > 0, (
        "predictor patch was never invoked; the speculative path did not run "
        "(check that enable_speculative_beam_history_d2h is honored)"
    )
    assert hook_state["sample_async_called"], "sample_async hook was never invoked"
    assert hook_state["update_requests_called"], "update_requests hook was never invoked"


# ---------------------------------------------------------------------------
# Stream-ordering contracts of the two D2H paths
#
# Both tests open a deliberate race window with torch.cuda._sleep so that the
# host-side work they issue afterwards runs while the GPU copies are still
# pending. A correct implementation is unaffected by the window; a missing
# ordering edge turns into a deterministic wrong read instead of a flake.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_side_stream_copier_keeps_sources_alive_until_copied() -> None:
    numel = 4096
    side_stream = torch.cuda.Stream()
    copier = _SideStreamCopier(side_stream, torch.cuda.stream(side_stream))

    # Start from an empty cache so the block `src` releases is the only free
    # block of its size; an older same-sized block left by earlier tests
    # would otherwise absorb the clobber and hide a missing record_stream.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.arange(numel, device="cuda", dtype=torch.int32)
    index = torch.randperm(numel, device="cuda")
    expected = base.cpu()[index.cpu()]

    with torch.cuda.stream(side_stream):
        torch.cuda._sleep(_RACE_WINDOW_CYCLES)

    src = base[index]  # main-stream temporary, same shape as the clobber below
    dst = copier.stage_copy_to_host(src)
    event = copier.commit()
    assert event is not None
    del src

    # Nothing is queued on the main stream, so this runs at once. Without the
    # record_stream edge it lands in the block `src` just released.
    clobber = torch.full((numel,), -1, device="cuda", dtype=torch.int32)
    assert not event.query(), (
        "the side-stream copy finished before the clobber was issued; the race "
        "window was not open and the test proves nothing (raise _RACE_WINDOW_CYCLES)"
    )

    event.synchronize()
    torch.testing.assert_close(dst, expected)
    del clobber


def _recording_non_blocking_copy(src: torch.Tensor) -> torch.Tensor:
    """Stand-in for `AsyncWorkerMixin._copy_to_host` without the async worker.

    Same non-blocking copy into pinned memory, but the destination is
    zero-filled first so a read that outruns the copy observes zeros rather
    than whatever the pinned block last held.
    """
    dst = torch.zeros_like(src, device="cpu", pin_memory=True)
    dst.copy_(src, non_blocking=True)
    return dst


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_speculative_fallback_builder_awaits_its_copies() -> None:
    num_beams = 3
    num_generated = 4
    pad = BEAM_SEARCH_PAD_TOKEN

    sampling_params = SamplingParams(n=num_beams, best_of=num_beams, use_beam_search=True)
    request = LlmRequest(
        request_id=0,
        seq_slot=0,
        max_new_tokens=16,
        input_tokens=[1, 2, 3],
        end_id=-1,
        sampling_config=SamplingConfig(sampling_params._get_sampling_config()),
        is_streaming=False,
    )
    request.state = LlmRequestState.GENERATION_IN_PROGRESS
    request.py_decoding_iter = num_generated
    request.decoding_iter = num_generated
    request.py_seq_slot = 0
    prompt_len = request.py_prompt_len
    # num_generated_tokens is derived from the request's token count; the last
    # token of the step is not added yet, hence num_generated - 1.
    request.set_generated_tokens([[0] * (num_generated - 1)] * num_beams)
    total = prompt_len + num_generated

    store = BeamSearchStore.create(
        cache_indirection_shape=(1, num_beams, total),
        max_num_sequences=1,
        max_beam_width=num_beams,
    )
    cba = store.ensure_cba()
    # Every beam finished, so should_stop is True for the slot.
    store.first_finish_reasons.fill_(FinishReason.LENGTH.value)
    # Identity ancestry: each beam keeps its own tokens.
    store.cache_indirection.copy_(
        torch.arange(num_beams, dtype=torch.int32).view(-1, 1).expand(-1, total).unsqueeze(0)
    )
    active_tokens = torch.tensor(
        [[31, 32, 33, 34], [41, 42, 43, 44], [51, 52, 53, 54]], dtype=torch.int32
    )
    store.original_tokens.zero_()
    store.original_tokens[0, :, prompt_len:] = active_tokens.cuda()
    # Already in descending order, so the ranking leaves the beams in place.
    store.cum_log_probs[0] = torch.tensor([7.0, 5.0, 3.0], device="cuda")
    # Empty pool: every output beam comes from the live ones.
    cba.cba_tokens.fill_(pad)
    cba.cba_lengths.zero_()
    cba.cba_cum_log_probs.zero_()
    cba.cba_normed_scores.fill_(float("-inf"))

    def _no_side_stream_copier() -> AbstractContextManager[_SideStreamCopier]:
        raise AssertionError("the fallback builder must not use the side-stream copier")

    handler = BeamSearchHandler(
        store=store,
        max_seq_len=total,
        max_num_sequences=1,
        use_speculative_d2h=True,
        has_multi_token_stop_words=lambda _request: False,
        copy_to_host=_recording_non_blocking_copy,
        make_side_stream_copier=_no_side_stream_copier,
    )
    builder = handler._speculative_builder(request)

    # Everything the builder enqueues now lands behind this sleep.
    torch.cuda._sleep(_RACE_WINDOW_CYCLES)
    history = builder()

    assert history is not None, (
        "the builder read should_stop before its D2H copy landed and dropped the history"
    )
    torch.testing.assert_close(history.tokens, active_tokens)
    assert history.cum_logprobs is not None
    torch.testing.assert_close(history.cum_logprobs, torch.tensor([7.0, 5.0, 3.0]))


if __name__ == "__main__":
    pytest.main([__file__])
