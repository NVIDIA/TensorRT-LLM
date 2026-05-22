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
  `TorchSampler._prepare_beam_history._builder`
  (the synchronous `.cpu()` issued when the host-side predictor
  decided the step is non-terminal but the beam still finalizes),
* the predictor-hit path that routes copies through the side stream.

The dummy model from `test_beam_search_util` produces deterministic
outputs, so we can compare runs token-for-token without depending on
real model weights.
"""

import gc
import pathlib as _pl
from copy import deepcopy
from typing import Any, Iterable

import pytest
from pydantic import ValidationError
from test_beam_search_util import DummyConfigLoader, DummyWeightLoader
from utils.util import assert_no_cuda_sync

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.models.checkpoints import HfCheckpointLoader
from tensorrt_llm._torch.pyexecutor.sampler import SampleStateTorch, TorchSampler
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.llmapi import KvCacheConfig


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
        sampler_type="TorchSampler",
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
) -> list[GenerationResult]:
    """Build a fresh LLM with the speculative flag configured, run beam search, tear down.

    Opt-in is via `TorchLlmArgs.enable_speculative_beam_history_d2h`.
    `predictor_override` patches
    `TorchSampler._predict_beam_search_is_likely_finishing` and
    `sampler_method_patches` patches arbitrary `TorchSampler` methods;
    either forces `TLLM_WORKER_USE_SINGLE_PROCESS=1` so class-level patches
    reach the sampler. `sampler_force_async_worker` enables the
    AsyncWorkerMixin path.
    """
    needs_single_process = predictor_override is not None or sampler_method_patches
    if needs_single_process:
        # Class-level patches do not cross process boundaries; force the
        # sampler to run in-process so the patch is observed.
        monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    if predictor_override is not None:
        monkeypatch.setattr(
            TorchSampler, "_predict_beam_search_is_likely_finishing", predictor_override
        )
    if sampler_method_patches:
        for name, replacement in sampler_method_patches.items():
            monkeypatch.setattr(TorchSampler, name, replacement)

    gc.collect(2)
    llm = _build_llm(
        fixed_params,
        input_prompts,
        enable_speculative_beam_history_d2h=speculative,
        sampler_force_async_worker=sampler_force_async_worker,
    )
    try:
        with llm:
            return _generate(
                llm, input_prompts, _make_sampling_params(fixed_params, stop_token_ids)
            )
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


@pytest.mark.threadleak(enabled=False)
def test_speculative_d2h_predictor_hit_is_sync_free(
    fixed_params: dict[str, Any],
    input_prompts: list[list[int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Speculative + always-hit path must not introduce host-device syncs
    inside `sample_async` or inside `update_requests` after the sampler
    event has been awaited.

    Mirrors the `assert_no_cuda_sync()` hook used by
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
        with assert_no_cuda_sync():
            return sample_async_orig(self, *args, **kwargs)

    def _update_requests_hook(self, state: SampleStateTorch, *args, **kwargs):  # type: ignore[no-untyped-def]
        hook_state["update_requests_called"] = True
        # Sampler event awaits all device work (incl. side-stream copies)
        # and is the one expected sync; do it outside assert_no_cuda_sync.
        sampler_event = state.sampler_event
        if sampler_event:
            sampler_event.synchronize()
        with assert_no_cuda_sync():
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


if __name__ == "__main__":
    pytest.main([__file__])
