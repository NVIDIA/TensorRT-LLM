# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
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

import dataclasses
import functools
import gc
import os
import pathlib as _pl
import types
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from typing import Any, Callable, Generator, cast

import pytest
import torch
from test_beam_search_util import (BeamSearchTestOutput, DummyConfig,
                                   DummyConfigLoader, DummyWeightLoader,
                                   get_expected_outputs)
from utils.llm_data import llm_models_root
from utils.util import assert_no_cuda_sync, force_ampere, run_test_with_warmup

from tensorrt_llm import LLM, DisaggregatedParams, SamplingParams, TorchLlmArgs
from tensorrt_llm._torch.models.checkpoints import HfCheckpointLoader
from tensorrt_llm._torch.pyexecutor.llm_request import (LlmRequest,
                                                        LlmRequestState,
                                                        SamplingConfig)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.sampler import (BeamHistory,
                                                    SampleStateTorch,
                                                    TorchSampler)
from tensorrt_llm._torch.pyexecutor.sampler.beam_search import (
    CBAGroupHost, _gather_beam_path, _prepare_beam_history_cba, finalize_beam)
from tensorrt_llm._torch.pyexecutor.sampler.sampler_strategy import (
    BEAM_SEARCH_PAD_TOKEN, BeamSearch, BeamSearchEarlyStop, BeamSearchMetadata,
    CBAState, _StrategyImpls, beam_search_sampling_batch_cba)
from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.bindings.internal.batch_manager import \
    LlmRequest as CppLlmRequest
from tensorrt_llm.executor import RequestError
from tensorrt_llm.executor.result import CompletionOutput, GenerationResult
from tensorrt_llm.llmapi import (CacheTransceiverConfig, CudaGraphConfig,
                                 KvCacheConfig)


@pytest.fixture(scope="module")
def input_prompts():
    return [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


@pytest.fixture(scope="module")
def fixed_params():
    return {"max_tokens": 8, "max_beam_width": 2}


@pytest.fixture(scope="module")
def model_kwargs(fixed_params) -> dict[str, Any]:

    assert fixed_params[
        "max_beam_width"] == 2, "This test only works for a beam width of 2"
    return dict(
        model=_pl.Path("dummy_path"),
        checkpoint_loader=HfCheckpointLoader(
            weight_loader=DummyWeightLoader(),
            config_loader=DummyConfigLoader(),
        ),
    )


@pytest.fixture(scope="module",
                params=[False, True],
                ids=["no_cuda_graph_and_overlap", "cuda_graph_and_overlap"])
def with_cuda_graph_and_overlap(request):
    return request.param


def _build_llm(fixed_params, input_prompts, llm_kwargs: dict[str, Any]):
    llm_kwargs = llm_kwargs.copy()
    kv_cache_config = llm_kwargs.pop(
        "kv_cache_config",
        KvCacheConfig(
            max_tokens=10000,  # pyright: ignore
        ),
    )
    if "max_batch_size" not in llm_kwargs:
        llm_kwargs = llm_kwargs | dict(
            max_batch_size=fixed_params["max_beam_width"] * len(
                input_prompts
            ),  # use small batch size to prevent large buffers from possibly hiding wrong data accesses.
        )
    return LLM(
        **llm_kwargs,
        kv_cache_config=kv_cache_config,
        max_seq_len=32,
        max_beam_width=fixed_params["max_beam_width"],
    )


@contextmanager
def _single_process_context():
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    try:
        yield
    finally:
        del os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"]


# NB: It is important that all tests instantiating 'LLM' with
#     TLLM_WORKER_USE_SINGLE_PROCESS=1 (i.e., single_process=True below)
#     use this fixture. Otherwise, more than one such 'LLM' object
#     could be alive at any given point in time and this has been
#     found to result in corruption of the cache_indirection tensors.
@pytest.fixture(scope="module")
def llm(fixed_params, input_prompts, model_kwargs, single_process: bool,
        with_cuda_graph_and_overlap: bool):
    check_no_sync = single_process  # single_process only used for sync check

    gc.collect(
        2)  # force destruction of any other LLM instances (cf. comment above)
    with _single_process_context() if single_process else nullcontext():
        llm_kwargs: dict[
            str,
            Any] = deepcopy(  # LLM.shutdown resets checkpoint_loader.config_loader
                model_kwargs)
        if not with_cuda_graph_and_overlap:
            llm_kwargs |= dict(
                disable_overlap_scheduler=True,
                cuda_graph_config=None,
            )
        else:
            llm_kwargs |= dict(
                disable_overlap_scheduler=False,
                cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4, 8],
                                                  enable_padding=True),
            )
        llm = _build_llm(
            fixed_params,
            input_prompts,
            llm_kwargs=llm_kwargs,
        )
    with llm:
        yield llm


def check_generation_logits(beam: CompletionOutput,
                            sampling_params: SamplingParams,
                            valid_tokens: int | None) -> None:
    """Check if the generation logits have the correct shape"""
    if sampling_params.return_generation_logits:
        gen_logits = beam.generation_logits
        # Fall back to this beam's own length rather than max_tokens: under
        # early_stopping=1 (the default, HF's `True`) the request stops once
        # best_of candidates are complete, so a beam that never hit a stop
        # token can be shorter than max_tokens.
        assert beam.token_ids is not None
        generated_tokens = (valid_tokens if valid_tokens is not None else len(
            beam.token_ids))
        assert gen_logits is not None, "generation logits should not be None"
        assert gen_logits.ndim == 2, f"generation logits should have 2 dimensions, but got {gen_logits.ndim}"
        assert gen_logits.shape[
            0] == generated_tokens, f"expected {generated_tokens} generation logits, but got {gen_logits.shape[0]}"
    else:
        assert beam.generation_logits is None, "generation logits should be None"


def check_logprobs(beam: CompletionOutput, sampling_params: SamplingParams,
                   valid_tokens: int | None) -> None:
    """Check if the logprobs have the correct shape"""
    assert beam.logprobs is not None
    if sampling_params.logprobs is not None:
        # Fall back to this beam's own length rather than max_tokens: under
        # early_stopping=1 (the default, HF's `True`) the request stops once
        # best_of candidates are complete, so a beam that never hit a stop
        # token can be shorter than max_tokens.
        assert beam.token_ids is not None
        generated_tokens = (valid_tokens if valid_tokens is not None else len(
            beam.token_ids))
        assert len(
            beam.logprobs
        ) == generated_tokens, f"expected {generated_tokens} logprobs, but got {len(beam.logprobs)}"
        log_sum = 0.0
        for logprob_dict in (beam.logprobs):
            assert isinstance(logprob_dict, dict)
            for logprob_value in logprob_dict.values():
                log_sum += logprob_value.logprob
        assert log_sum == beam.cumulative_logprob, f"expected {beam.cumulative_logprob} logprob, but got {log_sum}"
    else:
        assert len(beam.logprobs) == 0, "logprobs should be empty"


def check_cache_indirection(beam: CompletionOutput,
                            sampling_params: SamplingParams,
                            reference_cache_indirection: torch.Tensor,
                            prompt_length: int, beam_idx: int,
                            valid_tokens: int | None) -> None:
    """Check if the cache indirection seen by the model is the same as the expected cache indirection"""
    assert beam.additional_generation_outputs is not None
    cache_indirection = beam.additional_generation_outputs["cache_indirection"]
    assert cache_indirection is not None, "cache indirection should not be None"
    assert cache_indirection.shape[
        1] == sampling_params.best_of, f"expected {sampling_params.best_of} entries in dim 1 of cache indirection, but got {cache_indirection.shape[1]}"

    # Fall back to what this beam actually produced rather than max_tokens: with
    # early_stopping=1 (the default, HF's `True`) the request stops as soon as
    # best_of finished candidates exist, so a beam that never hit a stop token
    # can still be shorter than max_tokens.
    assert beam.token_ids is not None
    num_generated_tokens = (valid_tokens if valid_tokens is not None else len(
        beam.token_ids))
    # We return the cache indirection before the sampling step, therefore cache indirection does not reflect changes during the sampling of the last token
    num_valid_cache_indirection = num_generated_tokens - 1

    # check if the cache indirection is correct for the given deterministic input prompt
    # Check only the last cache indirection
    last_cache_indirection = cache_indirection[num_valid_cache_indirection,
                                               beam_idx]

    assert all(last_cache_indirection[:prompt_length] ==
               0), "prompt tokens should have a cache indirection of 0"
    # remove the prompt tokens from the cache indirection and check if the remaining cache indirection is correct
    valid_cache_indirection = last_cache_indirection[
        prompt_length:prompt_length + num_valid_cache_indirection]
    assert all(
        valid_cache_indirection == reference_cache_indirection[
            beam_idx, :num_valid_cache_indirection]
    ), f"expected {reference_cache_indirection[beam_idx, :num_valid_cache_indirection].tolist()} cache indirection, but got {valid_cache_indirection.tolist()}"


def validate_output_beam(beam_output: CompletionOutput,
                         expected_outputs: BeamSearchTestOutput,
                         sampling_params: SamplingParams, prompt_length: int,
                         beam_idx: int) -> None:
    """Perform several checks on the output of a single beam"""

    valid_tokens = None
    if sampling_params.stop_token_ids is not None and sampling_params.stop_token_ids[
            0] in (expected_output_token_ids :=
                   expected_outputs.outputs[beam_idx].tolist()):
        assert beam_output.finish_reason == "stop"
        valid_tokens = expected_output_token_ids.index(
            sampling_params.stop_token_ids[0]) + 1
    else:
        assert beam_output.finish_reason == "length"

    check_generation_logits(beam_output, sampling_params, valid_tokens)
    check_logprobs(beam_output, sampling_params, valid_tokens)
    check_cache_indirection(beam_output, sampling_params,
                            expected_outputs.cache_indirection, prompt_length,
                            beam_idx, valid_tokens)
    # Check output similarity

    assert valid_tokens is None or valid_tokens > 0
    # get_expected_outputs walks a fixed number of iterations; it has no notion
    # of the finished-candidate pool. Under early_stopping=1 the request stops
    # once best_of candidates are complete, so a beam that never hit a stop
    # token can end early. Compare the prefix it did produce -- a wrong beam
    # still diverges, only the unreached tail is dropped.
    assert beam_output.token_ids is not None
    num_valid = valid_tokens if valid_tokens is not None else len(
        beam_output.token_ids)
    expected_valid_token_ids = expected_outputs.outputs[
        beam_idx, :num_valid].tolist()
    assert beam_output.token_ids == expected_valid_token_ids, f"expected {expected_valid_token_ids} token ids, but got {beam_output.token_ids}"


def check_context_logits(output: GenerationResult,
                         sampling_params: SamplingParams):
    """Check if the context logits have the correct shape"""
    if sampling_params.return_context_logits:
        assert output.context_logits is not None, "context logits should not be None"
        assert len(output.prompt_token_ids) == output.context_logits.shape[
            0], f"expected {len(output.prompt_token_ids)} context logits, but got {output.context_logits.shape[0]}"
    else:
        assert output.context_logits is None, "context logits should be None"


@pytest.fixture(scope="module",
                params=[False, True],
                ids=["multi_process", "single_process"])
def single_process(request) -> bool:
    return cast(bool, request.param)


def validate_output(output: GenerationResult, input_prompt: list[int],
                    sampling_params: SamplingParams) -> None:
    """Perform several checks on the output of a single prompt"""
    check_context_logits(output, sampling_params)

    # validate number of outputs equals beam width
    num_output_beams = sampling_params.n
    assert len(
        output.outputs
    ) == num_output_beams, f"expected {num_output_beams} outputs, but got {len(output.outputs)}"
    # check each beam
    expected_outputs = get_expected_outputs(
        input_prompt[-1], num_iterations=sampling_params.max_tokens)

    for beam_idx, beam_output in enumerate(output.outputs):
        validate_output_beam(beam_output, expected_outputs, sampling_params,
                             len(input_prompt), beam_idx)


def validate_outputs(llm: LLM, input_prompts: list[list[int]],
                     sampling_params: SamplingParams,
                     monkeypatch: pytest.MonkeyPatch,
                     check_no_sync: bool) -> None:
    """Generate outputs for a list of prompts and validate the outputs"""

    outputs = llm.generate(deepcopy(input_prompts),
                           sampling_params=deepcopy(sampling_params))

    if check_no_sync:
        del outputs  # treat previous .generate as warmup, ignore results

        with monkeypatch.context() as patcher:
            sample_async_orig = TorchSampler.sample_async
            update_requests_orig = TorchSampler.update_requests

            _sample_async_hook_called = False
            _update_requests_hook_called = False

            def _sample_async_hook(*args, **kwargs):
                nonlocal _sample_async_hook_called
                _sample_async_hook_called = True

                with assert_no_cuda_sync():
                    return sample_async_orig(*args, **kwargs)

            def _update_requests_hook(self, state: SampleStateTorch, *args,
                                      **kwargs):
                nonlocal _update_requests_hook_called
                _update_requests_hook_called = True

                # await sampling event outside sync-check (because this syncs)
                sampler_event = state.sampler_event
                if sampler_event:
                    sampler_event.synchronize()

                with assert_no_cuda_sync():
                    state.sampler_event = None
                    try:
                        return update_requests_orig(self, state, *args,
                                                    **kwargs)
                    finally:
                        state.sampler_event = sampler_event

            # Intercept sampler methods to check that they do not sync (requires
            # TLLM_WORKER_USE_SINGLE_PROCESS).
            patcher.setattr(TorchSampler, "sample_async", _sample_async_hook)
            patcher.setattr(TorchSampler, "update_requests",
                            _update_requests_hook)

            outputs = llm.generate(deepcopy(input_prompts),
                                   sampling_params=deepcopy(sampling_params))

            assert _sample_async_hook_called
            assert _update_requests_hook_called

    num_prompts = len(input_prompts)
    assert isinstance(outputs, list)
    assert len(
        outputs
    ) == num_prompts, f"expected {num_prompts} outputs, but got {len(outputs)}"
    for output_idx, output in enumerate(outputs):
        validate_output(output, input_prompts[output_idx], sampling_params)


@pytest.mark.parametrize("return_log_probs", [True, False])
@pytest.mark.parametrize("gather_generation_logits", [True, False])
@pytest.mark.parametrize("gather_context_logits", [True, False])
@pytest.mark.parametrize("num_output_beams", [1, 2])
@pytest.mark.parametrize("num_prompts", [1, 3])
@pytest.mark.parametrize("stop_token_ids", [[15], None])
@pytest.mark.threadleak(enabled=False)
def test_beam_search_e2e(
    gather_context_logits: bool,
    gather_generation_logits: bool,
    return_log_probs: bool,
    num_output_beams: int,
    num_prompts: int,
    stop_token_ids: list[int] | None,
    llm: LLM,
    fixed_params,
    input_prompts,
    single_process: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm_args = cast(TorchLlmArgs, llm.args)  # type: ignore[redundant-cast]

    # create sampling parameters
    # additional_model_outputs is used to gather the cache indirection from the model.
    sampling_params = SamplingParams(
        max_tokens=fixed_params["max_tokens"],
        n=num_output_beams,
        best_of=fixed_params["max_beam_width"],
        use_beam_search=True,
        return_context_logits=gather_context_logits,
        return_generation_logits=gather_generation_logits,
        logprobs=return_log_probs,
        end_id=-1,
        stop_token_ids=stop_token_ids,
        include_stop_str_in_output=True,
        additional_model_outputs=["cache_indirection"],
    )
    validate_outputs(
        llm,
        input_prompts[:num_prompts],
        sampling_params,
        check_no_sync=single_process,
        monkeypatch=monkeypatch,
    )


@pytest.mark.threadleak(enabled=False)
def test_beam_search_disagg_e2e(
    fixed_params,
    input_prompts,
    model_kwargs: dict[str, Any],
) -> None:
    """Beam search is admitted under disaggregated serving.

    The context server's finished-candidate pool is not part of the handoff
    (TRTLLM-14792), so the CBA op runs with that side's end id masked: an end
    candidate stays in its beam slot rather than being pooled, travels as
    first_gen_tokens, and the generation server pools it there instead. Every
    early_stopping mode goes through the same route, so admission accepts them
    all rather than rejecting beam search outright.
    """
    sampling_params = SamplingParams(
        max_tokens=fixed_params["max_tokens"],
        n=fixed_params["max_beam_width"],
        best_of=fixed_params["max_beam_width"],
        use_beam_search=True,
        end_id=-1,
        include_stop_str_in_output=True,
    )

    disagg_kwargs = deepcopy(model_kwargs)
    disagg_kwargs |= dict(
        disable_overlap_scheduler=True,
        cuda_graph_config=None,
        kv_cache_config=KvCacheConfig(max_tokens=10000,
                                      enable_block_reuse=True,
                                      enable_partial_reuse=True,
                                      use_kv_cache_manager_v2=True),
        cache_transceiver_config=CacheTransceiverConfig(
            backend="NIXL",
            transceiver_runtime="PYTHON",
            kv_transfer_timeout_ms=1000,
            kv_transfer_sender_future_timeout_ms=1000,
        ),
    )

    prompts = [[1, 2, 3]]
    ctx_llm = _build_llm(fixed_params, prompts, disagg_kwargs)
    try:
        with ctx_llm:
            # Every mode goes through the same route, including the default
            # (early_stopping unset, i.e. True).
            for early_stopping in (None, 0, 1, 2):
                params = deepcopy(sampling_params)
                if early_stopping is not None:
                    params.early_stopping = early_stopping
                outputs = ctx_llm.generate(
                    deepcopy(prompts),
                    sampling_params=params,
                    disaggregated_params=[
                        DisaggregatedParams(request_type="context_only",
                                            disagg_request_id=200)
                    ],
                    use_tqdm=False,
                )
                # The context phase hands off one token per beam; admission no
                # longer rejects it, which is what this pins.
                assert len(outputs) == len(prompts)
                ctx_params = outputs[0].disaggregated_params
                assert ctx_params is not None
                assert len(ctx_params.first_gen_tokens
                           ) == fixed_params["max_beam_width"]
    finally:
        ctx_llm.shutdown()


@pytest.mark.threadleak(enabled=False)
def test_beam_search_disagg_first_token_is_end_id(
    fixed_params,
    input_prompts,
    model_kwargs: dict[str, Any],
) -> None:
    """A context phase that finishes on its only token still hands off.

    This is the case the handoff is built around, and nothing else reaches it:
    an end candidate would normally be pooled and its beam slot refilled, and
    the request would be marked finished before it reaches the
    disagg-transmission state that builds ContextPhaseParams -- so the
    generation server would get no first_gen_tokens at all and never learn the
    beam finished. Context-only requests therefore run with their end id
    masked to the "no end token" sentinel, which keeps the token in its slot
    and keeps the request on the handoff path.

    Triggering it by prompt would be a lottery, and end_id cannot simply be
    left unset: these prompts are token ids and the LLM has no tokenizer, so
    an unset end_id raises. The end id is picked after the fact instead -- run
    once to see what the beams sample, then declare beam 0's token the end id
    and rerun, so the context step finishes on its first and only token.
    """
    beam_width = fixed_params["max_beam_width"]
    base_params = SamplingParams(
        max_tokens=fixed_params["max_tokens"],
        n=beam_width,
        best_of=beam_width,
        use_beam_search=True,
        end_id=-1,
        include_stop_str_in_output=True,
    )

    disagg_kwargs = deepcopy(model_kwargs)
    disagg_kwargs |= dict(
        disable_overlap_scheduler=True,
        cuda_graph_config=None,
        kv_cache_config=KvCacheConfig(max_tokens=10000,
                                      enable_block_reuse=True,
                                      enable_partial_reuse=True,
                                      use_kv_cache_manager_v2=True),
        cache_transceiver_config=CacheTransceiverConfig(
            backend="NIXL",
            transceiver_runtime="PYTHON",
            kv_transfer_timeout_ms=1000,
            kv_transfer_sender_future_timeout_ms=1000,
        ),
    )

    prompts = [[1, 2, 3]]

    def _context_first_gen_tokens(llm, params: SamplingParams) -> list[int]:
        outputs = llm.generate(
            deepcopy(prompts),
            sampling_params=deepcopy(params),
            disaggregated_params=[
                DisaggregatedParams(request_type="context_only",
                                    disagg_request_id=201)
            ],
            use_tqdm=False,
        )
        assert len(outputs) == len(prompts)
        ctx_params = outputs[0].disaggregated_params
        assert ctx_params is not None
        # None here means the context phase never reached the transmission
        # state, so ContextPhaseParams was never built -- the handoff, tokens
        # included, was dropped.
        assert ctx_params.first_gen_tokens is not None, (
            "context phase produced no first_gen_tokens")
        return list(ctx_params.first_gen_tokens)

    ctx_llm = _build_llm(fixed_params, prompts, disagg_kwargs)
    try:
        with ctx_llm:
            # Probe: no end id, so nothing can finish.
            baseline_tokens = _context_first_gen_tokens(ctx_llm, base_params)
            assert len(baseline_tokens) == beam_width

            end_id = baseline_tokens[0]
            end_id_params = deepcopy(base_params)
            end_id_params.end_id = end_id

            end_id_tokens = _context_first_gen_tokens(ctx_llm, end_id_params)

            # Beam 0's token is the end id and it is still in the handoff.
            # Were it pooled and the slot refilled, some other candidate would
            # be here instead; were the request marked finished, there would be
            # no handoff to read at all.
            assert len(end_id_tokens) == beam_width
            assert end_id_tokens[0] == end_id
            # Masking is per request, so the other beams are unaffected.
            assert end_id_tokens == baseline_tokens
    finally:
        ctx_llm.shutdown()


@pytest.mark.parametrize("beam_width", [10])
@pytest.mark.threadleak(enabled=False)
def test_beam_search_large_beam_width_regression(
    beam_width: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Fix https://nvbugs/6242591
    """
    num_prompts = 2
    input_prompts = [[1, 2, 3], [4, 5, 6]]
    vocab_size = DummyConfig().vocab_size
    # The dummy model pushes each beam towards (token + 3) mod vocab_size, so an
    # end id a few multiples of 3 ahead of the prompt is reachable and different
    # beams hit it at different steps -> exercises the finished-slot reuse path.
    end_id = 24

    checkpoint_loader = HfCheckpointLoader(
        weight_loader=DummyWeightLoader(),
        config_loader=DummyConfigLoader(),
    )

    gc.collect(2)  # force destruction of any other LLM instances
    with _single_process_context():
        llm = LLM(
            model=_pl.Path("dummy_path"),
            checkpoint_loader=checkpoint_loader,
            max_beam_width=beam_width,
            max_batch_size=beam_width * num_prompts,
            max_seq_len=64,
            kv_cache_config=KvCacheConfig(max_tokens=10000),
            disable_overlap_scheduler=True,
            cuda_graph_config=None,
        )
        with llm:
            sampling_params = SamplingParams(
                max_tokens=16,
                n=beam_width,
                best_of=beam_width,
                use_beam_search=True,
                beam_search_diversity_rate=0.5,
                early_stopping=1,
                length_penalty=0.5,
                end_id=end_id,
            )
            outputs = llm.generate(deepcopy(input_prompts),
                                   sampling_params=deepcopy(sampling_params))

    assert isinstance(outputs, list)
    assert len(outputs) == num_prompts
    for output in outputs:
        beams = output.outputs
        assert len(beams) == beam_width, (
            f"expected {beam_width} beams, but got {len(beams)}")
        beam_sequences = []
        for beam_idx, beam in enumerate(beams):
            token_ids = beam.token_ids
            assert token_ids is not None, f"beam {beam_idx} has no token_ids"
            assert len(token_ids) > 0, f"beam {beam_idx} is empty"
            assert all(0 <= t < vocab_size for t in token_ids), (
                f"beam {beam_idx} has out-of-vocab tokens: {token_ids}")
            beam_sequences.append(tuple(token_ids))
        # Beams must not all collapse to the same sequence: corrupted state
        # produced duplicated / garbled beams. With a non-zero diversity rate
        # the beams are expected to differ.
        assert len(set(beam_sequences)) > 1, (
            f"all {beam_width} beams are identical: {beam_sequences[0]}")


@pytest.mark.parametrize(
    "beam_width_array, extra_params",
    [
        # Widening on its own.
        ([2, 3, 4], {}),
        # A constant array is the control: it exercises the VBWS plumbing
        # without ever changing width, so a failure here points at the
        # mechanism rather than at the transition.
        ([4, 4, 4], {}),
        # length_penalty normalizes by a per-beam generated length, which has
        # to follow the beams across a width change.
        ([2, 3, 4], {
            "length_penalty": 1.0
        }),
        # diversity_rate keys on the source beam's rank among the step's input
        # beams, which the width change renumbers.
        ([2, 3, 4], {
            "beam_search_diversity_rate": 0.5
        }),
        # The exhaustive modes size their candidate pool from both widths.
        ([2, 3, 4], {
            "early_stopping": 0
        }),
        ([2, 3, 4], {
            "early_stopping": 2
        }),
    ],
    ids=[
        "widening", "constant", "length_penalty", "diversity", "es_false",
        "es_never"
    ],
)
@pytest.mark.threadleak(enabled=False)
def test_beam_search_vbws_e2e(beam_width_array: list[int],
                              extra_params: dict[str, Any],
                              monkeypatch: pytest.MonkeyPatch) -> None:
    """Variable-Beam-Width-Search through the full engine path.

    Drives beam_width_array end to end (scheduler -> ModelEngine ->
    TorchSampler), which the operator-level tests cannot cover: the width
    comes from get_beam_width_by_iter(), and that only advances with the
    real decoding loop.

    NB: single request on purpose. ModelEngine requires every generation
    request in a batch to report the same per-iteration beam width, and
    beam_width_array is indexed by each request's own decoding_iter, so
    several requests admitted at different times would desynchronize and
    abort the batch (TRTLLM-14792).
    """
    max_beam_width = 4
    input_prompts = [[1, 2, 3]]
    vocab_size = DummyConfig().vocab_size
    # Decode well past the end of beam_width_array, so the width has to hold at
    # its last entry for most of the run: that clamp is the interesting part,
    # since getting it wrong reads past the array.
    #
    # NB: running beyond the array used to hang here. Both getBeamWidthByIter
    # implementations clamp the iteration index, but the C++ one used the
    # global kMaxBeamWidthArrayLength instead of the array's own length and
    # returned an out-of-bounds width once decoding outran the array. The C++
    # micro-batch scheduler reads that width, so the request was never admitted
    # into the generation batch again and decoding stalled forever. Exercising
    # the clamp therefore requires the C++ fix in llmRequest.cpp -- against an
    # older libtensorrt_llm.so this test hangs rather than fails.
    max_tokens = 12

    checkpoint_loader = HfCheckpointLoader(
        weight_loader=DummyWeightLoader(),
        config_loader=DummyConfigLoader(),
    )

    # Record the width the engine actually uses at each decoding iteration.
    # Asserting only on the outputs cannot distinguish a request that really
    # walked [2, 3, 4] from one that ran at a constant width the whole time,
    # so wrap the accessor every consumer (scheduler, ModelEngine, sampler)
    # goes through and keep the current-iteration widths it hands out.
    observed_widths: dict[int, int] = {}
    unwrapped_get_beam_width_by_iter = LlmRequest.get_beam_width_by_iter

    def recording_get_beam_width_by_iter(self: LlmRequest,
                                         for_next_iteration: bool = False
                                         ) -> int:
        width = unwrapped_get_beam_width_by_iter(self, for_next_iteration)
        if not for_next_iteration:
            # decoding_iter is 1-based once decoding starts; several callers
            # ask per step, so keep the first answer for each iteration.
            observed_widths.setdefault(self.decoding_iter, width)
        return width

    monkeypatch.setattr(LlmRequest, "get_beam_width_by_iter",
                        recording_get_beam_width_by_iter)

    # Inspect the requests right after every update_requests(): the sampling
    # op pads its output row out to the store's beam width, and appending
    # those columns would put BEAM_SEARCH_PAD_TOKEN into the request's token
    # history. Checking only the final outputs cannot see it, because
    # finalization rewrites every beam from the corrected paths -- but the
    # padded history is visible to streaming consumers and to anything reading
    # get_tokens() mid-flight.
    padded_histories: list[tuple[int, int, list[int]]] = []
    unwrapped_update_requests = TorchSampler.update_requests

    def recording_update_requests(self: TorchSampler, state, *args, **kwargs):
        result = unwrapped_update_requests(self, state, *args, **kwargs)
        for req in state.requests:
            if req.py_beam_width <= 1:
                continue
            for beam_idx in range(req.py_beam_width):
                tokens = list(req.get_tokens(beam_idx))
                if BEAM_SEARCH_PAD_TOKEN in tokens:
                    padded_histories.append(
                        (req.py_decoding_iter, beam_idx, tokens))
        return result

    monkeypatch.setattr(TorchSampler, "update_requests",
                        recording_update_requests)

    gc.collect(2)  # force destruction of any other LLM instances
    with _single_process_context():
        llm = LLM(
            model=_pl.Path("dummy_path"),
            checkpoint_loader=checkpoint_loader,
            max_beam_width=max_beam_width,
            max_batch_size=max_beam_width,
            max_seq_len=64,
            kv_cache_config=KvCacheConfig(max_tokens=10000),
            disable_overlap_scheduler=True,
            cuda_graph_config=None,
        )
        with llm:
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                n=max_beam_width,
                best_of=max_beam_width,
                use_beam_search=True,
                beam_width_array=beam_width_array,
                **extra_params,
                end_id=-1,
            )
            outputs = llm.generate(deepcopy(input_prompts),
                                   sampling_params=deepcopy(sampling_params))

    assert isinstance(outputs, list)
    assert len(outputs) == len(input_prompts)
    beams = outputs[0].outputs
    assert len(beams) == max_beam_width, (
        f"expected {max_beam_width} beams, but got {len(beams)}")
    for beam_idx, beam in enumerate(beams):
        token_ids = beam.token_ids
        assert token_ids is not None, f"beam {beam_idx} has no token_ids"
        assert len(token_ids) > 0, f"beam {beam_idx} is empty"
        assert all(0 <= t < vocab_size for t in token_ids), (
            f"beam {beam_idx} has out-of-vocab tokens: {token_ids}")

    # The width must actually vary along beam_width_array and then hold at its
    # last entry, rather than staying constant: decoding iteration i (1-based)
    # uses beam_width_array[i - 1], clamped to the final entry once decoding
    # outruns the array.
    decoding_iters = sorted(it for it in observed_widths if it >= 1)
    assert decoding_iters, (
        f"no decoding iterations recorded: {observed_widths}")
    actual = [observed_widths[it] for it in decoding_iters]
    expected = [
        beam_width_array[min(it, len(beam_width_array)) - 1]
        for it in decoding_iters
    ]
    assert actual == expected, (
        f"beam width per decoding iteration {decoding_iters} was {actual}, "
        f"expected {expected} from beam_width_array={beam_width_array}")
    # Guard against the whole run happening at a single width, which the
    # per-iteration comparison above would still accept if the engine only
    # ever reported one iteration.
    assert set(actual) == set(beam_width_array), (
        f"expected every width in {beam_width_array} to be exercised, "
        f"but only saw {sorted(set(actual))}")

    # No step may leave the padding sentinel in a request's token history.
    assert not padded_histories, (
        "BEAM_SEARCH_PAD_TOKEN leaked into the request token history; "
        "update_requests() must append only the beams the step produced, not "
        f"the full store width. First offenders: {padded_histories[:3]}")


###########################################################################
# Unit tests
###########################################################################


@pytest.fixture(autouse=True)
def _raise_dynamo_recompile_limit(monkeypatch):
    """Keep the whole module clear of Dynamo's per-code-object recompile cap.

    Nearly every test here builds a fresh engine or a fresh set of CBA tensors
    in this same process, and the CBA step is compiled with fullgraph=True.
    Dynamo counts recompiles per code object across the entire process, so the
    default cap is exhausted partway through the file and every later case
    fails to compile -- a hard failure under fullgraph, not a fallback. Each
    of those tests passes when run alone.

    The cap guards against runaway recompilation; it is not a correctness
    property, so raising it for the module is safe.
    """
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit",
                        max(256, torch._dynamo.config.recompile_limit))


def _kernel_test(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Mark a beam-search kernel unit test as CUDA-only and run its body with
    the default device set to CUDA, so the tensors it builds land on GPU (the
    kernels dispatch top-k through flashinfer's CUDA kernel on wide rows)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with torch.device("cuda"):
            return fn(*args, **kwargs)

    # functools.wraps is untyped, so `wrapper` decays to Any; name the type
    # again on the way out to satisfy the declared return.
    typed_wrapper: Callable[..., Any] = wrapper
    return typed_wrapper


class GeneralTestParams:
    # Test Parameters for the update_beam_history and finish_beams tests
    beam_width = 3
    max_beam_width = 4
    max_batch_size = 5
    max_seq_len = 123
    input_tokens = [20, 21, 22, 23, 24]
    prompt_len = len(input_tokens)
    num_generated_tokens = 5
    seq_len = prompt_len + num_generated_tokens
    num_logprobs = 0
    seq_slot = 4
    end_id = 99
    batch_size = 2
    vocab_size = 100


_CBA_NEG_INF = float("-inf")


def _with_cba(meta: BeamSearchMetadata,
              *,
              max_batch_size: int,
              beam_width: int,
              width: int,
              end_id: int = -1,
              prompt_len: int = 0) -> BeamSearchMetadata:
    """Attach a neutral candidate-beams-array state to a metadata object.

    Every beam-search step runs on ``beam_search_sampling_batch_cba``, which
    needs the pool. Tests that exercise the step's ranking want a pool that
    starts empty and never wins: all-``-inf`` normalized scores mean no pool
    entry can outrank a live beam, so what is observed is the selection the
    live beams produce.
    """
    return dataclasses.replace(meta,
                               cba=CBAState(
                                   end_ids=torch.full((max_batch_size, ),
                                                      end_id,
                                                      dtype=torch.int32),
                                   prompt_lens=torch.full((max_batch_size, ),
                                                          prompt_len,
                                                          dtype=torch.int32),
                                   original_tokens=torch.zeros(
                                       (max_batch_size, beam_width, width),
                                       dtype=torch.int32),
                                   cba_tokens=torch.full(
                                       (max_batch_size, beam_width, width),
                                       BEAM_SEARCH_PAD_TOKEN,
                                       dtype=torch.int32),
                                   cba_cum_log_probs=torch.zeros(
                                       (max_batch_size, beam_width),
                                       dtype=torch.float32),
                                   cba_normed_scores=torch.full(
                                       (max_batch_size, beam_width),
                                       _CBA_NEG_INF,
                                       dtype=torch.float32),
                                   cba_lengths=torch.zeros(
                                       (max_batch_size, beam_width),
                                       dtype=torch.int32),
                                   batch_dones=torch.zeros((max_batch_size, ),
                                                           dtype=torch.bool),
                                   cba_caps=torch.full((max_batch_size, ),
                                                       beam_width,
                                                       dtype=torch.int32),
                                   original_log_probs=torch.zeros(
                                       (max_batch_size, beam_width, width),
                                       dtype=torch.float32),
                                   cba_log_probs=torch.zeros(
                                       (max_batch_size, beam_width, width),
                                       dtype=torch.float32),
                                   max_seq_len=width,
                               ))


@_kernel_test
@pytest.mark.parametrize("params_as_tensors", [False, True])
@pytest.mark.parametrize(
    "length_penalty,diversity_rate",
    [(0.5, 0.0), (2.0, 2.0), (0.0, 0.5)],
)
def test_beam_candidate_topk_equivalence(length_penalty, diversity_rate,
                                         params_as_tensors):
    """The two-stage op must match naive full-matrix adjustment + flat topk
    for length penalty, diversity, and their combination."""
    from tensorrt_llm._torch.pyexecutor.sampler.beam_search import \
        beam_candidate_topk

    torch.manual_seed(7)
    batch_size, beam_width_in, beam_width_out, vocab_size = 5, 4, 4, 1000
    logprobs = -torch.rand((batch_size, beam_width_in, vocab_size)) * 20.0
    # emulate a finished beam row: -inf everywhere except a frozen entry
    logprobs[0, 1, :] = float("-inf")
    logprobs[0, 1, 0] = -1.5
    cand_gen_lengths = torch.randint(1,
                                     30, (batch_size, beam_width_in),
                                     dtype=torch.int32)

    def as_param(value):
        if value == 0.0:
            return None
        if params_as_tensors:
            return torch.full((batch_size, ), value)
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
            beam_width_in, dtype=logprobs.dtype).view(1, -1, 1)
    if length_penalty:
        factor = cand_gen_lengths.float().pow(length_penalty)
        adjusted = adjusted / factor.unsqueeze(-1)
    _, ref_indices = torch.topk(adjusted.view(batch_size, -1),
                                k=beam_width_out,
                                sorted=True,
                                dim=-1)
    ref_logprobs = logprobs.view(batch_size, -1).gather(1, ref_indices)

    torch.testing.assert_close(sorted_logprobs, ref_logprobs)
    torch.testing.assert_close(predecessor_beams,
                               (ref_indices // vocab_size).to(torch.int32))
    torch.testing.assert_close(tokens,
                               (ref_indices % vocab_size).to(torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_beam_topk_flashinfer_parity():
    """_beam_topk's flashinfer path (rows wider than the 10k crossover) must
    match torch.topk on beam-search-shaped inputs, including -inf-dominated rows
    (finished beams)."""
    from tensorrt_llm._torch.pyexecutor.sampler.beam_search import _beam_topk

    torch.manual_seed(11)
    bs, bw, vocab, k = 8, 4, 152064, 4
    logprobs = -torch.rand((bs * bw, vocab), device="cuda") * 20.0
    # finished-beam rows: all -inf except a single frozen entry at index 0
    logprobs[3, :] = float("-inf")
    logprobs[3, 0] = -1.5
    logprobs[17, :] = float("-inf")
    logprobs[17, 0] = -0.25

    ref_v, ref_i = torch.topk(logprobs, k, dim=-1, sorted=True)
    fi_v, fi_i = _beam_topk(logprobs, k)  # vocab > 10k -> flashinfer path
    assert fi_i.dtype == torch.int64
    torch.testing.assert_close(fi_v, ref_v)
    # indices may differ only where values are -inf ties; finite entries must match
    finite = torch.isfinite(ref_v)
    assert torch.equal(fi_i[finite], ref_i[finite])


@_kernel_test
def test_beam_search_sampling_batch_diversity_rate():
    """diversity_rate wired through beam_search_sampling_batch changes beam
    selection while stored cum_log_probs stay raw."""

    batch_size = 1
    beam_width = 2
    vocab_size = 8
    max_batch_size = 2
    seq_len = 6

    seq_slots = torch.arange(batch_size, dtype=torch.int64)
    slot = seq_slots[0]

    def make_metadata() -> BeamSearchMetadata:
        return BeamSearchMetadata(
            cache_indirection=torch.zeros(
                (max_batch_size, beam_width, seq_len + 1), dtype=torch.int32),
            cache_indirection_buffer=torch.full(
                (max_batch_size, beam_width, seq_len + 1),
                -1,
                dtype=torch.int32),
            cum_log_probs=torch.zeros((max_batch_size, beam_width),
                                      dtype=torch.float32),
            seq_slots=seq_slots,
            seq_lens=torch.full((batch_size, ), seq_len, dtype=torch.int32),
            finished_beams=torch.zeros((max_batch_size, beam_width),
                                       dtype=torch.int32),
            pending_harvest=torch.zeros((max_batch_size, beam_width),
                                        dtype=torch.bool),
            new_log_probs=torch.zeros((max_batch_size, beam_width),
                                      dtype=torch.float32),
            predecessor_beams=torch.zeros((max_batch_size, beam_width),
                                          dtype=torch.int32),
            beam_idx_arange=torch.arange(beam_width, dtype=torch.int32),
        )

    logits = torch.full((batch_size * beam_width, vocab_size), -20.0)
    logits[0, 1] = 10.0  # beam0 top: token1 (logprob ~0)
    logits[0, 2] = 8.0  # beam0 second: token2 (logprob ~ -2)
    logits[1, 3] = 30.0  # beam1 top: token3, but cum handicap below

    def run(diversity_rate):
        metadata = _with_cba(make_metadata(),
                             max_batch_size=max_batch_size,
                             beam_width=beam_width,
                             width=seq_len + 1)
        metadata.cum_log_probs[slot] = torch.tensor([0.0, -2.5])
        tokens, _ = beam_search_sampling_batch_cba(
            logits=logits,
            beam_width_in=beam_width,
            beam_width_out=beam_width,
            beam_search_args=metadata,
            temperature=1.0,
            early_stopping=BeamSearchEarlyStop.TRUE,
            diversity_rate=diversity_rate,
            return_probs=False,
        )
        return tokens, metadata

    # Without the adjustment both winners come from beam 0, whose candidates
    # (0.0 and ~-2.0) outrank beam 1's ~-2.5.
    tokens, meta = run(0.0)
    assert meta.predecessor_beams[slot].tolist() == [0, 0]

    # rate=1.0 adds rate * source_beam_index, lifting beam 1's candidate to
    # ~-1.5 and past beam 0's second one. Assert the effect -- that beam 1 now
    # contributes -- rather than the exact winning tokens, which also depend on
    # how the candidate pool is sized.
    tokens_div, meta_div = run(1.0)
    assert 1 in meta_div.predecessor_beams[slot].tolist(), (
        "diversity_rate should let the weaker source beam win a slot")
    assert tokens_div[0].tolist() != tokens[0].tolist(), (
        "diversity_rate should change the selected tokens")

    # Stored cum_log_probs are the raw scores, not diversity-adjusted: the
    # winner descending from beam 1 keeps its raw ~-2.5 (with the +1.0
    # adjustment it would be ~-1.5).
    expected_b0 = torch.log_softmax(logits[0], dim=-1)[1]
    torch.testing.assert_close(meta_div.cum_log_probs[slot, 0], expected_b0)
    torch.testing.assert_close(meta_div.cum_log_probs[slot, 1],
                               torch.tensor(-2.5),
                               atol=1e-3,
                               rtol=1e-3)


# Poison value for buffers the op is expected to overwrite wherever it writes
# at all. Zero is a plausible real value for most of them (a token id, a
# length, a beam index), so a zero-filled buffer cannot distinguish "written
# correctly" from "never written"; these can.
_UNWRITTEN_INT = -7
_UNWRITTEN_FLOAT = -7.0


def _assert_untouched_rows_intact(meta, before, batch, max_batch):
    """Rows outside seq_slots must come back bit-identical.

    The ops index every buffer with seq_slots, so a slot-agnostic write (a
    forgotten index, a broadcast over dim 0) still produces correct output for
    the slots under test and is invisible unless the unused rows are checked.
    """
    if batch >= max_batch:
        return
    unused = slice(batch, max_batch)
    for name, orig in before.items():
        current = _cba_field(meta, name)
        assert torch.equal(current[unused], orig[unused]), (
            f"{name} rows [{batch}:{max_batch}] were modified; the op must "
            "only write the rows selected by seq_slots")


def _cba_field(meta, name):
    return getattr(meta.cba, name) if hasattr(meta.cba, name) else getattr(
        meta, name)


def _snapshot_cba_rows(meta, names):
    return {name: _cba_field(meta, name).clone() for name in names}


def _make_cba_metadata(max_batch, K, attn_len, snap_len, seq_len, prompt_len,
                       end_id, batch):
    slots = torch.arange(batch, dtype=torch.int64)
    m = BeamSearchMetadata(
        cache_indirection=torch.full((max_batch, K, attn_len),
                                     _UNWRITTEN_INT,
                                     dtype=torch.int32),
        cache_indirection_buffer=torch.full((max_batch, K, attn_len),
                                            -1,
                                            dtype=torch.int32),
        cum_log_probs=torch.zeros((max_batch, K), dtype=torch.float32),
        new_log_probs=torch.full((max_batch, K),
                                 _UNWRITTEN_FLOAT,
                                 dtype=torch.float32),
        seq_slots=slots,
        seq_lens=torch.full((batch, ), seq_len, dtype=torch.int32),
        finished_beams=torch.zeros((max_batch, K), dtype=torch.int32),
        # One-shot latch the finish handler raises and the CBA step consumes;
        # tests set it alongside finished_beams to stage a pending harvest.
        pending_harvest=torch.zeros((max_batch, K), dtype=torch.bool),
        predecessor_beams=torch.zeros((max_batch, K), dtype=torch.int32),
        beam_idx_arange=torch.arange(K, dtype=torch.int32),
        cba=CBAState(
            end_ids=torch.full((max_batch, ), end_id, dtype=torch.int32),
            prompt_lens=torch.full((max_batch, ), prompt_len,
                                   dtype=torch.int32),
            original_tokens=torch.full((max_batch, K, attn_len),
                                       _UNWRITTEN_INT,
                                       dtype=torch.int32),
            cba_tokens=torch.full((max_batch, K, snap_len),
                                  BEAM_SEARCH_PAD_TOKEN,
                                  dtype=torch.int32),
            cba_cum_log_probs=torch.full((max_batch, K),
                                         _UNWRITTEN_FLOAT,
                                         dtype=torch.float32),
            cba_normed_scores=torch.full((max_batch, K),
                                         _CBA_NEG_INF,
                                         dtype=torch.float32),
            cba_lengths=torch.full((max_batch, K),
                                   _UNWRITTEN_INT,
                                   dtype=torch.int32),
            batch_dones=torch.zeros((max_batch, ), dtype=torch.bool),
            cba_caps=torch.full((max_batch, ), K, dtype=torch.int32),
            original_log_probs=torch.full((max_batch, K, attn_len),
                                          _UNWRITTEN_FLOAT,
                                          dtype=torch.float32),
            cba_log_probs=torch.full((max_batch, K, snap_len),
                                     _UNWRITTEN_FLOAT,
                                     dtype=torch.float32),
            max_seq_len=attn_len,
        ),
    )
    return m


@_kernel_test
def test_beam_search_cba_insert_and_slots():
    """One EOS candidate goes to CBA; slots continue with the 2 best actives."""
    K, vocab, end_id = 2, 5, 4
    prompt, gen = 2, 3
    seq_len = prompt + gen
    m = _make_cba_metadata(max_batch=2,
                           K=K,
                           attn_len=10,
                           snap_len=6,
                           seq_len=seq_len,
                           prompt_len=prompt,
                           end_id=end_id,
                           batch=1)
    # identity indirection; distinct original tokens per beam:
    # beam0 path tokens at abs pos 2..4 = [10, 11, 12]; beam1 = [20, 21, 22]
    for b in range(K):
        m.cache_indirection[0, b, :] = b
        m.cba.original_tokens[0, b, prompt:seq_len] = torch.tensor(
            [10 * (b + 1), 10 * (b + 1) + 1, 10 * (b + 1) + 2],
            dtype=torch.int32)
    m.cum_log_probs[0] = torch.tensor([-1.0, -1.2])

    logits = torch.full((K, vocab), -50.0)
    logits[0, end_id] = 10.0  # beam0 EOS: strongest candidate overall
    logits[0, 1] = 8.0  # beam0 t1: second
    logits[1, 2] = 9.0  # beam1 t2

    from tensorrt_llm._torch.pyexecutor.sampler.beam_search import \
        beam_search_sampling_batch_cba
    tokens, _ = beam_search_sampling_batch_cba(
        logits=logits,
        beam_width_in=K,
        beam_width_out=K,
        beam_search_args=m,
        temperature=1.0,
        early_stopping=0,
        length_penalty=1.0,
        return_probs=False,
    )

    # CBA got one entry: beam0 + EOS. logprob(EOS) = log_softmax([10, 8]) at
    # 10 => ~-0.127, so cum ~ -1.127, normed = cum / (gen+1)
    expected_cum = -1.0 + torch.log_softmax(logits[0], dim=-1)[end_id].item()
    assert m.cba.cba_lengths[0, 0].item() == gen + 1
    assert abs(m.cba.cba_cum_log_probs[0, 0].item() - expected_cum) < 1e-4
    assert abs(m.cba.cba_normed_scores[0, 0].item() - expected_cum / 4) < 1e-4
    assert m.cba.cba_normed_scores[0,
                                   1].item() == _CBA_NEG_INF  # only one entry
    # snapshot: beam0's generated tokens + EOS, padded
    assert m.cba.cba_tokens[0, 0].tolist() == [
        10, 11, 12, end_id, BEAM_SEARCH_PAD_TOKEN, BEAM_SEARCH_PAD_TOKEN
    ]
    # slots: (beam1,t2) ranks above (beam0,t1)? raw: b1t2 = -1.2+~0=-1.2;
    # b0t1 = -1.0-2.0=-3.0 (softmax vs 10.0) -> slot0 = (b1,t2), slot1=(b0,t1)
    assert m.predecessor_beams[0].tolist() == [1, 0]
    assert tokens[0].tolist() == [2, 1]
    # raw cums stored
    assert m.cum_log_probs[0, 0].item() > -1.5
    # not done: CBA not full
    assert not m.cba.batch_dones[0].item()
    assert (m.finished_beams[0] == FinishReason.NOT_FINISHED.value).all()


@_kernel_test
@pytest.mark.parametrize("early_stopping, expect_done", [(0, True), (2, False)])
def test_beam_search_cba_done_bound_by_early_stopping(early_stopping,
                                                      expect_done):
    """The done verdict's attainability bound depends on early_stopping when
    length_penalty > 0: FALSE (0) bounds by the current length, NEVER (2) by
    max_seq_len. With the same terrible actives and full CBA, FALSE stops but
    NEVER does not (a longer sequence could still beat the worst entry)."""
    K, vocab, end_id = 2, 5, 4
    prompt, gen = 2, 3
    seq_len = prompt + gen
    m = _make_cba_metadata(max_batch=1,
                           K=K,
                           attn_len=10,
                           snap_len=6,
                           seq_len=seq_len,
                           prompt_len=prompt,
                           end_id=end_id,
                           batch=1)
    for b in range(K):
        m.cache_indirection[0, b, :] = b
        m.cba.original_tokens[0, b, prompt:seq_len] = 7
    # CBA full; worst entry (min_kept) normed = -1.5.
    m.cba.cba_normed_scores[0] = torch.tensor([-0.1, -1.5])
    m.cba.cba_cum_log_probs[0] = torch.tensor([-0.4, -6.0])
    m.cba.cba_lengths[0] = torch.tensor([4, 4], dtype=torch.int32)
    m.cba.cba_tokens[0, :, :4] = 9
    # best active candidate cum ~ -9.4 (=-8 + log(1/4)). cand_len = gen+1 = 4,
    # max_gen = max_seq_len - prompt = 8. attainable:
    #   FALSE: -9.4/4 = -2.35 <= -1.5 -> done
    #   NEVER: -9.4/8 = -1.17;  -1.5 < -1.17 -> not done
    m.cum_log_probs[0] = torch.tensor([-8.0, -9.0])
    logits = torch.full((K, vocab), 0.0)  # uniform, no EOS domination
    logits[:, end_id] = -50.0

    from tensorrt_llm._torch.pyexecutor.sampler.beam_search import \
        beam_search_sampling_batch_cba
    beam_search_sampling_batch_cba(
        logits=logits,
        beam_width_in=K,
        beam_width_out=K,
        beam_search_args=m,
        temperature=1.0,
        early_stopping=early_stopping,
        length_penalty=1.0,
        return_probs=False,
    )
    assert m.cba.batch_dones[0].item() is expect_done
    # A beam with no reason of its own ends because the pool can no longer be
    # beaten, not because it hit a token, so the verdict publishes LENGTH.
    expected_reason = (FinishReason.LENGTH.value
                       if expect_done else FinishReason.NOT_FINISHED.value)
    assert (m.finished_beams[0] == expected_reason).all()
    # CBA untouched either way (no eligible end candidates this step).
    assert torch.allclose(m.cba.cba_normed_scores[0], torch.tensor([-0.1,
                                                                    -1.5]))


@_kernel_test
@pytest.mark.parametrize("penalty_as_tensor", [False, True])
def test_beam_search_cba_length_penalty_orders_pool(penalty_as_tensor):
    """length_penalty ranks pool entries by length-normalized score.

    Two beams finish on the same step with different generated lengths. The
    shorter one has the better raw cumulative log-prob, the longer one the
    better per-token score, so the penalty decides which entry the pool ranks
    first -- while both entries keep their raw cum_log_probs, since the
    normalization applies to the ranking key only.

    This is where length_penalty acts on the CBA path: candidate ranking
    itself does not use it (see beam_search_sampling_batch_cba), the pool
    scores and the attainability bound do.
    """
    K, vocab, end_id = 2, 5, 4
    prompt, gen = 2, 3
    seq_len = prompt + gen

    def run(length_penalty):
        m = _make_cba_metadata(max_batch=2,
                               K=K,
                               attn_len=10,
                               snap_len=6,
                               seq_len=seq_len,
                               prompt_len=prompt,
                               end_id=end_id,
                               batch=1)
        for b in range(K):
            m.cache_indirection[0, b, :] = b
        # Raw scores: beam0 ahead of beam1.
        m.cum_log_probs[0] = torch.tensor([-1.0, -2.0])

        # Both beams emit EOS this step, so both enter the pool.
        logits = torch.full((K, vocab), -50.0)
        logits[0, end_id] = 10.0
        logits[1, end_id] = 10.0

        penalty = (torch.full((1, ), length_penalty, dtype=torch.float32)
                   if penalty_as_tensor else length_penalty)
        before = _snapshot_cba_rows(
            m, ("cache_indirection", "cum_log_probs", "new_log_probs",
                "finished_beams", "predecessor_beams", "original_tokens",
                "cba_tokens", "cba_cum_log_probs", "cba_normed_scores",
                "cba_lengths", "batch_dones"))
        beam_search_sampling_batch_cba(
            logits=logits,
            beam_width_in=K,
            beam_width_out=K,
            beam_search_args=m,
            temperature=1.0,
            early_stopping=0,
            length_penalty=penalty,
            return_probs=False,
        )
        _assert_untouched_rows_intact(m, before, batch=1, max_batch=2)
        return m

    # Both runs put the same two hypotheses in the pool; only the ordering key
    # differs. Lengths are gen + 1 for the candidate that just ended.
    m_off = run(0.0)
    cums_off = sorted(
        round(v, 3) for v in m_off.cba.cba_cum_log_probs[0].tolist())
    normed_off = m_off.cba.cba_normed_scores[0].tolist()

    m_on = run(1.0)
    cums_on = sorted(
        round(v, 3) for v in m_on.cba.cba_cum_log_probs[0].tolist())
    normed_on = m_on.cba.cba_normed_scores[0].tolist()

    # Raw pool scores are identical either way: the penalty never touches them.
    assert cums_off == cums_on, (
        f"pool cum_log_probs must stay unnormalized, got {cums_off} vs "
        f"{cums_on}")

    # With the penalty off, the normalized score *is* the raw score.
    torch.testing.assert_close(torch.tensor(sorted(normed_off)),
                               torch.tensor(
                                   sorted(round(v, 3) for v in cums_off)),
                               atol=1e-3,
                               rtol=1e-3)

    # With it on, each entry is divided by its own length, so the scores move
    # apart from the raw ones.
    assert sorted(normed_on) != sorted(normed_off), (
        f"length_penalty should renormalize the pool scores; got {normed_on}")

    # And the division is by the entry's recorded length.
    lengths = m_on.cba.cba_lengths[0].tolist()
    for cum, normed, length in zip(m_on.cba.cba_cum_log_probs[0].tolist(),
                                   normed_on, lengths):
        if normed == _CBA_NEG_INF:
            continue  # unused pool slot
        assert abs(normed - cum / length) < 1e-3, (
            f"normed {normed} != cum {cum} / length {length}")


@_kernel_test
def test_beam_search_cba_replace_min():
    """A better finished path replaces the worst CBA entry when full."""
    K, vocab, end_id = 2, 5, 4
    prompt, gen = 2, 3
    seq_len = prompt + gen
    m = _make_cba_metadata(max_batch=1,
                           K=K,
                           attn_len=10,
                           snap_len=6,
                           seq_len=seq_len,
                           prompt_len=prompt,
                           end_id=end_id,
                           batch=1)
    for b in range(K):
        m.cache_indirection[0, b, :] = b
        m.cba.original_tokens[0, b, prompt:seq_len] = 30 + b
    m.cba.cba_normed_scores[0] = torch.tensor([-0.5, -2.0])
    m.cba.cba_cum_log_probs[0] = torch.tensor([-2.0, -8.0])
    m.cba.cba_lengths[0] = torch.tensor([4, 4], dtype=torch.int32)
    m.cba.cba_tokens[0, :, :4] = 5
    # beam0 emits EOS with cum ~ -4.0 -> normed -1.0: beats -2.0, not -0.5
    m.cum_log_probs[0] = torch.tensor([-4.0, -4.2])
    logits = torch.full((K, vocab), -50.0)
    logits[0, end_id] = 10.0
    logits[0, 1] = 9.0
    logits[1, 2] = 9.5

    from tensorrt_llm._torch.pyexecutor.sampler.beam_search import \
        beam_search_sampling_batch_cba
    beam_search_sampling_batch_cba(
        logits=logits,
        beam_width_in=K,
        beam_width_out=K,
        beam_search_args=m,
        temperature=1.0,
        early_stopping=0,
        length_penalty=1.0,
        return_probs=False,
    )
    expected_cum = -4.0 + torch.log_softmax(logits[0], dim=-1)[end_id].item()
    assert abs(m.cba.cba_normed_scores[0, 0].item() - (-0.5)) < 1e-6
    assert abs(m.cba.cba_normed_scores[0, 1].item() - expected_cum / 4) < 1e-4
    assert m.cba.cba_tokens[0, 1].tolist()[:4] == [30, 30, 30, end_id]


@_kernel_test
def test_beam_search_cba_harvest_stop_word_beam():
    """A beam latched finished (stop words) at step start is harvested into
    the CBA and its slot refills with an active candidate."""
    K, vocab, end_id = 2, 6, 5
    prompt, gen = 2, 3
    seq_len = prompt + gen
    m = _make_cba_metadata(max_batch=1,
                           K=K,
                           attn_len=10,
                           snap_len=6,
                           seq_len=seq_len,
                           prompt_len=prompt,
                           end_id=end_id,
                           batch=1)
    for b in range(K):
        m.cache_indirection[0, b, :] = b
        m.cba.original_tokens[0, b, prompt:seq_len] = torch.tensor(
            [70 + b, 71 + b, 72 + b], dtype=torch.int32)
    m.cum_log_probs[0] = torch.tensor([-1.5, -2.0])
    # beam 0 was latched STOP_WORDS by the finish handler after last step
    m.finished_beams[0, 0] = FinishReason.STOP_WORDS.value
    m.pending_harvest[0, 0] = True

    logits = torch.full((K, vocab), -50.0)
    logits[0, 1] = 10.0  # beam0's candidates must be ignored (harvested)
    logits[1, 2] = 9.0
    logits[1, 3] = 8.0

    from tensorrt_llm._torch.pyexecutor.sampler.beam_search import \
        beam_search_sampling_batch_cba
    tokens, _ = beam_search_sampling_batch_cba(
        logits=logits,
        beam_width_in=K,
        beam_width_out=K,
        beam_search_args=m,
        temperature=1.0,
        early_stopping=0,
        length_penalty=1.0,
        return_probs=False,
    )
    # harvested: beam0's own path (incl. the stop word already recorded),
    # length = gen (no appended token), normed = cum / gen
    assert m.cba.cba_lengths[0, 0].item() == gen
    assert abs(m.cba.cba_cum_log_probs[0, 0].item() - (-1.5)) < 1e-6
    assert abs(m.cba.cba_normed_scores[0, 0].item() - (-1.5 / gen)) < 1e-6
    assert m.cba.cba_tokens[0, 0].tolist() == [
        70, 71, 72, BEAM_SEARCH_PAD_TOKEN, BEAM_SEARCH_PAD_TOKEN,
        BEAM_SEARCH_PAD_TOKEN
    ]
    # both slots refilled from beam1 (beam0's row was masked)
    assert m.predecessor_beams[0].tolist() == [1, 1]
    assert tokens[0].tolist() == [2, 3]


@_kernel_test
def test_beam_search_cba_harvest_latch_clears_after_refill():
    """A harvested slot must not be harvested again on the next step.

    The harvest mask reads first_finish_reasons, which is also the request's
    persistent output finish reason and so survives the refill. Reading it as
    a transient latch means the *new* continuation occupying that slot looks
    finished on the following step: it gets harvested a second time, the pool
    gains an entry for a hypothesis that never ended, and the request can be
    declared done early.
    """
    from tensorrt_llm._torch.pyexecutor.sampler.beam_search import \
        beam_search_sampling_batch_cba

    K, vocab, end_id = 2, 6, 5
    prompt, gen = 2, 3
    seq_len = prompt + gen
    m = _make_cba_metadata(max_batch=1,
                           K=K,
                           attn_len=10,
                           snap_len=6,
                           seq_len=seq_len,
                           prompt_len=prompt,
                           end_id=end_id,
                           batch=1)
    for b in range(K):
        m.cache_indirection[0, b, :] = b
        m.cba.original_tokens[0, b, prompt:seq_len] = torch.tensor(
            [70 + b, 71 + b, 72 + b], dtype=torch.int32)
    m.cum_log_probs[0] = torch.tensor([-1.5, -2.0])
    # Step 1: beam 0 was latched STOP_WORDS by the finish handler.
    m.finished_beams[0, 0] = FinishReason.STOP_WORDS.value
    m.pending_harvest[0, 0] = True

    logits = torch.full((K, vocab), -50.0)
    logits[0, 1] = 10.0  # ignored: beam 0 is harvested, not expanded
    logits[1, 2] = 9.0
    logits[1, 3] = 8.0

    beam_search_sampling_batch_cba(
        logits=logits,
        beam_width_in=K,
        beam_width_out=K,
        beam_search_args=m,
        temperature=1.0,
        early_stopping=0,
        length_penalty=1.0,
        return_probs=False,
    )
    # Step 2 verifies the harvest happened and both slots refilled from beam 1.
    entries_after_first = int((m.cba.cba_normed_scores[0]
                               > _CBA_NEG_INF).sum().item())
    assert entries_after_first == 1, "the stop-word beam should be pooled once"
    assert m.predecessor_beams[0].tolist() == [1, 1]

    # Step 3: run again. Neither slot finished this step -- they hold the
    # continuations that refilled them, and the finish handler latched nothing
    # new, so nothing may be harvested.
    m.seq_lens = torch.full((1, ), seq_len + 1, dtype=torch.int32)
    logits2 = torch.full((K, vocab), -50.0)
    logits2[0, 1] = 7.0
    logits2[1, 2] = 6.0

    beam_search_sampling_batch_cba(
        logits=logits2,
        beam_width_in=K,
        beam_width_out=K,
        beam_search_args=m,
        temperature=1.0,
        early_stopping=0,
        length_penalty=1.0,
        return_probs=False,
    )
    # Step 4: the refilled continuation must not be pooled or masked again.
    entries_after_second = int((m.cba.cba_normed_scores[0]
                                > _CBA_NEG_INF).sum().item())
    assert entries_after_second == entries_after_first, (
        f"pool grew from {entries_after_first} to {entries_after_second}: an "
        "unfinished continuation was harvested because the stop-word latch "
        "outlived the refill")
    assert not m.cba.batch_dones[0].item(), (
        "request finished early: the re-harvest emptied the beam slots")


@_kernel_test
def test_beam_search_cba_reorders_stop_window():
    """The finish handler's stop-word window must follow beam swaps."""
    from tensorrt_llm._torch.pyexecutor.sampler.beam_search import \
        beam_search_sampling_batch_cba

    K, vocab, end_id = 2, 6, 5
    prompt, gen = 2, 3
    seq_len = prompt + gen
    m = _make_cba_metadata(max_batch=1,
                           K=K,
                           attn_len=10,
                           snap_len=6,
                           seq_len=seq_len,
                           prompt_len=prompt,
                           end_id=end_id,
                           batch=1)
    for b in range(K):
        m.cache_indirection[0, b, :] = b
    m.cum_log_probs[0] = torch.tensor([-5.0, -1.0])
    # window rows distinguishable per beam
    stop_window = torch.zeros((3, 1, K), dtype=torch.int32)
    stop_window[:, 0, 0] = 100
    stop_window[:, 0, 1] = 200
    m.stop_past_tokens = stop_window

    # beam1 dominates: both slots descend from beam 1
    logits = torch.full((K, vocab), -50.0)
    logits[1, 2] = 10.0
    logits[1, 3] = 9.0

    beam_search_sampling_batch_cba(
        logits=logits,
        beam_width_in=K,
        beam_width_out=K,
        beam_search_args=m,
        temperature=1.0,
        early_stopping=0,
        return_probs=False,
    )
    assert m.predecessor_beams[0].tolist() == [1, 1]
    # both window rows must now hold beam 1's history
    assert stop_window[:, 0, 0].tolist() == [200, 200, 200]
    assert stop_window[:, 0, 1].tolist() == [200, 200, 200]


def create_default_request(test_params: GeneralTestParams) -> LlmRequest:
    sampling_params = SamplingParams(n=test_params.beam_width,
                                     best_of=test_params.beam_width,
                                     use_beam_search=True)
    return LlmRequest(request_id=0,
                      seq_slot=test_params.seq_slot,
                      max_new_tokens=test_params.num_generated_tokens,
                      input_tokens=test_params.input_tokens,
                      end_id=test_params.end_id,
                      sampling_config=SamplingConfig(
                          sampling_params._get_sampling_config()),
                      return_log_probs=test_params.num_logprobs >= 0,
                      num_logprobs=test_params.num_logprobs,
                      is_streaming=False)


def create_default_sampler(test_params: GeneralTestParams) -> TorchSampler:
    sampler = TorchSampler(
        TorchSampler.Args(
            max_seq_len=test_params.max_seq_len,
            max_draft_len=0,
            max_num_sequences=test_params.max_batch_size,
            max_beam_width=test_params.max_beam_width,
            max_total_draft_tokens=0,
            disable_overlap_scheduler=True,
        ))
    max_beam_width = sampler.max_beam_width
    max_seq_len = sampler.max_seq_len
    max_batch_size = sampler.max_num_sequences

    # perform assertion tests for the selected parameter
    assert max_beam_width > test_params.beam_width, "Max beam width must be greater than beam width"
    assert max_seq_len > test_params.seq_len, "Max sequence length must be greater than sequence length"
    assert max_batch_size > test_params.batch_size, "Max batch size must be greater than batch size"
    assert max_batch_size > test_params.seq_slot, "Max batch size must be greater than sequence slot"
    beam_search_store = sampler.store.beam_search_store
    assert beam_search_store is not None
    assert beam_search_store.cache_indirection is not None
    assert beam_search_store.cache_indirection.shape == (
        max_batch_size, max_beam_width,
        max_seq_len), "Cache indirection shape mismatch"
    assert beam_search_store.original_tokens is not None
    assert beam_search_store.original_tokens.shape == (
        max_batch_size, max_beam_width,
        max_seq_len), "Original tokens shape mismatch"
    return sampler


def _vbws_request(beam_width_array: list[int] | None,
                  max_beam_width: int = 4) -> LlmRequest:
    """A beam-search request carrying ``beam_width_array`` (VBWS)."""
    sampling_params = SamplingParams(n=max_beam_width,
                                     best_of=max_beam_width,
                                     use_beam_search=True,
                                     beam_width_array=beam_width_array)
    return LlmRequest(request_id=0,
                      seq_slot=0,
                      max_new_tokens=16,
                      input_tokens=[1, 2, 3],
                      end_id=-1,
                      sampling_config=SamplingConfig(
                          sampling_params._get_sampling_config()),
                      is_streaming=False)


@pytest.mark.parametrize(
    "beam_width_array, expected",
    [
        # Index is (iteration - 1), clamped at both ends.
        ([2, 3, 4], [2, 2, 3, 4]),
        ([2, 2, 4], [2, 2, 2, 4]),
    ],
    ids=["widening", "flat_then_widening"],
)
def test_vbws_beam_width_by_iter_follows_array(beam_width_array: list[int],
                                               expected: list[int]):
    """get_beam_width_by_iter walks beam_width_array as decoding advances.

    Only non-decreasing arrays are covered: a narrowing one is rejected at
    admission (test_vbws_rejects_decreasing_beam_width_array), so pinning the
    width it would walk through would be pinning unreachable behaviour.
    """
    request = _vbws_request(beam_width_array)
    actual = []
    for iteration in range(len(expected)):
        request.decoding_iter = iteration
        actual.append(request.get_beam_width_by_iter())
    assert actual == expected

    # for_next_iteration looks one step ahead, i.e. it is the same sequence
    # shifted by one -- this is what feeds beam_width_out during sampling.
    request.decoding_iter = 0
    assert request.get_beam_width_by_iter(
        for_next_iteration=True) == expected[1]


def test_vbws_beam_width_by_iter_clamps_past_array_end():
    """Decoding longer than beam_width_array must hold the last width.

    The C++ formula used to clamp the iteration index with the global
    kMaxBeamWidthArrayLength (assuming a padded array), reading past the end of
    the raw user array and returning garbage; llmRequest.cpp now clamps with the
    array's own length. This pins the Python override, which clamps the same way
    and is kept so callers do not bind to a prebuilt library from before that
    fix; see LlmRequest.get_beam_width_by_iter.
    """
    beam_width_array = [2, 3, 4]
    request = _vbws_request(beam_width_array)
    # Run well past the end of the array.
    for iteration in range(len(beam_width_array), len(beam_width_array) + 8):
        request.decoding_iter = iteration
        assert request.get_beam_width_by_iter() == beam_width_array[-1]
        assert request.get_beam_width_by_iter(
            for_next_iteration=True) == beam_width_array[-1]


def test_vbws_cpp_formula_matches_past_array_end():
    """The C++ and Python clamps must agree once decoding outruns the array.

    The C++ implementation used to clamp the iteration index with the global
    kMaxBeamWidthArrayLength rather than the actual array length, so it read
    out of bounds and returned arbitrary widths (observed: 0, 32, 849 for a
    3-entry array). That starved the request in the C++ micro-batch scheduler
    and hung decoding; it is fixed in llmRequest.cpp. The scheduler calls into
    C++ directly, so pin the agreement here -- a failure means the two clamps
    have drifted apart again.
    """
    beam_width_array = [2, 3, 4]
    request = _vbws_request(beam_width_array)

    for iteration in range(len(beam_width_array) + 8):
        request.decoding_iter = iteration
        assert (request.get_beam_width_by_iter() ==
                CppLlmRequest.get_beam_width_by_iter(request, False))
        assert (request.get_beam_width_by_iter(
            for_next_iteration=True) == CppLlmRequest.get_beam_width_by_iter(
                request, True))

    # Past the end both must hold the last entry rather than read past it.
    for iteration in range(len(beam_width_array), len(beam_width_array) + 8):
        request.decoding_iter = iteration
        assert CppLlmRequest.get_beam_width_by_iter(
            request, False) == beam_width_array[-1]


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
def test_vbws_rejects_decreasing_beam_width_array(beam_width_array: list[int],
                                                  accepted: bool):
    """Only non-decreasing beam_width_array values are supported.

    The documented VBWS semantics only cover widening (see getBeamWidthByIter
    in llmRequest.h), and both samplers depend on it: a step writes the leading
    beam_width_out rows of the beam state while finalize reads py_beam_width
    (the array maximum) of them, so a narrowing array would return beams whose
    ancestry and cumulative log-probs are left over from an earlier, wider
    step. Reject at admission instead of emitting silently stale beams.
    """

    request = _vbws_request(beam_width_array)
    # The request itself is still constructible; admission is what rejects it,
    # and py_beam_width is the array maximum either way.
    assert request.py_beam_width == max(beam_width_array)

    # Drive the real admission check rather than a local re-implementation:
    # _validate_request only reads self.max_beam_width, so a stub carrying it
    # is enough to reach the beam_width_array branch without standing up an
    # executor. A test that mirrored the predicate would keep passing if the
    # production check were deleted.
    # Everything _validate_request touches besides the beam checks runs after
    # them and needs a live engine/sampler/KV cache manager, so stub those out;
    # the beam width and beam_width_array branches are reached with the real
    # code.
    executor = types.SimpleNamespace(
        max_beam_width=request.py_beam_width,
        kv_cache_transceiver=None,
        _validate_token_id_range=lambda _request: None,
        sampler=types.SimpleNamespace(validate_request=lambda _request: None),
        _validate_request_budget=lambda _request: None,
    )
    validate = functools.partial(
        PyExecutor._validate_request,
        executor,  # type: ignore[arg-type]  # stub: only max_beam_width is read
    )

    if accepted:
        validate(request)
    else:
        with pytest.raises(ValueError, match="decreases"):
            validate(request)


def test_beam_strategy_grouping_key_tolerates_trailing_fields():
    """The grouping key must not depend on the BeamSearch tuple's arity.

    strategy_grouping_key() pattern-matches the strategy tuple. It used to
    match exactly the fields known at the time, so appending row_stride made
    every beam-search strategy fall through to the "Unsupported strategy"
    branch and beam search failed for every request. Unit tests missed it
    because they build the ops and step classes directly, bypassing the
    grouping layer. Pin that trailing fields are ignored.
    """
    from tensorrt_llm._torch.pyexecutor.sampler.sampler_strategy import \
        FlashInferGroupedStrategySampler

    strategy = _beam_strategy(early_stopping=BeamSearchEarlyStop.TRUE)
    key = FlashInferGroupedStrategySampler.strategy_grouping_key(strategy)
    assert key == ("beam_search", strategy.beam_width_in,
                   strategy.beam_width_out, BeamSearchEarlyStop.TRUE)

    # row_stride is not part of the key: requests differing only in it still
    # group together.
    other = strategy._replace(row_stride=strategy.row_stride + 1)
    assert FlashInferGroupedStrategySampler.strategy_grouping_key(other) == key


def test_vbws_uniform_array_matches_fixed_width():
    """A constant beam_width_array must behave exactly like a fixed width."""
    max_beam_width = 4
    vbws = _vbws_request([max_beam_width] * 3, max_beam_width=max_beam_width)
    fixed = _vbws_request(None, max_beam_width=max_beam_width)
    for iteration in range(8):
        vbws.decoding_iter = iteration
        fixed.decoding_iter = iteration
        assert vbws.get_beam_width_by_iter() == fixed.get_beam_width_by_iter()
        assert vbws.get_beam_width_by_iter(
            for_next_iteration=True) == fixed.get_beam_width_by_iter(
                for_next_iteration=True)


def test_vbws_dummy_requests_excluded_from_width_check():
    """Every kind of dummy must be excluded from the mixed-width guard.

    ModelEngine rejects a generation batch whose requests report different
    per-iteration beam widths, but dummy requests carry no user request and are
    built at their own width -- CUDA-graph padding at the engine width,
    attention-DP and warmup dummies at width one. Filtering only CUDA-graph
    dummies let an attention-DP dummy abort an otherwise valid beam-search
    batch, so the guard filters on `is_dummy`; pin that it covers all three
    flags.
    """
    for flag in ("is_cuda_graph_dummy", "is_attention_dp_dummy",
                 "is_dummy_request"):
        request = _vbws_request([2, 3, 4])
        assert not request.is_dummy, f"{flag}: unset request must not be dummy"
        setattr(request, flag, True)
        assert request.is_dummy, (
            f"{flag} is not covered by is_dummy, so such requests would reach "
            "the mixed-beam-width check in ModelEngine and abort the batch")


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

    test_params = GeneralTestParams()
    beam_width = test_params.beam_width
    num_generated_tokens = test_params.num_generated_tokens

    torch.manual_seed(42)
    # current_path[beam, t] is the token beam `beam` held at position t before
    # correction; cache_indirection[beam, t] names the beam it descended from
    # at that position.
    current_path = torch.randint(0,
                                 test_params.vocab_size,
                                 (beam_width, num_generated_tokens),
                                 dtype=torch.int32,
                                 device="cuda")
    cache_indirection = torch.randint(0,
                                      beam_width,
                                      (beam_width, num_generated_tokens),
                                      dtype=torch.int64,
                                      device="cuda")

    corrected = _gather_beam_path(current_path=current_path,
                                  cache_indirection=cache_indirection)

    expected = torch.zeros_like(current_path)
    for beam in range(beam_width):
        for t in range(num_generated_tokens):
            expected[beam, t] = current_path[cache_indirection[beam, t], t]
    torch.testing.assert_close(corrected, expected)

    # An identity table must leave every beam untouched.
    identity = (torch.arange(beam_width, device="cuda", dtype=torch.int64).view(
        -1, 1).expand(-1, num_generated_tokens).contiguous())
    torch.testing.assert_close(
        _gather_beam_path(current_path=current_path,
                          cache_indirection=identity), current_path)


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

    request = _vbws_request(None, max_beam_width=num_beams)
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
    cache_indirection = (torch.arange(num_beams,
                                      dtype=torch.int64).view(-1, 1).expand(
                                          -1, total).contiguous().unsqueeze(0))
    original_tokens = torch.zeros((1, num_beams, total), dtype=torch.int32)
    original_tokens[0, :, prompt_len:] = torch.tensor(
        [[31, 32, 33, 34], [41, 42, 43, 44], [51, 52, 53, 54]],
        dtype=torch.int32)

    # Pool: entry 0 outranks every live beam, entry 1 sits between them,
    # entry 2 is an unused (-inf) slot that must never be selected.
    cba_tokens = torch.full((1, num_beams, total), pad, dtype=torch.int32)
    cba_tokens[0, 0, :2] = torch.tensor([11, 12], dtype=torch.int32)
    cba_tokens[0, 1, :3] = torch.tensor([21, 22, 23], dtype=torch.int32)

    cba_group = CBAGroupHost(
        pos={0: 0},
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
            dtype=torch.int32))
    assert history.cum_logprobs is not None
    torch.testing.assert_close(history.cum_logprobs,
                               torch.tensor([90.0, 7.0, 6.0]))


def test_finish_beams():
    """Test TorchSampler._finish_beams method.

    This test verifies that beams are correctly finalized.
    """

    @contextmanager
    def _uut_provider(
            is_warmup: bool) -> Generator[Callable[[], None], None, None]:
        test_params = GeneralTestParams()
        beam_width = test_params.beam_width
        num_generated_tokens = test_params.num_generated_tokens
        end_id = test_params.end_id
        batch_size = test_params.batch_size
        vocab_size = test_params.vocab_size
        num_logprobs = 1
        request = create_default_request(test_params)
        sampler = create_default_sampler(test_params)
        beam_search_store = sampler.store.beam_search_store
        assert beam_search_store is not None
        assert beam_search_store.cache_indirection is not None

        request.set_generated_tokens(
            torch.randint(0,
                          vocab_size, (beam_width, num_generated_tokens),
                          dtype=torch.int32).tolist())

        torch.manual_seed(42)
        # Do not keep end_id tokens in the tensor. This would interfere with the test.
        tokens = torch.randint(
            0,
            end_id, (batch_size, sampler.max_beam_width, num_generated_tokens),
            dtype=torch.int32)
        logprobs = torch.randn((batch_size, sampler.max_beam_width,
                                num_generated_tokens, num_logprobs),
                               dtype=torch.float32)
        cum_logprobs = logprobs[..., 0].sum(dim=-1)

        # assert that the  buffers are different from zero. Otherwise the test may pass if the function does not work.
        assert tokens.sum(
        ) > 0, "Tokens must not only contain zeros. Otherwise change the seed."
        assert torch.any(logprobs != 0) and torch.any(
            cum_logprobs != 0
        ), "Log probs and cumulative log probs must not only contain zeros. Otherwise change the seed."

        tokens[batch_size - 1, 0, num_generated_tokens //
               2:] = BEAM_SEARCH_PAD_TOKEN  # simulate early finished beam

        prompt_len = request.py_prompt_len

        token_history = []

        # test
        def _uut():
            nonlocal token_history

            for batch_idx in range(batch_size):
                beam_history = BeamHistory(
                    tokens=tokens[batch_idx, :beam_width],
                    logprobs=logprobs[batch_idx, :beam_width],
                    cum_logprobs=cum_logprobs[batch_idx, :beam_width])
                request.py_return_log_probs = False

                finalize_beam(request, beam_history)

                token_history.append(deepcopy(request.get_tokens()))

        yield _uut

        for batch_idx in range(batch_size):
            batch_final_tokens = token_history[batch_idx]

            if batch_idx < batch_size - 1:
                # requests are not finished yet
                final_tokens = torch.tensor(batch_final_tokens,
                                            dtype=torch.int32)[:, prompt_len:]
                torch.testing.assert_close(final_tokens,
                                           tokens[batch_idx, :beam_width])
            # Test the case where end_ids are present in the output
            else:
                # Given input for beam 0: [ token, token, ..., token, BEAM_SEARCH_PAD_TOKEN, BEAM_SEARCH_PAD_TOKEN, ..., BEAM_SEARCH_PAD_TOKEN]
                # Expected output for beam 0: [ token, token, ..., token]
                final_tokens_1p = torch.tensor(batch_final_tokens[1:],
                                               dtype=torch.int32)[:,
                                                                  prompt_len:]
                final_tokens_0 = torch.tensor(batch_final_tokens[0],
                                              dtype=torch.int32)[prompt_len:]
                torch.testing.assert_close(final_tokens_1p,
                                           tokens[batch_idx, 1:beam_width])
                torch.testing.assert_close(final_tokens_0.shape[0],
                                           num_generated_tokens // 2)
                torch.testing.assert_close(
                    final_tokens_0, tokens[batch_idx,
                                           0, :num_generated_tokens // 2])

    run_test_with_warmup(_uut_provider, max_sync_s=1)


def _beam_strategy(*,
                   early_stopping: BeamSearchEarlyStop,
                   beam_width_in: int = 2,
                   beam_width_out: int = 2,
                   temperature: float = 1.0,
                   length_penalty: float = 0.0,
                   diversity_rate: float = 0.0) -> BeamSearch:
    return BeamSearch(tag="beam_search",
                      beam_width_in=beam_width_in,
                      beam_width_out=beam_width_out,
                      temperature=temperature,
                      length_penalty=length_penalty,
                      diversity_rate=diversity_rate,
                      early_stopping=early_stopping)


class TestBeamSearchStepFromStrategies:
    """Cover the strategy-group -> BeamSearchStep construction.

    BeamSearchStep is abstract (`_select_and_update`), so only the concrete
    subclasses may be instantiated. These tests pin that contract: they fail
    with `TypeError: Can't instantiate abstract class BeamSearchStep` if the
    extraction of the shared fields is ever done by constructing the base.
    """

    @staticmethod
    def test_base_class_is_abstract():
        assert "_select_and_update" in _StrategyImpls.BeamSearchStep.__abstractmethods__
        with pytest.raises(TypeError, match="abstract"):
            # Deliberately instantiating the abstract base: that is what this
            # test pins, so mypy's (correct) complaint is expected here.
            _StrategyImpls.BeamSearchStep(  # type: ignore[abstract]
                2, 2, 2, torch.ones(1), None, None)

    @staticmethod
    @pytest.mark.parametrize(
        "early_stopping",
        [
            BeamSearchEarlyStop.TRUE,
            BeamSearchEarlyStop.FALSE,
            BeamSearchEarlyStop.NEVER,
        ],
    )
    @_kernel_test
    def test_from_strategies_builds_concrete_impl(early_stopping):
        # Every stopping mode runs on the candidate-beams-array step; the mode
        # only selects the done verdict computed inside it.
        expected_cls = _StrategyImpls.CBABeamSearchStep
        strategies = [_beam_strategy(early_stopping=early_stopping)]
        impl = expected_cls.from_strategies(strategies,
                                            cuda_device=torch.device("cuda"))
        assert type(impl) is expected_cls
        assert impl._beam_width_in == 2
        assert impl._beam_width_out == 2
        # Unset length_penalty / diversity_rate stay None so the ops can skip
        # the corresponding work.
        assert impl._length_penalty is None
        assert impl._diversity_rate is None

    @staticmethod
    @_kernel_test
    def test_from_strategies_carries_optional_fields():
        strategies = [
            _beam_strategy(early_stopping=BeamSearchEarlyStop.NEVER,
                           length_penalty=1.5,
                           diversity_rate=0.5)
        ]
        impl = _StrategyImpls.CBABeamSearchStep.from_strategies(
            strategies, cuda_device=torch.device("cuda"))
        assert impl._length_penalty is not None
        assert impl._diversity_rate is not None
        torch.testing.assert_close(impl._length_penalty,
                                   torch.tensor([1.5], device="cuda"))
        torch.testing.assert_close(impl._diversity_rate,
                                   torch.tensor([0.5], device="cuda"))
        assert impl._early_stopping is BeamSearchEarlyStop.NEVER

    @staticmethod
    @_kernel_test
    def test_common_fields_does_not_instantiate_base():
        """The shared extraction must return plain fields, not an instance."""
        strategies = [_beam_strategy(early_stopping=BeamSearchEarlyStop.FALSE)]
        fields = _StrategyImpls.BeamSearchStep._common_fields(
            strategies, torch.device("cuda"))
        assert not isinstance(fields, _StrategyImpls.BeamSearchStep)
        assert fields.beam_width_in == 2
        assert fields.beam_width_out == 2


@force_ampere  # Save H100 resource
class TestParameterValidation:
    """Ensure that unsupported request parameters do not crash/hang the engine."""

    @pytest.fixture(scope="module")
    @staticmethod
    def fixed_params():
        return {"max_tokens": 8, "max_beam_width": 4}

    @pytest.fixture(scope="module", params=[1, 4])
    @staticmethod
    def batch_size(request) -> int:
        return cast(int, request.param)

    @pytest.fixture(scope="module")
    @staticmethod
    def model_kwargs() -> dict[str, Any]:
        root = llm_models_root()
        assert root is not None
        return dict(model=root / "llama-models-v2" /
                    "TinyLlama-1.1B-Chat-v1.0", )

    # NB: Class-level fixture overrides do not work without this
    @pytest.fixture(scope="module")
    @staticmethod
    def llm(fixed_params, input_prompts, model_kwargs, batch_size: int):
        return _build_llm(
            fixed_params,
            input_prompts,
            (model_kwargs | dict(max_batch_size=batch_size)),
        )

    def _check_engine_responds(self, llm: LLM, input_prompts: list[str],
                               fixed_params: dict[str, Any]):
        _ = llm.generate(input_prompts,
                         sampling_params=SamplingParams(
                             max_tokens=fixed_params["max_tokens"],
                             n=1,
                             best_of=fixed_params["max_beam_width"],
                             use_beam_search=True,
                             end_id=-1,
                         ))

    @pytest.mark.timeout(120)
    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize("use_beam_search", [False, None])
    def test_use_beam_search_disabled_rejects_multiple_returns(
        self,
        llm: LLM,
        input_prompts: list[str],
        fixed_params: dict[str, Any],
        batch_size: int,
        use_beam_search: bool | None,
    ):
        # best_of > 1 without beam search is greedy multi-return, which the LLM
        # API rejects. Covers use_beam_search both explicitly False and omitted.
        if batch_size == 1:
            pytest.skip("Test does not depend on batch size")
        assert fixed_params["max_beam_width"] > 2
        params = dict(
            max_tokens=fixed_params["max_tokens"],
            n=1,
            best_of=fixed_params["max_beam_width"],
            end_id=-1,
        )
        if use_beam_search is not None:
            params["use_beam_search"] = use_beam_search
        with pytest.raises(
                ValueError,
                match=
                ".*Greedy decoding in the LLM API does not allow multiple returns.*"
        ):
            _ = llm.generate(input_prompts,
                             sampling_params=SamplingParams(**params))
        self._check_engine_responds(llm, input_prompts, fixed_params)

    @pytest.mark.timeout(120)
    @pytest.mark.threadleak(enabled=False)
    @pytest.mark.parametrize("early_stopping", [0, 2])
    def test_exhaustive_early_stopping_allowed_without_disagg(
        self,
        llm: LLM,
        input_prompts: list[str],
        fixed_params: dict[str, Any],
        batch_size: int,
        early_stopping: int,
    ):
        # Beam search is rejected wholesale under disaggregated serving (the
        # finished-candidate pool is not part of the handoff; TRTLLM-14792),
        # but regular serving must still accept every early_stopping mode.
        # NB: guards against the disagg check matching every request, e.g. by
        # testing a bound method rather than calling it.
        if batch_size == 1:
            pytest.skip("Test does not depend on batch size")
        outputs = llm.generate(input_prompts,
                               sampling_params=SamplingParams(
                                   max_tokens=fixed_params["max_tokens"],
                                   n=1,
                                   best_of=fixed_params["max_beam_width"],
                                   use_beam_search=True,
                                   early_stopping=early_stopping,
                                   end_id=-1,
                               ))
        assert isinstance(outputs, list)
        assert len(outputs) == len(input_prompts)
        for output in outputs:
            assert len(output.outputs) == 1
            token_ids = output.outputs[0].token_ids
            assert token_ids is not None and len(token_ids) > 0
        self._check_engine_responds(llm, input_prompts, fixed_params)

    @pytest.mark.timeout(120)
    @pytest.mark.threadleak(enabled=False)
    def test_smaller_beam_width(
        self,
        llm: LLM,
        input_prompts: list[str],
        fixed_params: dict[str, Any],
        batch_size: int,
    ):
        if batch_size == 1:
            pytest.skip("Test does not depend on batch size")
        assert fixed_params["max_beam_width"] > 2

        # A beam width above max_beam_width is rejected: buffers are only
        # allocated up to max_beam_width.
        with pytest.raises(RequestError,
                           match=".*is not equal to max_beam_width.*"):
            _ = llm.generate(input_prompts,
                             sampling_params=SamplingParams(
                                 max_tokens=fixed_params["max_tokens"],
                                 n=fixed_params["max_beam_width"] + 1,
                                 best_of=fixed_params["max_beam_width"] + 1,
                                 use_beam_search=True,
                                 end_id=-1,
                             ))
        self._check_engine_responds(llm, input_prompts, fixed_params)

        # A beam width below max_beam_width is rejected as well. TorchSampler
        # can sample it, but the attention metadata is stamped with
        # max_beam_width while the generation rows are laid out at the
        # per-request width, and the scheduler cannot keep widths from mixing
        # within a batch; see TRTLLM-14792.
        with pytest.raises(RequestError,
                           match=".*is not equal to max_beam_width.*"):
            _ = llm.generate(input_prompts,
                             sampling_params=SamplingParams(
                                 max_tokens=fixed_params["max_tokens"],
                                 n=2,
                                 best_of=2,
                                 use_beam_search=True,
                                 end_id=-1,
                             ))
        self._check_engine_responds(llm, input_prompts, fixed_params)

    @pytest.mark.timeout(120)
    @pytest.mark.threadleak(enabled=False)
    def test_logprobs_torch_sampler(
        self,
        llm: LLM,
        input_prompts: list[str],
        fixed_params: dict[str, Any],
        batch_size: int,
    ):
        if batch_size == 1:
            pytest.skip("Test does not depend on batch size")

        _ = llm.generate(input_prompts,
                         sampling_params=SamplingParams(
                             max_tokens=fixed_params["max_tokens"],
                             n=1,
                             best_of=fixed_params["max_beam_width"],
                             use_beam_search=True,
                             end_id=-1,
                             logprobs=0,
                         ))

        with pytest.raises(
                RequestError,
                match=
                ".*Beam search only supports returning the sampled logprob per token.*"
        ):
            _ = llm.generate(input_prompts,
                             sampling_params=SamplingParams(
                                 max_tokens=fixed_params["max_tokens"],
                                 n=1,
                                 best_of=fixed_params["max_beam_width"],
                                 use_beam_search=True,
                                 end_id=-1,
                                 logprobs=1,
                             ))

        with pytest.raises(
                RequestError,
                match=
                ".*Beam search does not support returning multiple logprobs per request.*"
        ):
            _ = llm.generate(input_prompts,
                             sampling_params=SamplingParams(
                                 max_tokens=fixed_params["max_tokens"],
                                 n=1,
                                 best_of=fixed_params["max_beam_width"],
                                 use_beam_search=True,
                                 end_id=-1,
                                 logprobs=2,
                             ))

        self._check_engine_responds(llm, input_prompts, fixed_params)


if __name__ == "__main__":
    pytest.main([__file__])
