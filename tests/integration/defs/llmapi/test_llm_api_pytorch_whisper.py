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
"""End-to-end Whisper (audio encoder-decoder) tests on the PyTorch backend.

Whisper feeds the encoder a log-mel feature tensor carried via
``multi_modal_data["audio"]`` (not encoder token ids); the decoder prompt is
the forced task-token prefix. Expected outputs below were pinned from
``openai/whisper-tiny`` fp32 greedy on the LibriSpeech test-clean utterance
1221-135766-0002 (the wav shipped next to the legacy whisper checkpoints in
``LLM_MODELS_ROOT``); HF transformers greedy produces the identical
transcript.
"""

import threading
from pathlib import Path

import pytest
import soundfile
import torch

from tensorrt_llm._torch.pyexecutor.engine.runners.encoder_decoder import EncoderDecoderRunner
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import SchedulerOutput
from tensorrt_llm.llmapi import (
    LLM,
    CudaGraphConfig,
    EncodeCudaGraphConfig,
    KvCacheConfig,
    SamplingParams,
    SchedulerConfig,
)

from ..conftest import llm_models_root

_MAX_NEW_TOKENS = 96
_MIN_GPU_MEMORY_MB = 16_000
_FREE_GPU_MEMORY_FRACTION = 0.2
_CROSS_KV_CACHE_FRACTION = 0.5
# Every Whisper request produces this many encoder positions, whatever the audio
# length. It sets the cross-KV pool capacity and the encoder graph shapes.
_ENCODER_OUTPUT_LEN = 1500
# whisper-tiny fp32 greedy on 1221-135766-0002.wav (matches HF transformers).
_EXPECTED_GREEDY_OUTPUT_TOKEN_IDS = [
    1939,
    613,
    4598,
    8028,
    389,
    3011,
    582,
    259,
    1570,
    365,
    1454,
    813,
    38675,
    3378,
    13,
    50257,
]
# In half precision the first token is a near-tie between " yet" (1939) and
# " Yet" (10890), and which one wins varies with dtype and GPU architecture
# (HF: fp32/fp16 -> 1939, bf16 -> 10890; TRT-LLM fp16: SM120 -> 10890,
# DGX B200 -> 1939). Accept either; the remaining tokens are stable.
_BORDERLINE_FIRST_TOKEN_IDS = (1939, 10890)
_EXPECTED_TRANSCRIPT_FRAGMENT = "thoughts affected hester"

pytestmark = [
    pytest.mark.skip_less_device(1),
    pytest.mark.skip_less_device_memory(_MIN_GPU_MEMORY_MB),
    pytest.mark.threadleak(enabled=False),
]


def _get_whisper_model_path() -> str:
    # llm_models_root() asserts (with a clear message) when no model cache is
    # reachable, so no None check is needed here.
    models_root = llm_models_root()
    candidates = [
        # HF-format snapshot in the CI share (next to the legacy checkpoints).
        Path(models_root) / "whisper-models" / "whisper-tiny",
        Path(models_root) / "whisper" / "whisper-tiny",
        Path(models_root) / "whisper-tiny",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    pytest.skip(
        f"HF-format whisper-tiny not found under {models_root} (tried "
        f"{[str(c) for c in candidates]})."
    )


def _get_audio_path() -> str:
    models_root = llm_models_root()
    for legacy_dir in ("large-v3", "large-v2"):
        candidate = Path(models_root) / "whisper-models" / legacy_dir / "1221-135766-0002.wav"
        if candidate.exists():
            return str(candidate)
    pytest.skip(f"1221-135766-0002.wav not found under {models_root}/whisper-models.")


def _make_llm(
    model_path: str,
    max_beam_width: int = 1,
    use_kv_cache_manager_v2: bool = True,
    torch_dtype: str | None = None,
    cuda_graph_batch_sizes: list[int] | None = None,
    tensor_parallel_size: int = 1,
    encoder_graphs: bool = False,
    enable_block_reuse: bool = False,
    use_python_scheduler: bool = True,
) -> LLM:
    """Build a Whisper LLM for the test matrix, optionally with encoder CUDA graphs."""
    # CudaGraphConfig captures the decode step; the enc-dec encoder step opts in
    # separately through `encoder_cuda_graph_config`. Requesting decoder graphs
    # must work for every dtype.
    cuda_graph_config = (
        CudaGraphConfig(
            batch_sizes=cuda_graph_batch_sizes,
            enable_padding=True,
        )
        if cuda_graph_batch_sizes is not None
        else None
    )
    encoder_kwargs = {}
    if encoder_graphs:
        # Whisper's encoder emits a fixed `_ENCODER_OUTPUT_LEN` positions per
        # request, so the graph key is the batch size alone; num_tokens and
        # seq_lens are derived from the model and are not supplied here.
        encoder_batch_sizes = list(cuda_graph_batch_sizes or [1])
        encoder_kwargs = {
            "encoder_max_batch_size": max(encoder_batch_sizes),
            "encoder_cuda_graph_config": EncodeCudaGraphConfig(
                batch_sizes=encoder_batch_sizes,
                enable_padding=True,
            ),
        }
    dtype_kwargs = {}
    if torch_dtype is not None:
        # The checkpoint's torch_dtype wins over `dtype` in the PyTorch
        # backend; model_kwargs is the effective override (same as T5/BART).
        dtype_kwargs = {"dtype": torch_dtype, "model_kwargs": {"torch_dtype": torch_dtype}}
    return LLM(
        model_path,
        attn_backend="TRTLLM",
        cuda_graph_config=cuda_graph_config,
        disable_overlap_scheduler=True,  # overlap scheduler unsupported
        enable_chunked_prefill=False,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=enable_block_reuse,
            free_gpu_memory_fraction=_FREE_GPU_MEMORY_FRACTION,
            cross_kv_cache_fraction=_CROSS_KV_CACHE_FRACTION,
            use_kv_cache_manager_v2=use_kv_cache_manager_v2,
        ),
        max_batch_size=2,
        max_beam_width=max_beam_width,
        # Cross-KV pool capacity; the default (1024) is smaller than the
        # 1500 encoder positions every Whisper request produces.
        max_input_len=_ENCODER_OUTPUT_LEN,
        max_num_tokens=2 * _ENCODER_OUTPUT_LEN,
        scheduler_config=SchedulerConfig(use_python_scheduler=use_python_scheduler),
        tensor_parallel_size=tensor_parallel_size,
        **encoder_kwargs,
        **dtype_kwargs,
    )


def _audio_prompt(wave, sample_rate, prompt: str = ""):
    return {"prompt": prompt, "multi_modal_data": {"audio": [(wave, sample_rate)]}}


def test_whisper_pytorch_transcribe_end_to_end(monkeypatch):
    """Greedy transcription: exact pinned token ids, single + batch-2.

    Deliberately no TRTLLM_SKIP_KV_CACHE_ESTIMATION: the engine must gate
    estimation off for enc-dec itself, so the default config is what's tested.
    """
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    model_path = _get_whisper_model_path()
    wave, sample_rate = soundfile.read(_get_audio_path())
    sampling_params = SamplingParams(temperature=0.0, max_tokens=_MAX_NEW_TOKENS)

    with _make_llm(model_path) as llm:
        outputs = llm.generate([_audio_prompt(wave, sample_rate)], sampling_params)
        completion = outputs[0].outputs[0]
        assert list(completion.token_ids) == _EXPECTED_GREEDY_OUTPUT_TOKEN_IDS
        assert _EXPECTED_TRANSCRIPT_FRAGMENT in completion.text.lower()

        # Batch of 2: the encoder packs both mels into one pass; each request
        # keeps its own cross-KV. Identical clips must transcribe identically.
        outputs = llm.generate(
            [_audio_prompt(wave, sample_rate) for _ in range(2)], sampling_params
        )
        for output in outputs:
            assert list(output.outputs[0].token_ids) == _EXPECTED_GREEDY_OUTPUT_TOKEN_IDS

        # Token-only prompts cannot feed a feature-driven encoder and must be
        # rejected at submission rather than poison the encoder batch.
        with pytest.raises(Exception, match="multi_modal_data"):
            llm.generate([[50258, 50259, 50359, 50363]], sampling_params)

        # A non-empty text prompt is the decoder prompt: forcing German makes
        # whisper-tiny emit a (rough) German rendering — assert it diverges
        # from the English transcript and is non-empty.
        outputs = llm.generate(
            [
                _audio_prompt(
                    wave,
                    sample_rate,
                    prompt="<|startoftranscript|><|de|><|transcribe|><|notimestamps|>",
                )
            ],
            sampling_params,
        )
        german_text = outputs[0].outputs[0].text.strip()
        assert german_text
        assert _EXPECTED_TRANSCRIPT_FRAGMENT not in german_text.lower()

        # PreprocessedInputs skips the input processor, so the executor-side
        # max_tokens clamp is what bounds generation to the decoder table (448)
        # rather than the encoder-sized max_seq_len. ignore_eos forces the
        # request to reach that cap.
        preprocessed = llm.preprocess(
            _audio_prompt(wave, sample_rate), SamplingParams(temperature=0.0)
        )
        fresh_params = SamplingParams(temperature=0.0, max_tokens=10_000, ignore_eos=True)
        result = llm.generate_async(preprocessed, fresh_params).result()
        generated = list(result.outputs[0].token_ids)
        assert len(generated) == 448 - len(preprocessed.prompt_token_ids)
        expected = _EXPECTED_GREEDY_OUTPUT_TOKEN_IDS
        assert generated[: len(expected)] == expected


_CONCURRENT_REQUESTS = 8
# How long the first scheduling iteration waits for the other requests to be queued.
_SUBMIT_TIMEOUT_S = 60
_BATCH_WAIT_TIMEOUT_ITERS = 10
_BATCH_WAIT_MAX_TOKENS_RATIO = 0.5
_BATCH_WAITING = {
    "batch_wait_timeout_iters": _BATCH_WAIT_TIMEOUT_ITERS,
    "batch_wait_max_tokens_ratio": _BATCH_WAIT_MAX_TOKENS_RATIO,
}
# While decoder work runs, batch waiting holds back encoder requests whose
# scheduled tokens stay below this threshold, until its deadline.
_BATCH_WAIT_TOKENS = _BATCH_WAIT_MAX_TOKENS_RATIO * _CONCURRENT_REQUESTS * _ENCODER_OUTPUT_LEN


@pytest.mark.parametrize(
    "llm_kwargs,requests_per_encoder_step",
    [
        # One request's encoder output fills encoder_max_num_tokens.
        pytest.param({"encoder_max_num_tokens": _ENCODER_OUTPUT_LEN}, 1, id="token-limit"),
        pytest.param({"encoder_max_batch_size": 2}, 2, id="request-limit"),
        # All requests' encoder outputs fit both limits together, exactly.
        pytest.param(
            {
                "encoder_max_num_tokens": _CONCURRENT_REQUESTS * _ENCODER_OUTPUT_LEN,
                "encoder_max_batch_size": _CONCURRENT_REQUESTS,
            },
            _CONCURRENT_REQUESTS,
            id="within-limits",
        ),
        # With batch waiting on, encoder requests that fill one encoder step, by
        # tokens or by requests, run at once, although their tokens stay below
        # the batch-waiting threshold while decoder work is in flight.
        pytest.param(
            {"encoder_max_num_tokens": _ENCODER_OUTPUT_LEN, **_BATCH_WAITING},
            1,
            id="token-limit-batch-waiting",
        ),
        pytest.param(
            {"encoder_max_batch_size": 2, **_BATCH_WAITING},
            2,
            id="request-limit-batch-waiting",
        ),
    ],
)
def test_whisper_pytorch_concurrent_requests_encoder_step_limits(
    monkeypatch: pytest.MonkeyPatch,
    llm_kwargs: dict[str, int | float],
    requests_per_encoder_step: int,
) -> None:
    """Concurrent requests are encoded within encoder_max_num_tokens and encoder_max_batch_size.

    max_num_tokens and max_batch_size let the scheduler offer every request's
    encoder work in one iteration. Each iteration that offers encoder requests
    runs one encoder step with as many of them as the encoder limits allow, and
    every request transcribes exactly. With batch waiting on, an iteration that
    offers less than one full encoder step may hold it back instead.
    """
    batch_waiting = "batch_wait_max_tokens_ratio" in llm_kwargs
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    model_path = _get_whisper_model_path()
    wave, sample_rate = soundfile.read(_get_audio_path())
    sampling_params = SamplingParams(temperature=0.0, max_tokens=_MAX_NEW_TOKENS)

    with LLM(
        model_path,
        attn_backend="TRTLLM",
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
        enable_chunked_prefill=False,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            free_gpu_memory_fraction=_FREE_GPU_MEMORY_FRACTION,
            cross_kv_cache_fraction=_CROSS_KV_CACHE_FRACTION,
        ),
        max_batch_size=_CONCURRENT_REQUESTS,
        max_input_len=_ENCODER_OUTPUT_LEN,
        max_num_tokens=_CONCURRENT_REQUESTS * _ENCODER_OUTPUT_LEN,
        **llm_kwargs,
    ) as llm:
        executor = llm._executor.engine
        schedule_request = executor.scheduler.schedule_request
        forward_encoder = executor.model_engine.forward_encoder
        # Per scheduling iteration: the encoder requests offered, whether decoder
        # work was scheduled, and the tokens batch waiting compares with its
        # threshold.
        offers: list[tuple[int, bool, int]] = []
        # Per encoder step: its scheduling iteration and its number of requests.
        encoder_steps: list[tuple[int, int]] = []
        all_submitted = threading.Event()
        # Whether each scheduling iteration saw every request queued in time.
        submit_waits: list[bool] = []

        def recording_schedule_request(*args: object, **kwargs: object) -> SchedulerOutput:
            """Hold the first scheduling iteration, then record what the scheduler offered."""
            # The first iteration waits until every request is queued, so that
            # the next request fetch takes all the others at once.
            submit_waits.append(all_submitted.wait(timeout=_SUBMIT_TIMEOUT_S))
            scheduler_output = schedule_request(*args, **kwargs)
            num_offered = len(scheduler_output.encoder_requests)
            offers.append(
                (
                    num_offered,
                    bool(scheduler_output.context_requests or scheduler_output.generation_requests),
                    num_offered * _ENCODER_OUTPUT_LEN + len(scheduler_output.generation_requests),
                )
            )
            return scheduler_output

        def recording_forward_encoder(
            encoder_requests: list[LlmRequest], *args: object, **kwargs: object
        ) -> tuple[torch.Tensor, list[int]]:
            """Record the size of the encoder step and the iteration that runs it."""
            encoder_steps.append((len(offers) - 1, len(encoder_requests)))
            return forward_encoder(encoder_requests, *args, **kwargs)

        monkeypatch.setattr(executor.scheduler, "schedule_request", recording_schedule_request)
        monkeypatch.setattr(executor.model_engine, "forward_encoder", recording_forward_encoder)
        prompts = [
            llm.preprocess(_audio_prompt(wave, sample_rate), sampling_params)
            for _ in range(_CONCURRENT_REQUESTS)
        ]
        try:
            futures = [llm.generate_async(prompt, sampling_params) for prompt in prompts]
        finally:
            all_submitted.set()
        for future in futures:
            assert list(future.result().outputs[0].token_ids) == _EXPECTED_GREEDY_OUTPUT_TOKEN_IDS

    assert all(submit_waits), f"the requests were not all queued within {_SUBMIT_TIMEOUT_S} s"
    steps_by_iteration: dict[int, list[int]] = {}
    for iteration, num_requests in encoder_steps:
        steps_by_iteration.setdefault(iteration, []).append(num_requests)
    waiting_iterations = 0
    full_steps_released_early = 0
    for iteration, (num_offered, has_decoder_work, num_tokens) in enumerate(offers):
        steps = steps_by_iteration.get(iteration, [])
        if batch_waiting and num_offered < requests_per_encoder_step:
            # Less than one full encoder step: batch waiting may hold it back.
            assert steps in ([], [num_offered])
        else:
            assert steps == ([min(num_offered, requests_per_encoder_step)] if num_offered else [])
        if (
            batch_waiting
            and num_offered >= requests_per_encoder_step
            and has_decoder_work
            and num_tokens < _BATCH_WAIT_TOKENS
            and waiting_iterations < _BATCH_WAIT_TIMEOUT_ITERS
        ):
            # Neither the token threshold nor the deadline released this step.
            full_steps_released_early += 1
        if num_offered and not steps:
            waiting_iterations += 1
    if batch_waiting:
        assert full_steps_released_early > 0
    if requests_per_encoder_step < _CONCURRENT_REQUESTS:
        # Some iteration offered more encoder requests than one step holds.
        assert max(num_offered for num_offered, _, _ in offers) > requests_per_encoder_step
    else:
        # Requests within the limits share an encoder step.
        assert max(num_requests for _, num_requests in encoder_steps) > 1


# The bf16 graphs-on cases capture decode graphs over
# batch_size * beam_width sequences.
_BEAM_SEARCH_CASES = [
    pytest.param(None, True, None, False, id="fp32-kv-v2-graphs-off-beam2"),
    pytest.param("bfloat16", True, [1, 2], True, id="bf16-kv-v2-decoder-graphs-on-beam2"),
]


def test_whisper_pytorch_block_reuse_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    """Greedy transcription when KV block reuse is requested.

    Whisper requests carry encoder features, not encoder token ids, so the
    executor disables reuse for both KV pools and must still admit and run
    them (https://nvbugs/6713231). This checks the Python reuse override;
    the C++ scheduler's missing-token guards are covered by its unit test.
    Distinct audio clips sharing the decoder prompt must match their
    reuse-disabled results, both sequentially and in a mixed batch.
    """
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    model_path = _get_whisper_model_path()
    wave, sample_rate = soundfile.read(_get_audio_path())
    sampling_params = SamplingParams(temperature=0.0, max_tokens=_MAX_NEW_TOKENS)
    # Derive a distinct speech clip from the existing CI fixture, keeping the
    # sample rate and decoder prompt identical. No extra model-share asset is needed.
    cropped_wave = wave[: len(wave) // 2].copy()
    batches = ([wave], [cropped_wave], [wave, cropped_wave])
    results = {}

    for enable_block_reuse in (False, True):
        batch_results = []
        with _make_llm(
            model_path,
            enable_block_reuse=enable_block_reuse,
            use_python_scheduler=False,
        ) as llm:
            executor = llm._executor.engine
            for resource_type in (
                ResourceManagerType.KV_CACHE_MANAGER,
                ResourceManagerType.CROSS_KV_CACHE_MANAGER,
            ):
                manager = executor.resource_manager.get_resource_manager(resource_type)
                assert manager is not None
                assert manager.enable_block_reuse is False
            assert executor.model_engine.attn_runtime_features.cache_reuse is False
            for batch in batches:
                outputs = llm.generate(
                    [_audio_prompt(clip, sample_rate) for clip in batch],
                    sampling_params,
                )
                assert len(outputs) == len(batch)
                batch_results.append(
                    [
                        (list(output.outputs[0].token_ids), output.outputs[0].text)
                        for output in outputs
                    ]
                )
        results[enable_block_reuse] = batch_results

    baseline = results[False]
    assert baseline[0][0][0] == _EXPECTED_GREEDY_OUTPUT_TOKEN_IDS
    assert _EXPECTED_TRANSCRIPT_FRAGMENT in baseline[0][0][1].lower()
    assert baseline[1][0][1].strip()
    # Ensure the second input actually distinguishes feature-conditioned caches.
    assert baseline[0][0][0] != baseline[1][0][0]
    assert baseline[2] == [baseline[0][0], baseline[1][0]]
    assert results[True] == baseline


@pytest.mark.parametrize(
    "torch_dtype,use_kv_cache_manager_v2,cuda_graph_batch_sizes,graphs_captured",
    _BEAM_SEARCH_CASES,
)
def test_whisper_pytorch_beam_search(
    monkeypatch,
    torch_dtype,
    use_kv_cache_manager_v2,
    cuda_graph_batch_sizes,
    graphs_captured,
):
    """Beam-2 transcription (cross-KV shared across beams)."""
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    model_path = _get_whisper_model_path()
    wave, sample_rate = soundfile.read(_get_audio_path())

    llm = _make_llm(
        model_path,
        max_beam_width=2,
        use_kv_cache_manager_v2=use_kv_cache_manager_v2,
        torch_dtype=torch_dtype,
        cuda_graph_batch_sizes=cuda_graph_batch_sizes,
    )
    with llm:
        beam_params = SamplingParams(
            best_of=2, n=1, temperature=0.0, use_beam_search=True, max_tokens=_MAX_NEW_TOKENS
        )
        outputs = llm.generate([_audio_prompt(wave, sample_rate)], beam_params)
        assert _EXPECTED_TRANSCRIPT_FRAGMENT in outputs[0].outputs[0].text.lower()
        _assert_cuda_graph_state(llm, captured=graphs_captured)


def _assert_cuda_graph_state(llm: LLM, captured: bool, encoder_captured: bool = False) -> None:
    """Introspect the in-process engine (single-process mode only).

    EncoderDecoderRunner owns both the encoder stage graphs and the decoder
    graphs.
    """
    model_engine = llm._executor.engine.model_engine
    assert model_engine.cuda_graph_runner.enabled == captured
    assert bool(model_engine.cuda_graph_runner.graphs) == captured

    assert isinstance(model_engine._runner, EncoderDecoderRunner)
    encoder_runner = model_engine._runner._encoder_stage._encoder_cuda_graph_runner
    if not encoder_captured:
        assert not encoder_runner.enabled
        assert not encoder_runner.graphs
        return
    assert encoder_runner.enabled
    assert encoder_runner.graphs
    assert encoder_runner.feature_mode
    assert encoder_runner.is_encoder_decoder
    # Capture alone is not enough: `pad_batch` and the shape checks in
    # `EncoderMixin._prepare_encoder_feature_graph_inputs` can route every request
    # to the eager encoder while `graphs` stays populated. That silent fallback would
    # pass every output assertion above. Only the replay counter rules it out,
    # and only against the warmup baseline: the capture pass replays each key
    # once immediately after capturing it, so anything at or below
    # `len(graphs)` is still explainable by warmup alone.
    assert encoder_runner.num_feature_replays > len(encoder_runner.graphs)


# Feature-combination matrix mirroring the T5/BART enc-dec coverage. Cases:
# (torch_dtype override or None for checkpoint fp32, kv manager v2, decoder
# cuda-graph batch sizes, graphs must capture, TP size, encoder graphs).
# Beam coverage lives in _BEAM_SEARCH_CASES. The
# fp32+graphs-requested case covers fp32 enc-dec capturing decoder graphs. The
# encoder-graphs case additionally captures the encoder step, which must not
# change a single token.
_FEATURE_COMBINATION_CASES = [
    pytest.param(None, True, None, False, 1, False, id="fp32-kv-v2-graphs-off-greedy"),
    pytest.param(None, True, [1, 2], True, 1, False, id="fp32-kv-v2-graphs-requested-greedy"),
    pytest.param(
        "bfloat16", True, [1, 2], True, 1, False, id="bf16-kv-v2-decoder-graphs-on-greedy"
    ),
    pytest.param("bfloat16", True, [1, 2], True, 1, True, id="bf16-kv-v2-encoder-graphs-on-greedy"),
    pytest.param("float16", True, None, False, 1, False, id="fp16-kv-v2-graphs-off-greedy"),
    pytest.param(
        None,
        True,
        None,
        False,
        2,
        False,
        id="fp32-kv-v2-graphs-off-greedy-tp2",
        marks=pytest.mark.skip_less_device(2),
    ),
]


@pytest.mark.parametrize(
    "torch_dtype,use_kv_cache_manager_v2,cuda_graph_batch_sizes,graphs_captured,tp_size,"
    "encoder_graphs",
    _FEATURE_COMBINATION_CASES,
)
def test_whisper_pytorch_feature_combinations(
    monkeypatch,
    torch_dtype,
    use_kv_cache_manager_v2,
    cuda_graph_batch_sizes,
    graphs_captured,
    tp_size,
    encoder_graphs,
):
    """Greedy transcription across dtype/kv-cache-manager/CUDA-graph/TP combos.

    Batch-1 and batch-2 must both reproduce the pinned token ids (fp32 exact;
    fp16/bf16 exact except the hardware-dependent borderline first token).
    """
    if tp_size == 1:
        monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    model_path = _get_whisper_model_path()
    wave, sample_rate = soundfile.read(_get_audio_path())
    sampling_params = SamplingParams(temperature=0.0, max_tokens=_MAX_NEW_TOKENS)

    llm = _make_llm(
        model_path,
        use_kv_cache_manager_v2=use_kv_cache_manager_v2,
        torch_dtype=torch_dtype,
        cuda_graph_batch_sizes=cuda_graph_batch_sizes,
        tensor_parallel_size=tp_size,
        encoder_graphs=encoder_graphs,
    )
    with llm:
        for batch_size in (1, 2):
            outputs = llm.generate(
                [_audio_prompt(wave, sample_rate) for _ in range(batch_size)],
                sampling_params,
            )
            for output in outputs:
                completion = output.outputs[0]
                token_ids = list(completion.token_ids)
                if torch_dtype is None:
                    assert token_ids == _EXPECTED_GREEDY_OUTPUT_TOKEN_IDS
                else:
                    assert token_ids[0] in _BORDERLINE_FIRST_TOKEN_IDS
                    assert token_ids[1:] == _EXPECTED_GREEDY_OUTPUT_TOKEN_IDS[1:]
                assert _EXPECTED_TRANSCRIPT_FRAGMENT in completion.text.lower()

        if tp_size == 1:
            _assert_cuda_graph_state(llm, captured=graphs_captured, encoder_captured=encoder_graphs)
