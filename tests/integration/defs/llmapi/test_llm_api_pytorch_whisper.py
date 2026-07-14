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

from pathlib import Path

import pytest
import soundfile

from tensorrt_llm.llmapi import LLM, CudaGraphConfig, KvCacheConfig, SamplingParams, SchedulerConfig

from ..conftest import llm_models_root

_MAX_NEW_TOKENS = 96
_MIN_GPU_MEMORY_MB = 16_000
_FREE_GPU_MEMORY_FRACTION = 0.2
_CROSS_KV_CACHE_FRACTION = 0.5
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
# Half precision (both HF and TRT-LLM) flips the first token " yet" (1939)
# to " Yet" (10890); the rest of the sequence is unchanged.
_EXPECTED_GREEDY_OUTPUT_TOKEN_IDS_HALF = [10890
                                          ] + _EXPECTED_GREEDY_OUTPUT_TOKEN_IDS[1:]
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
    use_kv_cache_manager_v2: bool = False,
    torch_dtype: str | None = None,
    cuda_graph_batch_sizes: list[int] | None = None,
    tensor_parallel_size: int = 1,
) -> LLM:
    # CudaGraphConfig captures the decode step only; fp32 enc-dec declines
    # graphs at engine init (workspace-sizing guard), so requesting them must
    # still work for every dtype.
    cuda_graph_config = (
        CudaGraphConfig(batch_sizes=cuda_graph_batch_sizes, enable_padding=True)
        if cuda_graph_batch_sizes is not None
        else None
    )
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
            enable_block_reuse=False,
            free_gpu_memory_fraction=_FREE_GPU_MEMORY_FRACTION,
            cross_kv_cache_fraction=_CROSS_KV_CACHE_FRACTION,
            use_kv_cache_manager_v2=use_kv_cache_manager_v2,
        ),
        max_batch_size=2,
        max_beam_width=max_beam_width,
        # Cross-KV pool capacity; the default (1024) is smaller than the
        # 1500 encoder positions every Whisper request produces.
        max_input_len=1500,
        max_num_tokens=3000,
        scheduler_config=SchedulerConfig(use_python_scheduler=True),
        tensor_parallel_size=tensor_parallel_size,
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


def test_whisper_pytorch_beam_search(monkeypatch):
    """Beam-2 transcription (cross-KV shared across beams)."""
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    model_path = _get_whisper_model_path()
    wave, sample_rate = soundfile.read(_get_audio_path())

    with _make_llm(model_path, max_beam_width=2) as llm:
        beam_params = SamplingParams(
            best_of=2, n=1, temperature=0.0, use_beam_search=True, max_tokens=_MAX_NEW_TOKENS
        )
        outputs = llm.generate([_audio_prompt(wave, sample_rate)], beam_params)
        assert _EXPECTED_TRANSCRIPT_FRAGMENT in outputs[0].outputs[0].text.lower()


def _assert_decoder_cuda_graph_state(llm: LLM, captured: bool) -> None:
    """Introspect the in-process engine (single-process mode only).

    Decoder graphs captured (or not); the enc-dec encoder step stays eager.
    """
    model_engine = llm._executor.engine.model_engine
    assert not model_engine.encoder_cuda_graph_runner.enabled
    assert not model_engine.encoder_cuda_graph_runner.graphs
    assert model_engine.cuda_graph_runner.enabled == captured
    assert bool(model_engine.cuda_graph_runner.graphs) == captured


# Feature-combination matrix mirroring the T5/BART enc-dec coverage. Cases:
# (torch_dtype override or None for checkpoint fp32, kv manager v2, decoder
# cuda-graph batch sizes, graphs must capture, TP size). KVCacheManagerV2
# requires beam width 1, so v2 rides greedy; the fp32+graphs-requested case
# asserts the engine declines graphs (fp32 enc-dec guard) yet stays exact.
_FEATURE_COMBINATION_CASES = [
    pytest.param(None, True, None, False, 1, id="fp32-kv-v2-graphs-off-greedy"),
    pytest.param(None, False, [1, 2], False, 1, id="fp32-kv-v1-graphs-requested-greedy"),
    pytest.param("bfloat16", False, [1, 2], True, 1, id="bf16-kv-v1-decoder-graphs-on-greedy"),
    pytest.param("bfloat16", True, None, False, 1, id="bf16-kv-v2-graphs-off-greedy"),
    pytest.param("float16", False, None, False, 1, id="fp16-kv-v1-graphs-off-greedy"),
    pytest.param(
        None,
        False,
        None,
        False,
        2,
        id="fp32-kv-v1-graphs-off-greedy-tp2",
        marks=pytest.mark.skip_less_device(2),
    ),
]


@pytest.mark.parametrize(
    "torch_dtype,use_kv_cache_manager_v2,cuda_graph_batch_sizes,graphs_captured,tp_size",
    _FEATURE_COMBINATION_CASES,
)
def test_whisper_pytorch_feature_combinations(
    monkeypatch,
    torch_dtype,
    use_kv_cache_manager_v2,
    cuda_graph_batch_sizes,
    graphs_captured,
    tp_size,
):
    """Greedy transcription across dtype/kv-cache-manager/CUDA-graph/TP combos.

    Batch-1 and batch-2 must both reproduce the pinned per-dtype token ids
    (fp32 exact; fp16/bf16 differ only in the borderline first token).
    """
    if tp_size == 1:
        monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    expected_token_ids = (_EXPECTED_GREEDY_OUTPUT_TOKEN_IDS if torch_dtype is None
                          else _EXPECTED_GREEDY_OUTPUT_TOKEN_IDS_HALF)

    model_path = _get_whisper_model_path()
    wave, sample_rate = soundfile.read(_get_audio_path())
    sampling_params = SamplingParams(temperature=0.0, max_tokens=_MAX_NEW_TOKENS)

    llm = _make_llm(
        model_path,
        use_kv_cache_manager_v2=use_kv_cache_manager_v2,
        torch_dtype=torch_dtype,
        cuda_graph_batch_sizes=cuda_graph_batch_sizes,
        tensor_parallel_size=tp_size,
    )
    with llm:
        for batch_size in (1, 2):
            outputs = llm.generate(
                [_audio_prompt(wave, sample_rate) for _ in range(batch_size)],
                sampling_params,
            )
            for output in outputs:
                completion = output.outputs[0]
                assert list(completion.token_ids) == expected_token_ids
                assert _EXPECTED_TRANSCRIPT_FRAGMENT in completion.text.lower()

        if tp_size == 1:
            _assert_decoder_cuda_graph_state(llm, captured=graphs_captured)
