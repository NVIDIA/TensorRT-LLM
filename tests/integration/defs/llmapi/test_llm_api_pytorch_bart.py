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

from pathlib import Path

import pytest
from transformers import AutoTokenizer

from tensorrt_llm.llmapi import (
    LLM,
    CudaGraphConfig,
    KvCacheConfig,
    RequestOutput,
    SamplingParams,
    SchedulerConfig,
)

from ..conftest import llm_models_root

_SOURCE_TEXT = (
    "Summarize: NVIDIA builds fast inference software for large language models. "
    "TensorRT-LLM supports encoder-decoder models such as BART and T5."
)
_MIXED_ENCODER_SOURCE_TEXTS = [
    _SOURCE_TEXT,
    (
        "Summarize: The city opened a new public library on Monday. Residents said "
        "the library has quiet rooms, computer access, and a large children section."
    ),
]
_MODEL_NAME = "bart-large-cnn"
_MAX_NEW_TOKENS = 8
_MAX_SEQUENCE_LENGTH = 128
_MAX_KV_TOKENS = 384
_MIN_GPU_MEMORY_MB = 16_000
_FREE_GPU_MEMORY_FRACTION = 0.2
_CROSS_KV_CACHE_FRACTION = 0.5
_EXPECTED_GREEDY_OUTPUT_TOKEN_IDS = [0, 565, 35354, 13963, 12, 6006, 448, 2]
_EXPECTED_TEXT_FRAGMENT = "TensorRT"
_MIXED_ENCODER_EXPECTED_TEXT_FRAGMENTS = [
    _EXPECTED_TEXT_FRAGMENT,
    "library",
]


def _test_case(
    torch_dtype: str,
    use_kv_cache_manager_v2: bool,
    enable_cuda_graph: bool,
    num_beams: int,
    num_return_sequences: int,
    exact_match: bool,
    feature_id: str,
):
    expected_output_token_ids = [_EXPECTED_GREEDY_OUTPUT_TOKEN_IDS] if num_beams == 1 else None
    assert not exact_match or expected_output_token_ids is not None

    return pytest.param(
        expected_output_token_ids,
        torch_dtype,
        use_kv_cache_manager_v2,
        enable_cuda_graph,
        num_beams,
        num_return_sequences,
        exact_match,
        id=f"{feature_id}-{_MODEL_NAME}",
    )


_TEST_CASES = [
    _test_case(
        torch_dtype="bfloat16",
        use_kv_cache_manager_v2=False,
        enable_cuda_graph=False,
        num_beams=1,
        num_return_sequences=1,
        exact_match=True,
        feature_id="bf16-kv-v1-cuda-graph-off-greedy",
    ),
    _test_case(
        torch_dtype="float16",
        use_kv_cache_manager_v2=False,
        enable_cuda_graph=False,
        num_beams=1,
        num_return_sequences=1,
        exact_match=False,
        feature_id="fp16-kv-v1-cuda-graph-off-greedy",
    ),
    _test_case(
        torch_dtype="bfloat16",
        use_kv_cache_manager_v2=False,
        enable_cuda_graph=False,
        num_beams=2,
        num_return_sequences=2,
        exact_match=False,
        feature_id="bf16-kv-v1-cuda-graph-off-beam2",
    ),
    _test_case(
        torch_dtype="bfloat16",
        use_kv_cache_manager_v2=True,
        enable_cuda_graph=False,
        num_beams=1,
        num_return_sequences=1,
        exact_match=True,
        feature_id="bf16-kv-v2-cuda-graph-off-greedy",
    ),
]


def _mixed_batch_test_case(
    torch_dtype: str,
    use_kv_cache_manager_v2: bool,
    num_beams: int,
    num_return_sequences: int,
    feature_id: str,
):
    return pytest.param(
        torch_dtype,
        use_kv_cache_manager_v2,
        num_beams,
        num_return_sequences,
        id=f"{feature_id}-{_MODEL_NAME}",
    )


_MIXED_BATCH_TEST_CASES = [
    _mixed_batch_test_case(
        torch_dtype="bfloat16",
        use_kv_cache_manager_v2=False,
        num_beams=1,
        num_return_sequences=1,
        feature_id="bf16-kv-v1-cuda-graph-off-greedy-batch2",
    ),
    _mixed_batch_test_case(
        torch_dtype="bfloat16",
        use_kv_cache_manager_v2=True,
        num_beams=1,
        num_return_sequences=1,
        feature_id="bf16-kv-v2-cuda-graph-off-greedy-batch2",
    ),
]

pytestmark = [
    pytest.mark.skip_less_device(1),
    pytest.mark.skip_less_device_memory(_MIN_GPU_MEMORY_MB),
    pytest.mark.threadleak(enabled=False),
]


def _get_bart_model_path() -> str:
    try:
        models_root = Path(llm_models_root())
    except AssertionError as exc:
        pytest.skip(str(exc))

    model_path = models_root / _MODEL_NAME
    if not model_path.exists():
        pytest.skip(f"{_MODEL_NAME} is not available under {models_root}")
    return str(model_path)


def _sampling_params(num_beams: int, num_return_sequences: int) -> SamplingParams:
    if num_beams == 1:
        assert num_return_sequences == 1
        return SamplingParams(
            max_tokens=_MAX_NEW_TOKENS,
            temperature=0.0,
        )

    return SamplingParams(
        best_of=num_beams,
        max_tokens=_MAX_NEW_TOKENS,
        n=num_return_sequences,
        temperature=0.0,
        use_beam_search=True,
    )


def _cuda_graph_config(
    enabled: bool,
    batch_sizes: list[int] | None = None,
) -> CudaGraphConfig | None:
    return CudaGraphConfig(batch_sizes=batch_sizes or [1]) if enabled else None


def _assert_bart_response(
    response: RequestOutput,
    num_return_sequences: int,
) -> list[list[int]]:
    assert response.finished

    assert len(response.outputs) == num_return_sequences
    token_ids_by_output = []
    for output in response.outputs:
        assert output.token_ids is not None
        assert 0 < len(output.token_ids) <= _MAX_NEW_TOKENS
        token_ids_by_output.append(output.token_ids)
    return token_ids_by_output


def _print_generated_text(
    tokenizer, case_id: str, label: str, token_ids_by_output: list[list[int]]
) -> None:
    for output_idx, token_ids in enumerate(token_ids_by_output):
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"{case_id} {label}[{output_idx}]: {text!r} token_ids={token_ids}")


def _assert_expected_generation(
    tokenizer,
    token_ids_by_output: list[list[int]],
    exact_match: bool,
    expected_token_ids_by_output: list[list[int]] | None,
    expected_text_fragment: str | None = _EXPECTED_TEXT_FRAGMENT,
) -> None:
    decoded_text_by_output = [
        tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in token_ids_by_output
    ]
    assert all(decoded_text_by_output)
    if expected_token_ids_by_output is None:
        if expected_text_fragment is not None:
            assert all(expected_text_fragment in text for text in decoded_text_by_output)
    else:
        assert token_ids_by_output[0] == expected_token_ids_by_output[0]
    if len(token_ids_by_output) > 1:
        assert len({tuple(token_ids) for token_ids in token_ids_by_output}) == len(
            token_ids_by_output
        )
    if not exact_match:
        return

    assert expected_token_ids_by_output is not None
    assert token_ids_by_output == expected_token_ids_by_output


@pytest.mark.parametrize(
    "expected_output_token_ids_by_output,torch_dtype,use_kv_cache_manager_v2,"
    "enable_cuda_graph,num_beams,num_return_sequences,exact_match",
    _TEST_CASES,
)
def test_bart_pytorch_generate_encoder_decoder_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    expected_output_token_ids_by_output: list[list[int]] | None,
    torch_dtype: str,
    use_kv_cache_manager_v2: bool,
    enable_cuda_graph: bool,
    num_beams: int,
    num_return_sequences: int,
    exact_match: bool,
) -> None:
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    monkeypatch.setenv("TRTLLM_SKIP_KV_CACHE_ESTIMATION", "1")

    model_path = _get_bart_model_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    case_id = (
        f"model={_MODEL_NAME}, dtype={torch_dtype}, kv_v2={use_kv_cache_manager_v2}, "
        f"cuda_graph={enable_cuda_graph}, beams={num_beams}, returns={num_return_sequences}"
    )
    sampling_params = _sampling_params(num_beams, num_return_sequences)

    with LLM(
        model_path,
        backend="pytorch",
        attn_backend="TRTLLM",
        cuda_graph_config=_cuda_graph_config(enable_cuda_graph),
        disable_overlap_scheduler=True,
        dtype=torch_dtype,
        enable_chunked_prefill=False,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=_MAX_KV_TOKENS,
            free_gpu_memory_fraction=_FREE_GPU_MEMORY_FRACTION,
            cross_kv_cache_fraction=_CROSS_KV_CACHE_FRACTION,
            use_kv_cache_manager_v2=use_kv_cache_manager_v2,
        ),
        max_batch_size=1,
        max_beam_width=num_beams,
        max_input_len=_MAX_SEQUENCE_LENGTH,
        max_num_tokens=_MAX_SEQUENCE_LENGTH,
        max_seq_len=_MAX_SEQUENCE_LENGTH,
        model_kwargs={"torch_dtype": torch_dtype},
        scheduler_config=SchedulerConfig(use_python_scheduler=True),
    ) as llm:
        response = llm.generate(
            _SOURCE_TEXT,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        token_ids = _assert_bart_response(
            response,
            num_return_sequences=num_return_sequences,
        )
        _print_generated_text(tokenizer, case_id, "output", token_ids)
        _assert_expected_generation(
            tokenizer,
            token_ids,
            exact_match,
            expected_output_token_ids_by_output,
        )


@pytest.mark.parametrize(
    "torch_dtype,use_kv_cache_manager_v2,num_beams,num_return_sequences",
    _MIXED_BATCH_TEST_CASES,
)
def test_bart_pytorch_generate_encoder_decoder_mixed_encoder_lengths_batch(
    monkeypatch: pytest.MonkeyPatch,
    torch_dtype: str,
    use_kv_cache_manager_v2: bool,
    num_beams: int,
    num_return_sequences: int,
) -> None:
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    monkeypatch.setenv("TRTLLM_SKIP_KV_CACHE_ESTIMATION", "1")

    model_path = _get_bart_model_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = _sampling_params(num_beams, num_return_sequences)
    case_id = (
        f"model={_MODEL_NAME}, dtype={torch_dtype}, kv_v2={use_kv_cache_manager_v2}, "
        f"cuda_graph=False, beams={num_beams}, returns={num_return_sequences}, "
        "mixed_encoder_lengths=True, batch_size=2"
    )
    with LLM(
        model_path,
        backend="pytorch",
        attn_backend="TRTLLM",
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
        dtype=torch_dtype,
        enable_chunked_prefill=False,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=_MAX_KV_TOKENS,
            free_gpu_memory_fraction=_FREE_GPU_MEMORY_FRACTION,
            cross_kv_cache_fraction=_CROSS_KV_CACHE_FRACTION,
            use_kv_cache_manager_v2=use_kv_cache_manager_v2,
        ),
        max_batch_size=len(_MIXED_ENCODER_SOURCE_TEXTS),
        max_beam_width=num_beams,
        max_input_len=_MAX_SEQUENCE_LENGTH,
        max_num_tokens=_MAX_SEQUENCE_LENGTH,
        max_seq_len=_MAX_SEQUENCE_LENGTH,
        model_kwargs={"torch_dtype": torch_dtype},
        scheduler_config=SchedulerConfig(use_python_scheduler=True),
    ) as llm:
        responses = llm.generate(
            _MIXED_ENCODER_SOURCE_TEXTS,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        assert len(responses) == len(_MIXED_ENCODER_SOURCE_TEXTS)

        for request_idx, response in enumerate(responses):
            token_ids = _assert_bart_response(
                response,
                num_return_sequences=num_return_sequences,
            )
            _print_generated_text(
                tokenizer,
                f"{case_id}, request={request_idx}",
                "output",
                token_ids,
            )
            _assert_expected_generation(
                tokenizer,
                token_ids,
                exact_match=False,
                expected_token_ids_by_output=None,
                expected_text_fragment=_MIXED_ENCODER_EXPECTED_TEXT_FRAGMENTS[request_idx],
            )
