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
    "Summarize: The engineering team released a faster inference service on Monday. "
    "The update improves batching, lowers latency, and adds detailed monitoring for operators."
)
_MIXED_ENCODER_SOURCE_TEXTS = [
    _SOURCE_TEXT,
    (
        "Summarize: The company opened a training center on Monday. Managers said "
        "the center adds classrooms, simulation labs, and career coaching for workers."
    ),
]
_MODEL_NAME = "bart-large-cnn"
_MAX_NEW_TOKENS = 10
_MAX_SEQUENCE_LENGTH = 128
_MAX_KV_TOKENS = 384
_MIN_GPU_MEMORY_MB = 16_000
_FREE_GPU_MEMORY_FRACTION = 0.2
_CROSS_KV_CACHE_FRACTION = 0.5
# "The update improves batching, lowers latency"
_EXPECTED_GREEDY_OUTPUT_TOKEN_IDS = [0, 133, 2935, 15296, 14398, 154, 6, 32222, 35940, 2]
_EXPECTED_BEAM_OUTPUT_TOKEN_IDS_BY_BEAMS = {
    2: [
        # "The update improves batching, lowers latency"
        [0, 133, 2935, 15296, 14398, 154, 6, 32222, 35940, 2],
        # "The update improves batching, lowers"
        [0, 0, 133, 2935, 15296, 14398, 154, 6, 32222, 2],
    ],
}
_MIXED_ENCODER_EXPECTED_TOKEN_IDS_BY_REQUEST = [
    # "The update improves batching, lowers latency"
    [[0, 133, 2935, 15296, 14398, 154, 6, 32222, 35940, 2]],
    # "The company opened a training center on Monday"
    [[0, 133, 138, 1357, 10, 1058, 1312, 15, 302, 2]],
]


def _test_case(
    torch_dtype: str,
    use_kv_cache_manager_v2: bool,
    enable_cuda_graph: bool,
    num_beams: int,
    num_return_sequences: int,
    exact_match: bool,
    feature_id: str,
    cuda_graph_batch_sizes: list[int] | None = None,
    kv_cache_dtype: str = "auto",
    tensor_parallel_size: int = 1,
    marks=None,
):
    expected_output_token_ids = (
        [_EXPECTED_GREEDY_OUTPUT_TOKEN_IDS]
        if num_beams == 1
        else _EXPECTED_BEAM_OUTPUT_TOKEN_IDS_BY_BEAMS[num_beams]
    )

    param_kwargs = {"id": f"{feature_id}-{_MODEL_NAME}"}
    if marks is not None:
        param_kwargs["marks"] = marks

    return pytest.param(
        expected_output_token_ids,
        torch_dtype,
        use_kv_cache_manager_v2,
        enable_cuda_graph,
        num_beams,
        num_return_sequences,
        exact_match,
        cuda_graph_batch_sizes,
        kv_cache_dtype,
        tensor_parallel_size,
        **param_kwargs,
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
        torch_dtype="bfloat16",
        use_kv_cache_manager_v2=False,
        enable_cuda_graph=True,
        num_beams=1,
        num_return_sequences=1,
        exact_match=True,
        cuda_graph_batch_sizes=[2],
        feature_id="bf16-kv-v1-cuda-graph-on-greedy",
    ),
    _test_case(
        torch_dtype="float16",
        use_kv_cache_manager_v2=False,
        enable_cuda_graph=False,
        num_beams=1,
        num_return_sequences=1,
        exact_match=True,
        feature_id="fp16-kv-v1-cuda-graph-off-greedy",
    ),
    _test_case(
        torch_dtype="bfloat16",
        use_kv_cache_manager_v2=False,
        enable_cuda_graph=False,
        num_beams=2,
        num_return_sequences=2,
        exact_match=True,
        feature_id="bf16-kv-v1-cuda-graph-off-beam2",
    ),
    _test_case(
        torch_dtype="bfloat16",
        use_kv_cache_manager_v2=False,
        enable_cuda_graph=True,
        num_beams=2,
        num_return_sequences=2,
        exact_match=True,
        feature_id="bf16-kv-v1-cuda-graph-on-beam2",
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
    _test_case(
        torch_dtype="bfloat16",
        use_kv_cache_manager_v2=True,
        enable_cuda_graph=True,
        num_beams=1,
        num_return_sequences=1,
        exact_match=True,
        feature_id="bf16-kv-v2-cuda-graph-on-greedy",
    ),
    # Tensor parallelism (TP=2) coverage
    _test_case(
        torch_dtype="bfloat16",
        use_kv_cache_manager_v2=False,
        enable_cuda_graph=False,
        num_beams=1,
        num_return_sequences=1,
        exact_match=True,
        tensor_parallel_size=2,
        feature_id="bf16-kv-v1-cuda-graph-off-greedy-tp2",
        marks=pytest.mark.skip_less_device(2),
    ),
    _test_case(
        torch_dtype="bfloat16",
        use_kv_cache_manager_v2=False,
        enable_cuda_graph=True,
        num_beams=1,
        num_return_sequences=1,
        exact_match=True,
        tensor_parallel_size=2,
        feature_id="bf16-kv-v1-cuda-graph-on-greedy-tp2",
        marks=pytest.mark.skip_less_device(2),
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
        feature_id="bf16-kv-v1-decoder-cuda-graph-on-greedy-batch2",
    ),
    _mixed_batch_test_case(
        torch_dtype="bfloat16",
        use_kv_cache_manager_v2=True,
        num_beams=1,
        num_return_sequences=1,
        feature_id="bf16-kv-v2-decoder-cuda-graph-on-greedy-batch2",
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


def _decoder_cuda_graph_config(
    batch_sizes: list[int] | None = None,
) -> CudaGraphConfig:
    # CudaGraphConfig is decode-only. It keeps encoder CUDA graphs disabled,
    # which is what mixed encoder-length tests want while still covering
    # decoder graph capture/replay.
    return CudaGraphConfig(
        batch_sizes=batch_sizes or [1],
        enable_padding=True,
    )


def _assert_decoder_cuda_graph_state(
    llm: LLM,
    enabled: bool,
    batch_sizes: list[int] | None,
) -> None:
    model_engine = llm._executor.engine.model_engine

    if not enabled:
        assert not model_engine.encoder_cuda_graph_runner.enabled
        assert not model_engine.cuda_graph_runner.enabled
        assert not model_engine.encoder_cuda_graph_runner.graphs
        assert not model_engine.cuda_graph_runner.graphs
        return

    _assert_decoder_cuda_graphs_captured(llm)
    if batch_sizes is not None:
        assert model_engine.cuda_graph_runner.padding_dummy_requests


def _assert_decoder_cuda_graphs_captured(llm: LLM) -> None:
    model_engine = llm._executor.engine.model_engine

    assert not model_engine.encoder_cuda_graph_runner.enabled
    assert not model_engine.encoder_cuda_graph_runner.graphs
    assert model_engine.cuda_graph_runner.enabled
    assert model_engine.cuda_graph_runner.graphs


def _assert_bart_response(
    response: RequestOutput,
    num_return_sequences: int,
    max_tokens: int = _MAX_NEW_TOKENS,
) -> list[list[int]]:
    assert response.finished

    assert len(response.outputs) == num_return_sequences
    token_ids_by_output = []
    for output in response.outputs:
        assert output.token_ids is not None
        assert 0 < len(output.token_ids) <= max_tokens
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
    expected_token_ids_by_output: list[list[int]],
) -> None:
    decoded_text_by_output = [
        tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in token_ids_by_output
    ]
    assert all(decoded_text_by_output)
    assert token_ids_by_output[0] == expected_token_ids_by_output[0]
    if len(token_ids_by_output) > 1:
        assert len({tuple(token_ids) for token_ids in token_ids_by_output}) == len(
            token_ids_by_output
        )
    if not exact_match:
        return

    assert token_ids_by_output == expected_token_ids_by_output


def _run_bart_pytorch_generate_encoder_decoder(
    monkeypatch: pytest.MonkeyPatch,
    expected_output_token_ids_by_output: list[list[int]],
    torch_dtype: str,
    use_kv_cache_manager_v2: bool,
    enable_cuda_graph: bool,
    num_beams: int,
    num_return_sequences: int,
    exact_match: bool,
    cuda_graph_batch_sizes: list[int] | None,
    kv_cache_dtype: str = "auto",
    tensor_parallel_size: int = 1,
) -> None:
    if tensor_parallel_size == 1:
        monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    monkeypatch.setenv("TRTLLM_SKIP_KV_CACHE_ESTIMATION", "1")

    model_path = _get_bart_model_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    case_id = (
        f"model={_MODEL_NAME}, dtype={torch_dtype}, kv_v2={use_kv_cache_manager_v2}, "
        f"cuda_graph={enable_cuda_graph}, beams={num_beams}, returns={num_return_sequences}, "
        f"kv_dtype={kv_cache_dtype}, tp={tensor_parallel_size}"
    )
    sampling_params = _sampling_params(num_beams, num_return_sequences)

    with LLM(
        model_path,
        backend="pytorch",
        attn_backend="TRTLLM",
        cuda_graph_config=_decoder_cuda_graph_config(cuda_graph_batch_sizes)
        if enable_cuda_graph
        else None,
        disable_overlap_scheduler=True,
        dtype=torch_dtype,
        enable_chunked_prefill=False,
        tensor_parallel_size=tensor_parallel_size,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=_MAX_KV_TOKENS,
            free_gpu_memory_fraction=_FREE_GPU_MEMORY_FRACTION,
            cross_kv_cache_fraction=_CROSS_KV_CACHE_FRACTION,
            use_kv_cache_manager_v2=use_kv_cache_manager_v2,
            dtype=kv_cache_dtype,
        ),
        max_batch_size=max(cuda_graph_batch_sizes or [1]),
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
        # CUDA graph state introspection reaches into the in-process engine,
        # which is only available when the executor runs single-process (TP=1).
        # For TP>1 the executor is a multi-process proxy without a local engine,
        # so we rely on the generated-output assertions above for correctness.
        if tensor_parallel_size == 1:
            _assert_decoder_cuda_graph_state(
                llm,
                enable_cuda_graph,
                cuda_graph_batch_sizes,
            )


@pytest.mark.parametrize(
    "expected_output_token_ids_by_output,torch_dtype,use_kv_cache_manager_v2,"
    "enable_cuda_graph,num_beams,num_return_sequences,exact_match,cuda_graph_batch_sizes,"
    "kv_cache_dtype,tensor_parallel_size",
    _TEST_CASES,
)
def test_bart_pytorch_generate_encoder_decoder_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    expected_output_token_ids_by_output: list[list[int]],
    torch_dtype: str,
    use_kv_cache_manager_v2: bool,
    enable_cuda_graph: bool,
    num_beams: int,
    num_return_sequences: int,
    exact_match: bool,
    cuda_graph_batch_sizes: list[int] | None,
    kv_cache_dtype: str,
    tensor_parallel_size: int,
) -> None:
    _run_bart_pytorch_generate_encoder_decoder(
        monkeypatch,
        expected_output_token_ids_by_output,
        torch_dtype,
        use_kv_cache_manager_v2,
        enable_cuda_graph,
        num_beams,
        num_return_sequences,
        exact_match,
        cuda_graph_batch_sizes,
        kv_cache_dtype,
        tensor_parallel_size,
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
        f"decoder_cuda_graph=True, beams={num_beams}, returns={num_return_sequences}, "
        "mixed_encoder_lengths=True, batch_size=2"
    )
    with LLM(
        model_path,
        backend="pytorch",
        attn_backend="TRTLLM",
        cuda_graph_config=_decoder_cuda_graph_config([2]),
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
                exact_match=True,
                expected_token_ids_by_output=_MIXED_ENCODER_EXPECTED_TOKEN_IDS_BY_REQUEST[
                    request_idx
                ],
            )

        _assert_decoder_cuda_graphs_captured(llm)
