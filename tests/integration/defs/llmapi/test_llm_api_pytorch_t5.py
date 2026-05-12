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
from transformers import AutoConfig, AutoTokenizer

from tensorrt_llm.llmapi import (
    LLM,
    CudaGraphConfig,
    KvCacheConfig,
    RequestOutput,
    SamplingParams,
    SchedulerConfig,
)

from ..conftest import llm_models_root

_SOURCE_TEXT = "translate English to German: The house is wonderful."
_MIXED_ENCODER_SOURCE_TEXTS = [
    _SOURCE_TEXT,
    "translate English to German: The book is on the table.",
]
_MAX_NEW_TOKENS = 4
_MAX_SEQUENCE_LENGTH = 64
_MAX_KV_TOKENS = 256
_MIN_GPU_MEMORY_MB = 16_000
_FLAN_T5_XXL_MIN_GPU_MEMORY_MB = 80_000
_FREE_GPU_MEMORY_FRACTION = 0.2
_CROSS_KV_CACHE_FRACTION = 0.5
_EXPECTED_TRANSLATION_FRAGMENT = "Haus"
_EXPECTED_OUTPUT_TOKEN_IDS_BY_MODEL = {
    "t5-small": [644, 4598, 229, 19250],
    "t5-base": [644, 4598, 229, 19250],
    "t5-large": [644, 4598, 229, 19250],
    "flan-t5-small": [644, 4598, 229, 9685],
    "byt5-small": [258, 35, 119, 114],
}
# Known HF references for returned beam hypotheses. The tests exact-match greedy
# outputs and the best beam when a reference is available; lower-ranked BF16
# alternatives can differ on very close scores, so beam tests also assert that
# all requested outputs are present, non-empty, and distinct.
_HF_BEAM_OUTPUT_TOKEN_IDS_BY_MODEL_AND_BEAMS = {
    ("t5-small", 2): [
        [644, 4598, 229, 19250],
        [644, 4598, 229, 3],
    ],
    ("t5-base", 2): [
        [644, 4598, 229, 19250],
        [644, 4598, 229, 3],
    ],
    ("t5-large", 2): [
        [644, 4598, 229, 19250],
        [644, 4598, 229, 3],
    ],
    ("flan-t5-small", 2): [
        [644, 4598, 229, 9685],
        [644, 4598, 229, 19250],
    ],
}
_MIXED_ENCODER_OUTPUT_TOKEN_IDS_BY_MODEL_AND_BEAMS = {
    ("t5-small", 1): [
        [[644, 4598, 229, 19250]],
        [[644, 4675, 4186, 219]],
    ],
    ("t5-small", 2): [
        _HF_BEAM_OUTPUT_TOKEN_IDS_BY_MODEL_AND_BEAMS[("t5-small", 2)],
        [
            [644, 4675, 229, 219],
            [644, 4675, 4186, 219],
        ],
    ],
    ("flan-t5-small", 2): [
        _HF_BEAM_OUTPUT_TOKEN_IDS_BY_MODEL_AND_BEAMS[("flan-t5-small", 2)],
        [
            [316, 4675, 229, 219],
            [316, 4675, 229, 256],
        ],
    ],
}
_MIXED_ENCODER_EXPECTED_TEXT_FRAGMENTS_BY_MODEL = {
    "t5-small": [_EXPECTED_TRANSLATION_FRAGMENT, "Buch"],
    "flan-t5-small": [_EXPECTED_TRANSLATION_FRAGMENT, "Buch"],
}


def _test_case(
    model_name: str,
    torch_dtype: str,
    use_kv_cache_manager_v2: bool,
    enable_cuda_graph: bool,
    num_beams: int,
    num_return_sequences: int,
    exact_match: bool,
    feature_id: str,
    marks=(),
):
    if num_beams == 1:
        expected_output_token_ids = (
            [_EXPECTED_OUTPUT_TOKEN_IDS_BY_MODEL[model_name]]
            if model_name in _EXPECTED_OUTPUT_TOKEN_IDS_BY_MODEL
            else None
        )
    elif num_return_sequences == num_beams:
        expected_output_token_ids = _HF_BEAM_OUTPUT_TOKEN_IDS_BY_MODEL_AND_BEAMS.get(
            (model_name, num_beams)
        )
    else:
        expected_output_token_ids = (
            [_EXPECTED_OUTPUT_TOKEN_IDS_BY_MODEL[model_name]]
            if model_name in _EXPECTED_OUTPUT_TOKEN_IDS_BY_MODEL
            else None
        )

    assert not exact_match or expected_output_token_ids is not None

    return pytest.param(
        model_name,
        expected_output_token_ids,
        torch_dtype,
        use_kv_cache_manager_v2,
        enable_cuda_graph,
        num_beams,
        num_return_sequences,
        exact_match,
        id=f"{feature_id}-{model_name}",
        marks=marks,
    )


_TEST_CASES = [
    # Primary coverage: v1 cache manager, CUDA graph, and beam search.
    _test_case("t5-small", "bfloat16", False, True, 2, 2, False, "bf16-kv-v1-cuda-graph-on-beam2"),
    _test_case(
        "flan-t5-small", "bfloat16", False, True, 2, 2, False, "bf16-kv-v1-cuda-graph-on-beam2"
    ),
    _test_case("t5-base", "bfloat16", False, True, 2, 2, False, "bf16-kv-v1-cuda-graph-on-beam2"),
    _test_case("t5-large", "bfloat16", False, True, 2, 2, False, "bf16-kv-v1-cuda-graph-on-beam2"),
    _test_case(
        "flan-t5-base", "bfloat16", False, True, 2, 2, False, "bf16-kv-v1-cuda-graph-on-beam2"
    ),
    _test_case(
        "flan-t5-large", "bfloat16", False, True, 2, 2, False, "bf16-kv-v1-cuda-graph-on-beam2"
    ),
    _test_case(
        "flan-t5-xl", "bfloat16", False, True, 2, 2, False, "bf16-kv-v1-cuda-graph-on-beam2"
    ),
    _test_case(
        "flan-t5-xxl",
        "bfloat16",
        False,
        True,
        2,
        2,
        False,
        "bf16-kv-v1-cuda-graph-on-beam2",
        marks=pytest.mark.skip_less_device_memory(_FLAN_T5_XXL_MIN_GPU_MEMORY_MB),
    ),
    # Non-CUDA-graph smoke for the same v1 beam path.
    _test_case(
        "t5-small", "bfloat16", False, False, 2, 2, False, "bf16-kv-v1-cuda-graph-off-beam2"
    ),
    # Greedy smoke for the priority v1 CUDA graph path.
    _test_case("t5-small", "bfloat16", False, True, 1, 1, True, "bf16-kv-v1-cuda-graph-on-greedy"),
    # Precision coverage for beam search. KVCacheManagerV2 currently requires
    # max_beam_width == 1, so beam-search precision coverage uses v1.
    _test_case("t5-small", "float16", False, True, 2, 2, False, "fp16-kv-v1-cuda-graph-on-beam2"),
    _test_case("t5-small", "float32", False, True, 2, 2, False, "fp32-kv-v1-cuda-graph-on-beam2"),
    _test_case(
        "flan-t5-small", "float16", False, True, 2, 2, False, "fp16-kv-v1-cuda-graph-on-beam2"
    ),
    _test_case(
        "flan-t5-small", "float32", False, True, 2, 2, False, "fp32-kv-v1-cuda-graph-on-beam2"
    ),
    # Precision coverage for v2 on its supported CUDA graph path.
    _test_case("t5-small", "bfloat16", True, True, 1, 1, True, "bf16-kv-v2-cuda-graph-on-greedy"),
    _test_case("t5-small", "float16", True, True, 1, 1, True, "fp16-kv-v2-cuda-graph-on-greedy"),
    _test_case("t5-small", "float32", True, True, 1, 1, True, "fp32-kv-v2-cuda-graph-on-greedy"),
    _test_case(
        "flan-t5-small", "bfloat16", True, True, 1, 1, True, "bf16-kv-v2-cuda-graph-on-greedy"
    ),
    _test_case(
        "flan-t5-small", "float16", True, True, 1, 1, True, "fp16-kv-v2-cuda-graph-on-greedy"
    ),
    _test_case(
        "flan-t5-small", "float32", True, True, 1, 1, True, "fp32-kv-v2-cuda-graph-on-greedy"
    ),
    # ByT5 sanity coverage keeps the known-stable expected output path.
    _test_case(
        "byt5-small", "bfloat16", True, False, 1, 1, True, "bf16-kv-v2-cuda-graph-off-greedy"
    ),
]


def _mixed_batch_test_case(
    model_name: str,
    torch_dtype: str,
    use_kv_cache_manager_v2: bool,
    num_beams: int,
    num_return_sequences: int,
    exact_match: bool,
    feature_id: str,
    marks=(),
):
    expected_output_token_ids_by_request = (
        _MIXED_ENCODER_OUTPUT_TOKEN_IDS_BY_MODEL_AND_BEAMS.get((model_name, num_beams))
        if exact_match or num_beams > 1
        else None
    )
    assert not exact_match or expected_output_token_ids_by_request is not None

    return pytest.param(
        model_name,
        expected_output_token_ids_by_request,
        torch_dtype,
        use_kv_cache_manager_v2,
        num_beams,
        num_return_sequences,
        exact_match,
        id=f"{feature_id}-{model_name}",
        marks=marks,
    )


_MIXED_BATCH_TEST_CASES = [
    _mixed_batch_test_case(
        "t5-small",
        "bfloat16",
        False,
        2,
        2,
        False,
        "bf16-kv-v1-cuda-graph-on-beam2-batch2",
    ),
    _mixed_batch_test_case(
        "flan-t5-small",
        "bfloat16",
        False,
        2,
        2,
        False,
        "bf16-kv-v1-cuda-graph-on-beam2-batch2",
    ),
    _mixed_batch_test_case(
        "t5-small",
        "bfloat16",
        False,
        1,
        1,
        True,
        "bf16-kv-v1-cuda-graph-on-greedy-batch2",
    ),
    _mixed_batch_test_case(
        "t5-small",
        "bfloat16",
        True,
        1,
        1,
        True,
        "bf16-kv-v2-cuda-graph-on-greedy-batch2",
    ),
]

pytestmark = [
    pytest.mark.skip_less_device(1),
    pytest.mark.skip_less_device_memory(_MIN_GPU_MEMORY_MB),
    pytest.mark.threadleak(enabled=False),
]


def _get_t5_model_path(model_name: str) -> str:
    try:
        models_root = Path(llm_models_root())
    except AssertionError as exc:
        pytest.skip(str(exc))

    model_path = models_root / model_name
    if not model_path.exists():
        pytest.skip(f"{model_name} is not available under {models_root}")
    return str(model_path)


def _sampling_params(num_beams: int, num_return_sequences: int) -> SamplingParams:
    if num_beams == 1:
        assert num_return_sequences == 1
        return SamplingParams(
            max_tokens=_MAX_NEW_TOKENS,
            return_encoder_output=True,
            temperature=0.0,
        )

    return SamplingParams(
        best_of=num_beams,
        max_tokens=_MAX_NEW_TOKENS,
        n=num_return_sequences,
        return_encoder_output=True,
        temperature=0.0,
        use_beam_search=True,
    )


def _cuda_graph_config(
    enabled: bool,
    batch_sizes: list[int] | None = None,
) -> CudaGraphConfig | None:
    return CudaGraphConfig(batch_sizes=batch_sizes or [1]) if enabled else None


def _assert_t5_response(
    response: RequestOutput,
    encoder_input_len: int,
    hidden_size: int,
    num_return_sequences: int,
) -> list[list[int]]:
    assert response.finished
    assert response.encoder_output is not None
    assert response.encoder_output.device.type == "cpu"
    assert tuple(response.encoder_output.shape) == (encoder_input_len, hidden_size)

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
    expected_text_fragment: str = _EXPECTED_TRANSLATION_FRAGMENT,
) -> None:
    decoded_text_by_output = [
        tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in token_ids_by_output
    ]
    assert all(decoded_text_by_output)
    if expected_token_ids_by_output is None:
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
    "model_name,expected_output_token_ids_by_output,torch_dtype,use_kv_cache_manager_v2,"
    "enable_cuda_graph,num_beams,num_return_sequences,exact_match",
    _TEST_CASES,
)
def test_t5_pytorch_generate_encoder_decoder_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
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

    model_path = _get_t5_model_path(model_name)
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoder_input_token_ids = tokenizer(_SOURCE_TEXT, add_special_tokens=True)["input_ids"]
    decoder_start_token_id = config.decoder_start_token_id
    assert decoder_start_token_id is not None
    case_id = (
        f"model={model_name}, dtype={torch_dtype}, kv_v2={use_kv_cache_manager_v2}, "
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
        text_response = llm.generate(
            {
                "encoder_inputs": _SOURCE_TEXT,
            },
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        text_token_ids = _assert_t5_response(
            text_response,
            encoder_input_len=len(encoder_input_token_ids),
            hidden_size=config.d_model,
            num_return_sequences=num_return_sequences,
        )
        _print_generated_text(tokenizer, case_id, "encoder_inputs output", text_token_ids)
        _assert_expected_generation(
            tokenizer,
            text_token_ids,
            exact_match,
            expected_output_token_ids_by_output,
        )

        explicit_token_response = llm.generate(
            {
                "encoder_input_token_ids": encoder_input_token_ids,
                "decoder_input_token_ids": [decoder_start_token_id],
            },
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        explicit_token_ids = _assert_t5_response(
            explicit_token_response,
            encoder_input_len=len(encoder_input_token_ids),
            hidden_size=config.d_model,
            num_return_sequences=num_return_sequences,
        )
        _print_generated_text(tokenizer, case_id, "explicit token output", explicit_token_ids)
        _assert_expected_generation(
            tokenizer,
            explicit_token_ids,
            exact_match,
            expected_output_token_ids_by_output,
        )

    assert explicit_token_ids == text_token_ids


@pytest.mark.parametrize(
    "model_name,expected_output_token_ids_by_request,torch_dtype,use_kv_cache_manager_v2,"
    "num_beams,num_return_sequences,exact_match",
    _MIXED_BATCH_TEST_CASES,
)
def test_t5_pytorch_generate_encoder_decoder_cuda_graph_mixed_encoder_lengths_batch(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    expected_output_token_ids_by_request: list[list[list[int]] | None] | None,
    torch_dtype: str,
    use_kv_cache_manager_v2: bool,
    num_beams: int,
    num_return_sequences: int,
    exact_match: bool,
) -> None:
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    monkeypatch.setenv("TRTLLM_SKIP_KV_CACHE_ESTIMATION", "1")

    model_path = _get_t5_model_path(model_name)
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    decoder_start_token_id = config.decoder_start_token_id
    assert decoder_start_token_id is not None
    sampling_params = _sampling_params(num_beams, num_return_sequences)
    case_id = (
        f"model={model_name}, dtype={torch_dtype}, kv_v2={use_kv_cache_manager_v2}, "
        f"cuda_graph=True, beams={num_beams}, returns={num_return_sequences}, "
        "mixed_encoder_lengths=True, batch_size=2"
    )
    encoder_input_token_ids_by_request = [
        tokenizer(source_text, add_special_tokens=True)["input_ids"]
        for source_text in _MIXED_ENCODER_SOURCE_TEXTS
    ]

    with LLM(
        model_path,
        backend="pytorch",
        attn_backend="TRTLLM",
        cuda_graph_config=_cuda_graph_config(
            True, batch_sizes=[1, len(_MIXED_ENCODER_SOURCE_TEXTS)]
        ),
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
        text_responses = llm.generate(
            [{"encoder_inputs": source_text} for source_text in _MIXED_ENCODER_SOURCE_TEXTS],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        explicit_token_responses = llm.generate(
            [
                {
                    "encoder_input_token_ids": encoder_input_token_ids,
                    "decoder_input_token_ids": [decoder_start_token_id],
                }
                for encoder_input_token_ids in encoder_input_token_ids_by_request
            ],
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        assert len(text_responses) == len(_MIXED_ENCODER_SOURCE_TEXTS)
        assert len(explicit_token_responses) == len(_MIXED_ENCODER_SOURCE_TEXTS)

        for request_idx, encoder_input_token_ids in enumerate(encoder_input_token_ids_by_request):
            expected_token_ids = (
                None
                if expected_output_token_ids_by_request is None
                else expected_output_token_ids_by_request[request_idx]
            )
            expected_text_fragment = _MIXED_ENCODER_EXPECTED_TEXT_FRAGMENTS_BY_MODEL[model_name][
                request_idx
            ]

            text_response = text_responses[request_idx]
            text_token_ids = _assert_t5_response(
                text_response,
                encoder_input_len=len(encoder_input_token_ids),
                hidden_size=config.d_model,
                num_return_sequences=num_return_sequences,
            )
            _print_generated_text(
                tokenizer,
                f"{case_id}, request={request_idx}",
                "encoder_inputs output",
                text_token_ids,
            )
            _assert_expected_generation(
                tokenizer,
                text_token_ids,
                exact_match=exact_match,
                expected_token_ids_by_output=expected_token_ids,
                expected_text_fragment=expected_text_fragment,
            )

            explicit_token_response = explicit_token_responses[request_idx]
            explicit_token_ids = _assert_t5_response(
                explicit_token_response,
                encoder_input_len=len(encoder_input_token_ids),
                hidden_size=config.d_model,
                num_return_sequences=num_return_sequences,
            )
            _print_generated_text(
                tokenizer,
                f"{case_id}, request={request_idx}",
                "explicit token output",
                explicit_token_ids,
            )
            _assert_expected_generation(
                tokenizer,
                explicit_token_ids,
                exact_match=exact_match,
                expected_token_ids_by_output=expected_token_ids,
                expected_text_fragment=expected_text_fragment,
            )

            if expected_token_ids is not None:
                assert explicit_token_ids == text_token_ids
