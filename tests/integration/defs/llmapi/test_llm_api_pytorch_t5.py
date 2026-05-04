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
_MAX_NEW_TOKENS = 4
_MAX_SEQUENCE_LENGTH = 64
_MAX_KV_TOKENS = 256
_MIN_GPU_MEMORY_MB = 16_000
_FREE_GPU_MEMORY_FRACTION = 0.2
_CROSS_KV_CACHE_FRACTION = 0.5
_EXPECTED_OUTPUT_TOKEN_IDS_BY_MODEL = {
    "t5-small": [644, 4598, 229, 19250],
    "flan-t5-small": [644, 4598, 229, 9685],
    "byt5-small": [258, 35, 119, 114],
}


def _test_case(
    model_name: str,
    torch_dtype: str,
    use_kv_cache_manager_v2: bool,
    enable_cuda_graph: bool,
    num_beams: int,
    exact_match: bool,
    feature_id: str,
):
    return pytest.param(
        model_name,
        _EXPECTED_OUTPUT_TOKEN_IDS_BY_MODEL[model_name],
        torch_dtype,
        use_kv_cache_manager_v2,
        enable_cuda_graph,
        num_beams,
        exact_match,
        id=f"{feature_id}-{model_name}",
    )


_TEST_CASES = [
    _test_case("t5-small", "bfloat16", True, False, 1, True, "bf16-kv-v2-cuda-graph-off-greedy"),
    _test_case("t5-small", "float16", True, False, 1, True, "fp16-kv-v2-cuda-graph-off-greedy"),
    _test_case("t5-small", "float32", True, False, 1, True, "fp32-kv-v2-cuda-graph-off-greedy"),
    _test_case("t5-small", "bfloat16", False, False, 1, True, "bf16-kv-v1-cuda-graph-off-greedy"),
    _test_case("t5-small", "bfloat16", True, True, 1, True, "bf16-kv-v2-cuda-graph-on-greedy"),
    _test_case("t5-small", "bfloat16", False, False, 2, False, "bf16-kv-v1-cuda-graph-off-beam2"),
    _test_case("t5-small", "bfloat16", False, False, 3, False, "bf16-kv-v1-cuda-graph-off-beam3"),
    _test_case("t5-small", "bfloat16", False, True, 2, False, "bf16-kv-v1-cuda-graph-on-beam2"),
    _test_case("t5-small", "bfloat16", False, True, 3, False, "bf16-kv-v1-cuda-graph-on-beam3"),
    _test_case(
        "flan-t5-small", "bfloat16", True, False, 1, True, "bf16-kv-v2-cuda-graph-off-greedy"
    ),
    _test_case(
        "flan-t5-small", "float32", True, False, 1, True, "fp32-kv-v2-cuda-graph-off-greedy"
    ),
    _test_case(
        "flan-t5-small", "bfloat16", False, False, 1, True, "bf16-kv-v1-cuda-graph-off-greedy"
    ),
    _test_case(
        "flan-t5-small", "bfloat16", False, False, 2, False, "bf16-kv-v1-cuda-graph-off-beam2"
    ),
    _test_case(
        "flan-t5-small", "bfloat16", False, False, 3, False, "bf16-kv-v1-cuda-graph-off-beam3"
    ),
    _test_case("byt5-small", "bfloat16", True, False, 1, True, "bf16-kv-v2-cuda-graph-off-greedy"),
    _test_case("byt5-small", "float32", True, False, 1, True, "fp32-kv-v2-cuda-graph-off-greedy"),
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


def _sampling_params(num_beams: int) -> SamplingParams:
    if num_beams == 1:
        return SamplingParams(
            max_tokens=_MAX_NEW_TOKENS,
            return_encoder_output=True,
            temperature=0.0,
        )

    return SamplingParams(
        best_of=num_beams,
        max_tokens=_MAX_NEW_TOKENS,
        return_encoder_output=True,
        temperature=0.0,
        use_beam_search=True,
    )


def _cuda_graph_config(enabled: bool) -> CudaGraphConfig | None:
    return CudaGraphConfig(batch_sizes=[1]) if enabled else None


def _assert_t5_response(
    response: RequestOutput, encoder_input_len: int, hidden_size: int
) -> list[int]:
    assert response.finished
    assert response.encoder_output is not None
    assert response.encoder_output.device.type == "cpu"
    assert tuple(response.encoder_output.shape) == (encoder_input_len, hidden_size)

    assert len(response.outputs) == 1
    output = response.outputs[0]
    assert output.token_ids is not None
    assert 0 < len(output.token_ids) <= _MAX_NEW_TOKENS
    return output.token_ids


def _print_generated_text(tokenizer, case_id: str, label: str, token_ids: list[int]) -> None:
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"{case_id} {label}: {text!r} token_ids={token_ids}")


def _assert_expected_generation(
    tokenizer, token_ids: list[int], exact_match: bool, expected_token_ids: list[int]
) -> None:
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    assert decoded_text
    if not exact_match:
        return

    assert token_ids == expected_token_ids


@pytest.mark.parametrize(
    "model_name,expected_output_token_ids,torch_dtype,use_kv_cache_manager_v2,"
    "enable_cuda_graph,num_beams,exact_match",
    _TEST_CASES,
)
def test_t5_pytorch_generate_encoder_decoder_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    expected_output_token_ids: list[int],
    torch_dtype: str,
    use_kv_cache_manager_v2: bool,
    enable_cuda_graph: bool,
    num_beams: int,
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
        f"cuda_graph={enable_cuda_graph}, beams={num_beams}"
    )
    sampling_params = _sampling_params(num_beams)

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
        )
        _print_generated_text(tokenizer, case_id, "encoder_inputs output", text_token_ids)
        _assert_expected_generation(
            tokenizer, text_token_ids, exact_match, expected_output_token_ids
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
        )
        _print_generated_text(tokenizer, case_id, "explicit token output", explicit_token_ids)
        _assert_expected_generation(
            tokenizer, explicit_token_ids, exact_match, expected_output_token_ids
        )

    assert explicit_token_ids == text_token_ids
