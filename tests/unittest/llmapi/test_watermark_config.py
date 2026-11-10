# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public text-watermark configuration and request-transport contracts."""

import pickle
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tensorrt_llm._torch.pyexecutor._util import create_torch_sampler_args
from tensorrt_llm._torch.pyexecutor.llm_request import executor_request_to_llm_request
from tensorrt_llm._torch.pyexecutor.request_inputs import ExecutorRequest, PythonRequestInputs
from tensorrt_llm.bindings import executor as trtllm
from tensorrt_llm.llmapi.llm import LLM
from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig, TorchLlmArgs
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, CompletionRequest
from tensorrt_llm.usage.llmapi_config import build_capture_manifest
from tensorrt_llm.watermarking import WatermarkConfig

pytestmark = pytest.mark.cpu_only


def test_explicit_watermark_requires_engine_config_before_sampling_preparation() -> None:
    llm = SimpleNamespace(args=SimpleNamespace(watermark_config=None))
    with pytest.raises(ValueError, match="requires an engine watermark_config"):
        LLM._prepare_sampling_params(llm, SamplingParams(watermarking=True))


@pytest.mark.parametrize("key", [0, 2**64 - 1])
def test_watermark_config_unsigned_key_boundaries(key: int) -> None:
    config = WatermarkConfig(key=key)
    assert config.key == key
    assert config.algorithm == "gumbel"
    assert config.context_width == 4
    assert config.deduplicate_contexts_max_history == 8192
    assert config.alpha == 0.1


@pytest.mark.parametrize(
    "overrides",
    [
        {"key": -1},
        {"key": 2**64},
        {"key": True},
        {"key": "17"},
        {"key": 1.5},
        {"context_width": 0},
        {"context_width": 17},
        {"context_width": True},
        {"deduplicate_contexts_max_history": 0},
        {"deduplicate_contexts_max_history": 1.5},
        {"alpha": -0.01},
        {"alpha": 1.01},
        {"alpha": float("nan")},
        {"alpha": float("inf")},
        {"algorithm": "synthid"},
        {"unknown_option": True},
    ],
)
def test_watermark_config_rejects_invalid_values(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        WatermarkConfig(**({"key": 17} | overrides))


def test_watermark_config_requires_a_key() -> None:
    with pytest.raises(ValidationError):
        WatermarkConfig()


def test_watermark_key_is_private_in_repr_and_telemetry() -> None:
    config = WatermarkConfig(key=123456789012345)
    assert "123456789012345" not in repr(config)
    assert "key=" not in repr(config)
    assert all(entry.path != "key" for entry in build_capture_manifest(WatermarkConfig))
    assert all(
        not entry.path.startswith("watermark_config")
        for entry in build_capture_manifest(TorchLlmArgs)
    )
    # Worker transport and config revalidation must retain the secret.
    assert WatermarkConfig(**config.model_dump()).key == config.key
    assert pickle.loads(pickle.dumps(config)).key == config.key


def test_engine_watermark_config_defaults_and_speculative_rejection() -> None:
    assert TorchLlmArgs(model="test-model", skip_tokenizer_init=True).watermark_config is None
    args = TorchLlmArgs(model="test-model", skip_tokenizer_init=True, watermark_config={"key": 17})
    assert args.watermark_config == WatermarkConfig(key=17)
    with pytest.raises(ValidationError, match="does not support speculative decoding"):
        TorchLlmArgs(
            model="test-model",
            skip_tokenizer_init=True,
            watermark_config={"key": 17},
            speculative_config=MTPDecodingConfig(max_draft_len=1),
        )


def test_engine_watermark_config_rejects_disaggregated_serving() -> None:
    with pytest.raises(ValidationError, match="does not support disaggregated serving"):
        TorchLlmArgs(
            model="test-model",
            skip_tokenizer_init=True,
            watermark_config={"key": 17},
            cache_transceiver_config={},
        )


@pytest.mark.parametrize("value", [None, True, False])
@pytest.mark.parametrize("request_type", [CompletionRequest, ChatCompletionRequest])
def test_protocol_forwards_watermark_override(request_type: type, value: bool | None) -> None:
    content = (
        {"prompt": "Hello"}
        if request_type is CompletionRequest
        else {"messages": [{"role": "user", "content": "Hello"}]}
    )
    request = request_type(model="test-model", watermarking=value, **content)
    assert request.to_sampling_params().watermarking is value


@pytest.mark.parametrize("value", [0, 1, "true", "false", {}])
def test_protocol_and_sampling_params_reject_non_boolean_overrides(value: object) -> None:
    with pytest.raises(ValidationError):
        CompletionRequest(model="test-model", prompt="Hello", watermarking=value)
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="test-model", messages=[{"role": "user", "content": "Hello"}], watermarking=value
        )
    with pytest.raises(ValueError, match="watermarking must be a bool or None"):
        SamplingParams(watermarking=value)


def test_sampling_params_rejects_explicit_watermark_beam_search() -> None:
    with pytest.raises(ValueError, match="does not support beam search"):
        SamplingParams(watermarking=True, use_beam_search=True, n=2)
    assert SamplingParams(watermarking=False, use_beam_search=True, n=2).watermarking is False
    assert SamplingParams().watermarking is None


@pytest.mark.parametrize("value", [None, True, False])
def test_watermark_override_survives_worker_transport_and_children(value: bool | None) -> None:
    envelope = ExecutorRequest(
        trtllm.Request(
            input_token_ids=[1, 2, 3, 4],
            max_tokens=2,
            sampling_config=trtllm.SamplingConfig(num_return_sequences=2),
        ),
        PythonRequestInputs(watermarking=value),
    )
    transported = pickle.loads(pickle.dumps(envelope))
    parent = executor_request_to_llm_request(42, transported, [43], False)
    for request in (parent, *parent.child_requests):
        assert request.py_watermarking is value


def test_sampler_factory_preserves_engine_watermark_config() -> None:
    config = WatermarkConfig(key=17)
    args = create_torch_sampler_args(
        max_seq_len=128,
        speculative_config=None,
        max_beam_width=1,
        disable_overlap_scheduler=False,
        enable_async_worker=False,
        enable_speculative_beam_history_d2h=False,
        max_num_sequences=4,
        watermark_config=config,
    )
    assert args.watermark_config is config
