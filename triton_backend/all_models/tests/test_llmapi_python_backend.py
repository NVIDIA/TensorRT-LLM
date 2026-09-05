# Copyright 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import asyncio
import json
import sys
import threading
from dataclasses import dataclass
from typing import Dict, List, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Mock pb_utils
sys.modules["triton_python_backend_utils"] = MagicMock()

from helpers import (convert_request_input_to_dict,
                     get_lora_request_from_request,
                     get_output_config_from_request, get_parameter,
                     get_sampling_params_from_request,
                     get_streaming_from_request)
# Use PYTHONPATH=../llmapi/tensorrt_llm/1/
from model import TritonPythonModel


@dataclass
class MockTritonTensor:
    _name: str
    _tensor: Union[np.ndarray, torch.Tensor]

    def name(self) -> str:
        return self._name

    def as_numpy(self) -> np.ndarray:
        if self.is_cpu():
            return self._tensor
        else:
            return self._tensor.as_numpy()

    def is_cpu(self) -> bool:
        if isinstance(self._tensor, np.ndarray):
            return True
        else:
            return False

    def to_dlpack(self):
        if self.is_cpu():
            return self._tensor.__dlpack__()
        else:
            return self._tensor.to_dlpack()


@dataclass
class MockTritonError:
    message: str


class MockTritonModelException(Exception):
    """Stands in for pb_utils.TritonModelException in tests."""


@dataclass
class MockTritonResponse:
    tensors: Dict[str, MockTritonTensor]
    error: MockTritonError

    def __init__(self,
                 output_tensors: List[MockTritonTensor],
                 error: MockTritonError = None):
        self.tensors = {}
        for tensor in output_tensors:
            self.tensors[tensor.name()] = tensor
        self.error = error

    def output_tensors(self):
        return self.tensors.values()

    def has_error(self):
        return self.error is not None


@dataclass
class MockTritonRequest:
    tensors: Dict[str, MockTritonTensor]

    def get_input_tensor_by_name(self, name: str) -> MockTritonTensor:
        return self.tensors[name] if name in self.tensors else None

    def get_response_sender(self):
        return None


def mock_pb_utils_get_input_tensor_by_name_side_effect(
        request: MockTritonRequest, name: str) -> MockTritonTensor:
    return request.get_input_tensor_by_name(name)


def make_mock_triton_request(
        tensors: Dict[str, np.ndarray]) -> MockTritonRequest:
    return MockTritonRequest({
        k: MockTritonTensor(k, np.array(v))
        for k, v in tensors.items()
    })


@pytest.fixture(autouse=True)
def apply_patches():

    patch("model.pb_utils.Tensor", new=MockTritonTensor).start()
    patch("model.pb_utils.InferenceResponse", new=MockTritonResponse).start()
    patch("model.pb_utils.TritonError", new=MockTritonError).start()
    patch("model.pb_utils.InferenceRequest", new=MockTritonRequest).start()
    patch("model.pb_utils.get_input_tensor_by_name",
          new=mock_pb_utils_get_input_tensor_by_name_side_effect).start()
    patch("model.pb_utils.TritonModelException",
          new=MockTritonModelException).start()


def inputs(streaming=False):
    """Different request configurations for testing."""
    return {
        "text_input": ["Tell me a story."],
        "streaming": [streaming],
        "sampling_param_temperature": [0.8],
        "sampling_param_beam_width": [4],
        "sampling_param_top_k": [0],
        "sampling_param_top_p": [1.0],
        "sampling_param_stop": ['\n', 'stop'],
        "sampling_param_exclude_input_from_output": [True],
        "sampling_param_max_tokens": [100],
        "sampling_param_frequency_penalty": [0.0],
        "sampling_param_presence_penalty": [0.0],
        "sampling_param_seed": [2],
        "return_finish_reason": [True],
        "return_stop_reason": [True],
        "sampling_param_return_perf_metrics": [True]
    }


@pytest.fixture
def mock_model():
    model = TritonPythonModel()
    model.initialize({"model_config": json.dumps({"max_batch_size": 64})})
    return model


def test_get_sampling_params_from_request():
    request = make_mock_triton_request(inputs(streaming=False))
    config = get_sampling_params_from_request(request)
    assert config["temperature"] == 0.8
    # assert config["beam_width"] == 4
    assert config["top_k"] == 0
    assert config["top_p"] == 1.0
    assert config["max_tokens"] == 100
    assert config["frequency_penalty"] == 0.0
    assert config["presence_penalty"] == 0.0
    assert config["seed"] == 2
    assert config["return_perf_metrics"] == True
    assert np.array_equal(config["stop"], np.array(['\n', 'stop']))


def test_get_streaming_from_request():
    for streaming in [True, False]:
        request = make_mock_triton_request(inputs(streaming=streaming))
        assert get_streaming_from_request(request) == streaming


def test_get_output_config_from_request():
    request = make_mock_triton_request(inputs(streaming=False))
    output_config = get_output_config_from_request(request)
    assert output_config["return_finish_reason"] == True
    assert output_config["return_stop_reason"] == True


def test_convert_request_input_to_dict():
    request = make_mock_triton_request({
        "param_a": [1],
        "param_b": [True],
        "missing_param": [10]
    })

    param_mappings = {
        "param_a": "mapped_a",
        "param_b": "mapped_b",
        "non_existent": "mapped_c"
    }

    default_values = {"param_b": False, "non_existent": "default_value"}

    result = convert_request_input_to_dict(request=request,
                                           param_mappings=param_mappings,
                                           default_values=default_values,
                                           batch_size=1,
                                           batch_index=0)

    assert result == {
        "mapped_a": 1,
        "mapped_b": True,
        "mapped_c": "default_value"
    }


def test_get_lora_request_from_request(tmp_path):
    # Mock the deferred `from tensorrt_llm.executor.request import LoRARequest`
    # inside the helper so the test doesn't require a built TRT-LLM.
    fake_lora_request_cls = MagicMock()
    trtllm_mod = MagicMock()
    executor_mod = MagicMock()
    request_mod = MagicMock()
    request_mod.LoRARequest = fake_lora_request_cls
    # Real existing path so the helper's eager os.path.exists check passes.
    adapter_dir = str(tmp_path)

    with patch.dict(
            sys.modules, {
                "tensorrt_llm": trtllm_mod,
                "tensorrt_llm.executor": executor_mod,
                "tensorrt_llm.executor.request": request_mod,
            }):
        # No LoRA inputs -> returns None (backwards-compatible default)
        request = make_mock_triton_request({"text_input": ["hi"]})
        assert get_lora_request_from_request(request) is None

        # All three inputs (bytes STRING tensors, like dtype=object) ->
        # constructs LoRARequest with decoded strings + default ckpt source.
        fake_lora_request_cls.reset_mock()
        request = make_mock_triton_request({
            "lora_id": [42],
            "lora_name": [b"my-adapter"],
            "lora_path": [adapter_dir.encode("utf-8")],
        })
        result = get_lora_request_from_request(request)
        fake_lora_request_cls.assert_called_once_with(lora_name="my-adapter",
                                                      lora_int_id=42,
                                                      lora_path=adapter_dir,
                                                      lora_ckpt_source="hf")
        assert result is fake_lora_request_cls.return_value

        # Same inputs but unicode STRING tensors (dtype='<U...') -> still
        # decodes through _decode_string_scalar's str fall-through.
        fake_lora_request_cls.reset_mock()
        request = make_mock_triton_request({
            "lora_id": [42],
            "lora_name": ["unicode-adapter"],
            "lora_path": [adapter_dir],
        })
        get_lora_request_from_request(request)
        fake_lora_request_cls.assert_called_once_with(
            lora_name="unicode-adapter",
            lora_int_id=42,
            lora_path=adapter_dir,
            lora_ckpt_source="hf")

        # Explicit lora_ckpt_source="nemo" propagates through.
        fake_lora_request_cls.reset_mock()
        request = make_mock_triton_request({
            "lora_id": [42],
            "lora_name": [b"nemo-adapter"],
            "lora_path": [adapter_dir.encode("utf-8")],
            "lora_ckpt_source": [b"nemo"],
        })
        get_lora_request_from_request(request)
        fake_lora_request_cls.assert_called_once_with(lora_name="nemo-adapter",
                                                      lora_int_id=42,
                                                      lora_path=adapter_dir,
                                                      lora_ckpt_source="nemo")

        # Invalid lora_ckpt_source -> raises (must be hf or nemo).
        request = make_mock_triton_request({
            "lora_id": [42],
            "lora_name": [b"a"],
            "lora_path": [adapter_dir.encode("utf-8")],
            "lora_ckpt_source": [b"bogus"],
        })
        with pytest.raises(MockTritonModelException):
            get_lora_request_from_request(request)

        # All three partial-input permutations -> raise TritonModelException.
        for partial in (
            {
                "lora_id": [42]
            },
            {
                "lora_name": [b"adapter"],
                "lora_path": [adapter_dir.encode("utf-8")]
            },
            {
                "lora_path": [adapter_dir.encode("utf-8")]
            },
        ):
            with pytest.raises(MockTritonModelException):
                get_lora_request_from_request(make_mock_triton_request(partial))

        # lora_path that doesn't exist -> raises TritonModelException (not
        # raw ValueError from LoRARequest.__post_init__).
        request = make_mock_triton_request({
            "lora_id": [42],
            "lora_name": [b"a"],
            "lora_path": [b"/nonexistent-path-xyzzy"],
        })
        with pytest.raises(MockTritonModelException):
            get_lora_request_from_request(request)


def test_get_parameter():
    # Test valid parameter cases
    model_config = {
        "parameters": {
            "valid_int": {
                "string_value": "42"
            },
            "valid_bool": {
                "string_value": "True"
            },
            "valid_str": {
                "string_value": "test_str"
            },
            "invalid_number": {
                "string_value": "not_a_number"
            },
            "empty_param": {
                "string_value": ""
            },
            "env_var_param": {
                "string_value": "${ENV_VAR}"
            }
        }
    }

    # Valid parameter reads
    assert get_parameter(model_config, "valid_int", int) == 42
    assert get_parameter(model_config, "valid_bool", bool) is True
    assert get_parameter(model_config, "valid_str", str) == "test_str"

    # Invalid parameter handling
    assert get_parameter(model_config, "invalid_number", int) is None
    assert get_parameter(model_config, "non_existent_param") is None

    # Special cases
    assert get_parameter(model_config, "empty_param") is None
    assert get_parameter(model_config, "env_var_param") is None


def _make_multimodal_model(enabled: bool):
    """A bare model with just the multimodal state `_convert_request` reads."""
    model = TritonPythonModel.__new__(TritonPythonModel)
    model.multimodal_enabled = enabled
    model._mm_tokenizer = "tokenizer"
    model._mm_processor = "processor"
    model._mm_model_type = "qwen2_5_vl"
    return model


def test_convert_request_ignores_image_url_when_multimodal_disabled():
    # A deployment that already declares an `image_url` input for another
    # purpose must be unaffected until it opts in via triton_config.multimodal.
    model = _make_multimodal_model(enabled=False)
    request = make_mock_triton_request({
        **inputs(),
        "image_url": [b"https://example.com/a.jpg"],
    })

    prompt, _, _, _, _ = asyncio.run(model._convert_request(request))

    assert prompt == "Tell me a story."


def test_convert_request_delegates_to_shared_inputs_helper():
    # The backend must not build the multimodal prompt itself: placeholder
    # handling depends on the model's ContentFormat, which the shared helper in
    # tensorrt_llm.inputs owns. Assert we hand it the right arguments.
    model = _make_multimodal_model(enabled=True)
    captured = {}

    async def fake_helper(**kwargs):
        captured.update(kwargs)
        return {
            "prompt": "rendered",
            "multi_modal_data": {
                "image": ["decoded"]
            }
        }

    inputs_mod = MagicMock()
    inputs_mod.async_build_multimodal_prompt = fake_helper
    request = make_mock_triton_request({
        **inputs(),
        "image_url": [b"https://example.com/a.jpg", b"/tmp/b.png"],
    })

    with patch.dict(sys.modules, {
            "tensorrt_llm": MagicMock(),
            "tensorrt_llm.inputs": inputs_mod,
    }):
        prompt, _, _, _, _ = asyncio.run(model._convert_request(request))

    assert prompt["multi_modal_data"] == {"image": ["decoded"]}
    assert captured["model_type"] == "qwen2_5_vl"
    assert captured["tokenizer"] == "tokenizer"
    assert captured["processor"] == "processor"
    assert captured["modality"] == "image"
    assert captured["prompt"] == "Tell me a story."
    # Bytes tensors are decoded, order preserved.
    assert captured["media"] == ["https://example.com/a.jpg", "/tmp/b.png"]


def _bare_model_for_execute():
    """A model with only the state `_execute_single_request` touches."""
    model = TritonPythonModel.__new__(TritonPythonModel)
    model.logger = MagicMock()
    model.lock = threading.Lock()
    model.req_id_to_request_data = {}
    model.triton_user_id_to_req_ids = {}
    model._ongoing_request_count = 0
    model.decoupled = False
    model.output_dtype = np.object_
    return model


class _RecordingSender:

    def __init__(self):
        self.sent = []

    def send(self, response, flags=None):
        self.sent.append((response, flags))


def test_execute_single_request_reports_preprocessing_failure():
    # A bad image URL fails inside _convert_request, before the request is
    # registered in req_id_to_request_data. The error must still reach the
    # client with COMPLETE_FINAL, otherwise it waits forever.
    model = _bare_model_for_execute()
    sender = _RecordingSender()
    request = make_mock_triton_request({"text_input": ["describe this"]})
    request.get_response_sender = lambda: sender
    request.request_id = lambda: "triton-user-1"

    async def failing_convert(_request):
        raise RuntimeError(
            "Cannot connect to host example.invalid:443 [Name or service not known]"
        )

    model._convert_request = failing_convert

    with patch.dict(sys.modules, {"tensorrt_llm": MagicMock()}):
        with pytest.raises(RuntimeError):
            asyncio.run(model._execute_single_request(request))

    pb_utils = sys.modules["triton_python_backend_utils"]
    assert len(sender.sent) == 1, "client must receive exactly one response"
    response, flags = sender.sent[0]
    assert flags == pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
    assert response.has_error()
    assert "example.invalid" in response.error.message


def test_execute_single_request_skips_response_when_cancelled():
    # Once the request IS registered, an empty map means the cancellation loop
    # already sent COMPLETE_FINAL and removed the entry, so the error handler
    # must stay silent rather than send a second final response.
    model = _bare_model_for_execute()
    sender = _RecordingSender()
    request = make_mock_triton_request({"text_input": ["describe this"]})
    request.get_response_sender = lambda: sender
    request.request_id = lambda: "triton-user-2"

    async def convert(_request):
        return ("a prompt", {}, False, {}, None)

    class CancellingIterator:

        def __aiter__(self):
            return self

        async def __anext__(self):
            # Stand in for cancellation_loop: it sends COMPLETE_FINAL and drops
            # the entry while generation is in flight.
            with model.lock:
                model.req_id_to_request_data.clear()
            raise RuntimeError("request aborted")

    engine = MagicMock()
    engine.generate_async.return_value = CancellingIterator()
    model._convert_request = convert
    model._llm_engine = engine

    with patch.dict(sys.modules, {"tensorrt_llm": MagicMock()}):
        with pytest.raises(RuntimeError):
            asyncio.run(model._execute_single_request(request))

    assert sender.sent == [], "must not double-send after cancellation"
