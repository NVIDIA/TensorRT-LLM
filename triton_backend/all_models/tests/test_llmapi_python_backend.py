# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
from dataclasses import dataclass
from typing import Dict, List, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Mock pb_utils
sys.modules["triton_python_backend_utils"] = MagicMock()

from helpers import (convert_request_input_to_dict,
                     get_output_config_from_request, get_parameter,
                     get_sampling_params_from_request,
                     get_streaming_from_request)
# Use PYTHONPATH=../llmapi/tensorrt_llm/1/
from model import *


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
    patch("model.pb_utils.TritonModelException", new=Exception).start()


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
