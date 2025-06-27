# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.llm_request import (LlmRequest, LlmResponse,
                                                        SamplingConfig)


def create_sampling_config():
    """Create test sampling configuration."""
    config = SamplingConfig()
    # Used setattr method due to incompatible binding.
    setattr(config, 'top_p', [0.9])
    setattr(config, 'num_return_sequences', 2)
    return config


def test_create_request():
    """Test basic LlmRequest creation and attribute initialization."""
    sampling_config = create_sampling_config()
    request = LlmRequest(
        request_id=1,
        max_new_tokens=10,
        input_tokens=[1, 2, 3],
        sampling_config=sampling_config,
        is_streaming=False,
    )

    # Verify basic attributes
    assert request.py_request_id == 1
    assert request.py_max_new_tokens == 10
    assert request.py_prompt_len == 3
    assert request.py_orig_prompt_len == 3
    assert request.py_client_id is None

    # Verify default values
    assert not request.py_return_log_probs
    assert not request.py_return_context_logits
    assert not request.py_return_generation_logits
    assert request.py_return_logits_device_memory
    assert not request.py_is_draft
    assert not request.py_exclude_last_generation_logits

    # Verify PyResult is initialized
    assert request.py_result is not None


def test_create_request_with_optional_params():
    """Test LlmRequest creation with optional parameters."""
    sampling_config = create_sampling_config()
    request = LlmRequest(
        request_id=2,
        max_new_tokens=20,
        input_tokens=[10, 20, 30, 40],
        sampling_config=sampling_config,
        is_streaming=False,
        client_id=100,
        return_log_probs=True,
        return_context_logits=True,
        return_generation_logits=True,
        return_logits_device_memory=False,
        exclude_last_generation_logits=True,
        is_draft=True,
    )

    # Verify optional parameters
    assert request.py_client_id == 100
    assert request.py_return_log_probs
    assert request.py_return_context_logits
    assert request.py_return_generation_logits
    assert not request.py_return_logits_device_memory
    assert request.py_exclude_last_generation_logits
    assert request.py_is_draft


def test_create_child_request():
    """Test create_child_request method."""
    sampling_config = create_sampling_config()
    # Create parent request with various attributes
    parent_request = LlmRequest(
        request_id=1,
        max_new_tokens=10,
        input_tokens=[1, 2, 3],
        sampling_config=sampling_config,
        is_streaming=False,
        client_id=50,
        return_log_probs=True,
        return_context_logits=True,
    )

    # Create child request
    child_request = parent_request.create_child_request(2)

    # Verify child request attributes
    assert child_request.request_id == 2
    assert child_request.py_request_id == 2
    assert child_request.py_parent_request_id == 1

    # Verify copied configuration
    assert child_request.py_client_id == 50
    assert child_request.py_max_new_tokens == 10
    assert child_request.get_tokens() == [[1, 2, 3]]
    assert child_request.py_return_log_probs
    assert child_request.py_return_context_logits

    # Verify runtime state
    assert child_request.py_batch_idx is None  # Reset to None

    # Verify PyResult is new instance
    assert child_request.py_result is not None
    assert child_request.py_result is not parent_request.py_result

    # Cannot create child request more than num_return_sequences.
    with pytest.raises(RuntimeError):
        child_request.create_child_request(3)


def test_child_inherits_parent_attributes():
    """Test that child requests properly inherit parent attributes"""
    sampling_config = create_sampling_config()
    # Set up parent with various attributes
    parent_request = LlmRequest(request_id=100,
                                max_new_tokens=20,
                                input_tokens=[1, 2, 3, 4],
                                sampling_config=sampling_config,
                                is_streaming=True,
                                client_id=2000,
                                return_log_probs=True,
                                return_context_logits=True,
                                return_generation_logits=True)

    child = parent_request.create_child_request(2)

    # Verify inheritance
    assert child.py_client_id == parent_request.py_client_id
    assert child.py_max_new_tokens == parent_request.py_max_new_tokens
    assert child.py_return_log_probs == parent_request.py_return_log_probs
    assert (child.py_return_context_logits ==
            parent_request.py_return_context_logits)
    assert (child.py_return_generation_logits ==
            parent_request.py_return_generation_logits)


def test_parent_child_independence():
    """Test that parent and child requests are independent"""
    sampling_config = create_sampling_config()
    input_tokens = [1, 2, 3]
    input_len = len(input_tokens)
    parent_request = LlmRequest(request_id=100,
                                max_new_tokens=10,
                                input_tokens=input_tokens,
                                sampling_config=sampling_config,
                                is_streaming=False,
                                client_id=1000)

    # Create child requests
    child_request = parent_request.create_child_request(2)

    # Verify initial tokens are the same content but different objects
    assert parent_request.get_tokens() == child_request.get_tokens()
    assert (parent_request.get_tokens() is not
            child_request.get_tokens()), \
        "Parent and child should have independent token lists"

    # Test token generation independence
    # Add new tokens to each request
    parent_request.add_new_token(10, beam=0)
    child_request.add_new_token(20, beam=0)

    # Verify tokens are updated independently in the first beam.
    assert 10 in parent_request.get_tokens()[0]
    assert 20 not in parent_request.get_tokens()[0]

    assert 20 in child_request.get_tokens()[0]
    assert 10 not in child_request.get_tokens()[0][input_len:]
    assert 30 not in child_request.get_tokens()[0]

    # Test that each has independent PyResult
    assert parent_request.py_result is not child_request.py_result


def test_create_response():
    """Test create_response method of parent and child requests."""
    sampling_config = create_sampling_config()
    request = LlmRequest(
        request_id=1,
        max_new_tokens=10,
        input_tokens=[1, 2, 3],
        sampling_config=sampling_config,
        is_streaming=False,
        client_id=100,
    )

    child_request = request.create_child_request(2)
    child_response = child_request.create_response()

    # Test when result is not None
    response = request.create_response(use_fast_logits=True, mpi_world_rank=1)
    assert response is not None
    assert isinstance(response, LlmResponse)
    assert response.request_id == request.py_request_id
    assert response.client_id == request.py_client_id
    assert response.error_msg is None
    assert response.result is not None
    assert response.result.sequence_index == 0

    assert child_response is not None
    assert child_response.request_id == request.py_request_id
    assert child_response.client_id == child_request.py_client_id
    assert child_response.error_msg is None
    assert child_response.result is not None
    assert child_response.result.sequence_index == 1


def test_creates_none_response_when_result_is_none():
    """None response should be returned when request result is None."""
    sampling_config = create_sampling_config()
    request = LlmRequest(
        request_id=1,
        max_new_tokens=10,
        input_tokens=[1, 2, 3],
        sampling_config=sampling_config,
        is_streaming=False,
        client_id=100,
    )

    # Mock create_result to return None
    request.create_result = MagicMock(return_value=None)

    # Test when result is None
    response = request.create_response()

    assert response is None
