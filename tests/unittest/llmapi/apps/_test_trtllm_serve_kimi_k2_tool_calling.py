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

import json
import os
import subprocess
import tempfile
from typing import Any, Dict

import pytest
import yaml
from utils.util import skip_gpu_memory_less_than_138gb, skip_non_hopper_unittest

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture(scope="module")
def model_name():
    return "Kimi-K2-Instruct"


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file():
    """Create temporary extra LLM API options file for Kimi K2."""
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "kimi_k2_extra_options.yaml")
    try:
        # Configuration optimized for Kimi K2 MoE model
        extra_llm_api_options_dict = {
            "cuda_graph_config": {
                "batch_sizes": [1, 4]
            },
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.8
            },
            "enable_attention_dp": False,
            "enable_chunked_prefill": False,
            "max_batch_size": 8,
            "max_seq_len": 2048,
            "max_num_tokens": 2048
        }

        with open(temp_file_path, 'w') as f:
            yaml.dump(extra_llm_api_options_dict, f)

        yield temp_file_path
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@pytest.fixture(scope="module")
def server(model_name: str, temp_extra_llm_api_options_file: str):
    """Start server with Kimi K2 model configuration."""
    model_path = get_model_path(model_name)

    # Configuration for MoE model - typically requires multiple GPUs
    args = [
        "--tp_size",
        "8",  # Tensor parallelism for large MoE model
        "--ep_size",
        "8",  # Expert parallelism for MoE
        f"--extra_llm_api_options={temp_extra_llm_api_options_file}",
    ]

    with RemoteOpenAIServer(model_path, args,
                            max_server_start_wait_s=7200) as remote_server:
        yield remote_server


def run_kimi_k2_example(message: str,
                        specify_output_format: bool = True) -> Dict[str, Any]:
    """Run the Kimi K2 tool calling example script using subprocess."""
    script_path = os.path.join(os.path.dirname(__file__),
                               "kimi_k2_tool_calling_example.py")

    cmd = [
        "python3",
        script_path,
        "--message",
        message,
    ]

    if specify_output_format:
        cmd.append("--specify_output_format")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        return {"error": f"Script failed: {e.stderr}", "success": False}
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse JSON output: {e}", "success": False}


@skip_non_hopper_unittest
@skip_gpu_memory_less_than_138gb
def test_kimi_k2_not_specify_output_format_tool_calling(
        server: RemoteOpenAIServer):
    """Test Kimi K2 tool calling with standard format."""
    # Run the example script
    result = run_kimi_k2_example("What's the weather like in Shanghai today?")

    # Verify the script ran successfully
    assert result[
        "success"], f"Script failed: {result.get('error', 'Unknown error')}"

    # Verify response
    assert result["response"] is not None

    # Verify tool calls
    tool_calls = result["tool_calls"]
    assert len(
        tool_calls) > 0, f"No tool calls found in output: {result['response']}"

    tool_call = tool_calls[0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"

    # Verify tool results
    tool_results = result["tool_results"]
    assert len(tool_results) > 0
    assert tool_results[0] == "Cloudy"


@skip_non_hopper_unittest
@skip_gpu_memory_less_than_138gb
def test_kimi_k2_specified_format_tool_calling(server: RemoteOpenAIServer):
    """Test Kimi K2 tool calling with specified output format."""
    # Run the example script with specified format
    result = run_kimi_k2_example("What's the weather like in Beijing today?",
                                 specify_output_format=True)

    # Verify the script ran successfully
    assert result[
        "success"], f"Script failed: {result.get('error', 'Unknown error')}"

    # Verify response
    assert result["response"] is not None

    # Verify tool calls
    tool_calls = result["tool_calls"]
    assert len(
        tool_calls) > 0, f"No tool calls found in output: {result['response']}"

    tool_call = tool_calls[0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"

    # Verify tool results
    tool_results = result["tool_results"]
    assert len(tool_results) > 0
    assert tool_results[0] == "Sunny"


@skip_non_hopper_unittest
@skip_gpu_memory_less_than_138gb
def test_kimi_k2_multiple_tool_calls(server: RemoteOpenAIServer):
    """Test Kimi K2 with multiple tool calls."""
    # Run the example script with a message requiring multiple tool calls
    result = run_kimi_k2_example(
        "What's the weather in Tokyo and what time is it in JST timezone?")

    # Verify the script ran successfully
    assert result[
        "success"], f"Script failed: {result.get('error', 'Unknown error')}"

    # Verify response
    assert result["response"] is not None

    # Verify tool calls
    tool_calls = result["tool_calls"]
    assert len(
        tool_calls
    ) >= 1, f"Expected multiple tool calls in output: {result['response']}"

    # Verify tool results
    tool_results = result["tool_results"]
    assert len(tool_results) > 0
