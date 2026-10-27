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
"""DeepSeek-V3.2's vendored prompt encoder against the checkpoint's own golden prompts.

The golden checks need the checkpoint's ``encoding/`` directory under ``LLM_MODELS_ROOT``.
"""

import copy
import json
import os
from pathlib import Path

import pytest

from tensorrt_llm.tokenizer.deepseek_v32 import encode_messages

pytestmark = pytest.mark.cpu_only

MODELS_ROOT = Path(os.environ.get("LLM_MODELS_ROOT", "/code/llm-models"))
ENCODING_DIR = MODELS_ROOT / "DeepSeek-V3.2-hf" / "encoding"

_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def test_tools_preamble_keeps_its_blank_lines():
    """The preamble is rendered as the checkpoint's encoder writes it, blank lines included."""
    prompt = encode_messages(
        [
            {"role": "system", "content": "S", "tools": [_TOOL]},
            {"role": "user", "content": "q"},
        ],
        thinking_mode="chat",
    )

    assert "## Tools\n\nYou have access to a set of tools" in prompt


@pytest.mark.parametrize(
    "case_file, golden_file",
    [
        ("test_input.json", "test_output.txt"),
        ("test_input_search_w_date.json", "test_output_search_w_date.txt"),
        ("test_input_search_wo_date.json", "test_output_search_wo_date.txt"),
    ],
)
def test_matches_the_checkpoint_golden_prompts(case_file, golden_file):
    """The checkpoint's own test: tools on the first message, thinking mode."""
    case_path = ENCODING_DIR / case_file
    if not case_path.exists():
        pytest.skip(
            f"the DeepSeek-V3.2 checkpoint encoder tests are not available at {ENCODING_DIR}; "
            "set LLM_MODELS_ROOT to run them"
        )
    case = json.loads(case_path.read_text())
    messages = copy.deepcopy(case["messages"])
    if "tools" in case:
        messages[0]["tools"] = case["tools"]

    prompt = encode_messages(
        messages, thinking_mode="thinking", drop_thinking=True, add_default_bos_token=True
    )

    assert prompt == (ENCODING_DIR / golden_file).read_text().strip()
