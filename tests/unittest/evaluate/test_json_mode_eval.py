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

import json
from types import SimpleNamespace

import datasets
import pytest

from tensorrt_llm.evaluate.json_mode_eval import JsonModeEval

pytestmark = pytest.mark.cpu_only

_SCHEMA = json.dumps(
    {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
)


class _ChatTemplateTokenizer:
    """Renders messages and then appends a fixed generation prompt."""

    def __init__(self, generation_prompt: str) -> None:
        self.generation_prompt = generation_prompt

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs) -> str:
        assert not tokenize and add_generation_prompt
        rendered = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return rendered + self.generation_prompt


@pytest.fixture
def evaluator(monkeypatch: pytest.MonkeyPatch) -> JsonModeEval:
    sample = {
        "prompt": [
            {"role": "system", "content": "Answer in JSON."},
            {"role": "user", "content": "The name is Ada."},
        ],
        "completion": '{"name": "Ada"}',
        "schema": _SCHEMA,
    }
    monkeypatch.setattr(
        datasets, "load_dataset", lambda *args, **kwargs: datasets.Dataset.from_list([sample])
    )
    return JsonModeEval(dataset_path="unused")


def _render_first_prompt(evaluator: JsonModeEval, generation_prompt: str) -> str:
    llm = SimpleNamespace(tokenizer=_ChatTemplateTokenizer(generation_prompt))
    prompt, sampling_args, _, _ = next(iter(evaluator.generate_samples()))
    assert sampling_args["guided_decoding"].json is not None
    return evaluator.do_apply_chat_template(llm, prompt)


def test_harmony_prompt_opens_the_final_channel(evaluator: JsonModeEval) -> None:
    rendered = _render_first_prompt(evaluator, "<|start|>assistant")

    assert rendered.endswith("<|start|>assistant<|channel|>final<|message|>")


@pytest.mark.parametrize(
    "generation_prompt",
    ["<|im_start|>assistant\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"],
)
def test_other_generation_prompts_are_unchanged(
    evaluator: JsonModeEval, generation_prompt: str
) -> None:
    rendered = _render_first_prompt(evaluator, generation_prompt)

    assert rendered.endswith(generation_prompt)
    assert "<|channel|>" not in rendered
