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
"""Evaluators tokenize chat-template prompts without adding special tokens.

A chat template already writes the special tokens the model expects (the Llama,
Mistral and Gemma templates start with BOS). trtllm-serve tokenizes a rendered
chat prompt with ``add_special_tokens=False``; the evaluators must do the same,
or the tokenizer puts a second BOS in front of the template's.
"""

from types import SimpleNamespace
from unittest.mock import patch

import datasets
import pytest

from tensorrt_llm.evaluate.interface import Evaluator
from tensorrt_llm.evaluate.lm_eval import LmEvalEvaluator, LmEvalWrapper
from tensorrt_llm.evaluate.nemo_skills_eval import NemoSkillsEvaluator
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only

_BOS = "<s>"


class _ChatTemplateTokenizer:
    """Renders messages after a BOS, as the Llama, Mistral and Gemma templates do."""

    chat_template = "{{ bos_token }}{% for message in messages %}...{% endfor %}"
    name_or_path = "test/chat-model"
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        assert not tokenize
        rendered = "".join(f"[{m['role']}]{m['content']}" for m in messages)
        return _BOS + rendered + ("[assistant]" if add_generation_prompt else "")


class _RecordingLLM:
    """Records each request's prompt and SamplingParams; answers with canned texts."""

    def __init__(self, responses=()):
        self.tokenizer = _ChatTemplateTokenizer()
        self.requests = []
        self._responses = iter(responses)

    def generate_async(self, prompt, sampling_params=None, streaming=False):
        self.requests.append((prompt, sampling_params))
        output = SimpleNamespace(
            prompt=prompt,
            outputs=[SimpleNamespace(text=next(self._responses, ""), token_ids=[1, 2])],
        )
        return SimpleNamespace(result=lambda: output)


class _ToyEvaluator(Evaluator):
    def generate_samples(self):
        yield "What is 2 + 3?", {"max_tokens": 4}, "5"
        yield "What is 4 + 4?", None, "8"

    def compute_score(self, outputs, references, *auxiliaries):
        return 0.0


@pytest.mark.parametrize("apply_chat_template", [False, True])
def test_evaluator_tokenizes_chat_prompts_without_special_tokens(apply_chat_template):
    llm = _RecordingLLM()
    sampling_params = SamplingParams(max_tokens=16)

    _ToyEvaluator(apply_chat_template=apply_chat_template).evaluate(llm, sampling_params)

    assert len(llm.requests) == 2
    for prompt, request_params in llm.requests:
        assert prompt.startswith(_BOS) is apply_chat_template
        assert request_params.add_special_tokens is not apply_chat_template
    # Each request starts from the caller's SamplingParams, which stay unchanged.
    assert [params.max_tokens for _, params in llm.requests] == [4, 16]
    assert sampling_params.add_special_tokens is True
    assert sampling_params.max_tokens == 16


_DOCS = [
    {"question": "2+2?", "answer": "4"},
    {"question": "3+4?", "answer": "7"},
    {"question": "5+6?", "answer": "11"},
    {"question": "10-3?", "answer": "7"},
]
# Three correct answers and one wrong one: exact_match 0.75.
_RESPONSES = ["The answer is 4.", "The answer is 7.", "The answer is 12.", "The answer is 7."]


def _toy_task():
    """A real generate_until ConfigurableTask over an in-memory dataset."""
    from lm_eval.api.task import ConfigurableTask

    return ConfigurableTask(
        config={
            "task": "toy_arith",
            "custom_dataset": lambda **kwargs: datasets.DatasetDict(
                {"test": datasets.Dataset.from_list(_DOCS)}
            ),
            "test_split": "test",
            "output_type": "generate_until",
            "doc_to_text": "Q: {{question}}\nA:",
            "doc_to_target": "{{answer}}",
            "generation_kwargs": {"until": ["\n"], "do_sample": False},
            "filter_list": [
                {
                    "name": "strict-match",
                    "filter": [
                        {"function": "regex", "regex_pattern": r"(-?[0-9]+)"},
                        {"function": "take_first"},
                    ],
                }
            ],
            "metric_list": [
                {"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}
            ],
        }
    )


@pytest.mark.parametrize("apply_chat_template", [False, True])
def test_lm_eval_wrapper_tokenizes_chat_prompts_without_special_tokens(apply_chat_template):
    from lm_eval.evaluator import evaluate

    llm = _RecordingLLM(_RESPONSES)
    results = evaluate(
        lm=LmEvalWrapper(llm, apply_chat_template=apply_chat_template),
        task_dict={"toy_arith": _toy_task()},
        bootstrap_iters=0,
        log_samples=False,
        apply_chat_template=apply_chat_template,
    )

    assert results["results"]["toy_arith"]["exact_match,strict-match"] == pytest.approx(0.75)
    assert len(llm.requests) == len(_DOCS)
    for prompt, sampling_params in llm.requests:
        assert prompt.startswith(_BOS) is apply_chat_template
        assert sampling_params.add_special_tokens is not apply_chat_template


@pytest.mark.parametrize("apply_chat_template", [False, True])
def test_lm_eval_evaluator_passes_chat_template_setting_to_wrapper(apply_chat_template):
    evaluator = LmEvalEvaluator.__new__(LmEvalEvaluator)
    evaluator.__dict__.update(
        MULTIMODAL=False,
        task_name="toy_arith",
        task_dict={},
        num_samples=None,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=False,
        system_prompt=None,
        chat_template_kwargs=None,
        log_samples=False,
        output_path=None,
        output_dir=None,
    )
    wrappers = []

    def fake_evaluate(lm, **kwargs):
        wrappers.append(lm)
        return {"results": {"toy_arith": {"exact_match,strict-match": 0.5}}}

    with (
        patch("lm_eval.evaluate", fake_evaluate),
        patch("lm_eval.utils.make_table", return_value=""),
    ):
        evaluator.evaluate(_RecordingLLM())

    assert len(wrappers) == 1
    assert wrappers[0].add_special_tokens is not apply_chat_template


def test_nemo_skills_generation_tokenizes_chat_prompts_without_special_tokens():
    evaluator = NemoSkillsEvaluator.__new__(NemoSkillsEvaluator)
    evaluator.__dict__.update(
        apply_chat_template=True, system_prompt=None, chat_template_kwargs=None
    )
    llm = _RecordingLLM(["4"])
    sampling_params = SamplingParams(max_tokens=16)

    texts = evaluator._batched_generate(
        llm, [[{"role": "user", "content": "What is 2 + 2?"}]], sampling_params
    )

    assert texts == ["4"]
    prompt, request_params = llm.requests[0]
    assert prompt.startswith(_BOS)
    assert request_params.add_special_tokens is False
    assert sampling_params.add_special_tokens is True
