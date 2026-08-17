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
import asyncio
import json
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.llmapi.reasoning_parser import (
    HARMONY_FINAL_CHANNEL_TRIGGER,
    HARMONY_REASONING_PARSER,
    adapt_guided_decoding_params_for_reasoning_parser,
    resolve_guided_decoding_reasoning_parser,
    resolve_raw_guided_decoding_reasoning_parser,
)
from tensorrt_llm.llmapi.thinking_budget import (
    ThinkingBudgetLogitsProcessor,
    add_thinking_budget_logits_processor,
)
from tensorrt_llm.sampling_params import (
    MAX_TOP_LOGPROBS,
    GuidedDecodingParams,
    SamplingParams,
    check_logprobs_limit,
)
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    KVCacheTruncateRequest,
    ResponsesRequest,
    ensure_request_chat_template_allowed,
)
from tensorrt_llm.serve.resource_governor import ResourceGovernor

pytestmark = pytest.mark.cpu_only


def _apply_generation_config_sampling_defaults(
    mode: str,
    sampling_params: SamplingParams,
    generation_config_explicit_values: dict,
) -> SamplingParams:
    llm = SimpleNamespace(
        args=SimpleNamespace(backend="pytorch", generation_config=mode),
        _generation_config_explicit_values=generation_config_explicit_values,
    )
    BaseLLM._apply_generation_config_sampling_defaults(llm, sampling_params)
    return sampling_params


def test_generation_config_mode_controls_sampling_defaults():
    values = {"temperature": 0.7}

    trtllm_params = _apply_generation_config_sampling_defaults(
        "trtllm", SamplingParams(end_id=1), values
    )
    auto_params = _apply_generation_config_sampling_defaults(
        "auto", SamplingParams(end_id=1), values
    )

    assert trtllm_params.temperature is None
    assert auto_params.temperature == 0.7


def test_generation_config_sampling_precedence():
    params = SamplingParams(end_id=1, temperature=0.2)
    prepared = _apply_generation_config_sampling_defaults(
        "auto",
        params,
        {
            "temperature": 0.7,
            "top_p": 0.9,
        },
    )

    # Check each precedence level:
    # 1. Request-provided values
    # 2. Generation config values
    # 3. TRT-LLM defaults
    assert prepared.temperature == 0.2
    assert prepared.top_p == 0.9
    assert prepared.top_k is None


def test_generation_config_applies_all_supported_sampling_fields():
    # Use typical_p because it is not supported by the generation config.
    values = {
        "early_stopping": True,
        "length_penalty": 0.8,
        "min_p": 0.05,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.1,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.9,
        "typical_p": 0.95,
    }

    prepared = _apply_generation_config_sampling_defaults("auto", SamplingParams(end_id=1), values)

    for field_name, expected in values.items():
        if field_name != "typical_p":
            assert getattr(prepared, field_name) == expected

    unsupported_values = _apply_generation_config_sampling_defaults(
        "auto",
        SamplingParams(end_id=1),
        {
            "top_p": None,
            "early_stopping": "never",
        },
    )
    assert unsupported_values.top_p is None
    assert unsupported_values.early_stopping is None


@pytest.mark.parametrize(
    "request_obj",
    [
        CompletionRequest(model="test", prompt="hi"),
        ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "hi"}]),
        ResponsesRequest(model="test", input="hi"),
    ],
)
def test_generation_config_overrides_omitted_serve_defaults(request_obj):
    params = request_obj.to_sampling_params()

    assert params.temperature == 1.0
    prepared = _apply_generation_config_sampling_defaults(
        "auto", params, {"temperature": 0.7, "top_p": 0.9}
    )

    assert prepared.temperature == 0.7
    assert prepared.top_p == 0.9


@pytest.mark.parametrize(
    "request_obj",
    [
        CompletionRequest(model="test", prompt="hi", temperature=0.2),
        ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
        ),
        ResponsesRequest(model="test", input="hi", temperature=0.2),
    ],
)
def test_explicit_serve_request_overrides_generation_config(request_obj):
    prepared = _apply_generation_config_sampling_defaults(
        "auto",
        request_obj.to_sampling_params(),
        {"temperature": 0.7},
    )

    assert prepared.temperature == 0.2


def test_absent_generation_config_value_preserves_existing_defaults():
    direct_params = _apply_generation_config_sampling_defaults("auto", SamplingParams(end_id=1), {})
    serve_params = _apply_generation_config_sampling_defaults(
        "auto",
        CompletionRequest(model="test", prompt="hi").to_sampling_params(),
        {},
    )

    assert direct_params.temperature is None
    assert serve_params.temperature == 1.0


@pytest.mark.parametrize("field", ["logprobs", "prompt_logprobs", "top_logprobs"])
def test_check_logprobs_limit(field):
    check_logprobs_limit(field, None)
    check_logprobs_limit(field, 0)
    check_logprobs_limit(field, MAX_TOP_LOGPROBS)

    with pytest.raises(ValueError, match=f"{field} must be positive"):
        check_logprobs_limit(field, -1)

    with pytest.raises(ValueError, match=f"less than or equal to {MAX_TOP_LOGPROBS}"):
        check_logprobs_limit(field, MAX_TOP_LOGPROBS + 1)


@pytest.mark.parametrize("field", ["logprobs", "prompt_logprobs"])
def test_logprobs_request_limit(field):
    SamplingParams(**{field: MAX_TOP_LOGPROBS})

    with pytest.raises(ValueError, match=f"less than or equal to {MAX_TOP_LOGPROBS}"):
        SamplingParams(**{field: MAX_TOP_LOGPROBS + 1})


def test_chat_top_logprobs_request_limit():
    with pytest.raises(ValueError, match=f"less than or equal to {MAX_TOP_LOGPROBS}"):
        ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            logprobs=True,
            top_logprobs=MAX_TOP_LOGPROBS + 1,
        )


def test_harmony_guided_decoding_triggers_on_final_channel():
    guided_decoding = GuidedDecodingParams(json={"type": "object"})

    adapted = adapt_guided_decoding_params_for_reasoning_parser(
        guided_decoding, HARMONY_REASONING_PARSER
    )

    assert adapted is not guided_decoding
    assert adapted.structural_tag is not None
    stag = json.loads(adapted.structural_tag)
    assert stag["type"] == "structural_tag"

    fmt = stag["format"]
    assert fmt["type"] == "triggered_tags"
    assert fmt["triggers"] == [HARMONY_FINAL_CHANNEL_TRIGGER]
    assert fmt["stop_after_first"] is True

    tag = fmt["tags"][0]
    assert tag["begin"] == HARMONY_FINAL_CHANNEL_TRIGGER
    assert tag["end"] == ""
    assert tag["content"] == {
        "type": "json_schema",
        "json_schema": {"type": "object"},
    }


def test_harmony_guided_decoding_accepts_json_schema_string():
    json_schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }
    guided_decoding = GuidedDecodingParams(json=json.dumps(json_schema))

    adapted = adapt_guided_decoding_params_for_reasoning_parser(
        guided_decoding, HARMONY_REASONING_PARSER
    )

    stag = json.loads(adapted.structural_tag)
    content = stag["format"]["tags"][0]["content"]
    assert content == {
        "type": "json_schema",
        "json_schema": json_schema,
    }


def test_guided_decoding_preserves_top_level_schema_property():
    json_schema = {
        "type": "object",
        "schema": {"type": "string"},
    }

    adapted = adapt_guided_decoding_params_for_reasoning_parser(
        GuidedDecodingParams(json=json_schema), HARMONY_REASONING_PARSER
    )

    stag = json.loads(adapted.structural_tag)
    assert stag["format"]["tags"][0]["content"]["json_schema"] == json_schema


@pytest.mark.parametrize(
    ("reasoning_parser", "model_type", "expected"),
    [
        (None, "gpt_oss", HARMONY_REASONING_PARSER),
        (None, "llama", None),
        ("qwen3", "llama", "qwen3"),
    ],
)
def test_resolve_guided_decoding_reasoning_parser(reasoning_parser, model_type, expected):
    assert resolve_guided_decoding_reasoning_parser(reasoning_parser, model_type) == expected


@pytest.mark.parametrize(
    ("reasoning_parser", "model_type", "guided_backend", "expected"),
    [
        (None, "gpt_oss", "xgrammar", HARMONY_REASONING_PARSER),
        (HARMONY_REASONING_PARSER, "llama", "xgrammar", HARMONY_REASONING_PARSER),
        (None, "gpt_oss", "llguidance", None),
        ("qwen3", "qwen3", "xgrammar", None),
        ("gemma4", "gemma4", "xgrammar", None),
    ],
)
def test_resolve_raw_guided_decoding_reasoning_parser(
    reasoning_parser, model_type, guided_backend, expected
):
    assert (
        resolve_raw_guided_decoding_reasoning_parser(reasoning_parser, model_type, guided_backend)
        == expected
    )


@pytest.mark.parametrize(
    ("reasoning_parser", "model_type", "guided_backend"),
    [
        (None, "gpt_oss", "llguidance"),
        ("qwen3", "qwen3", "xgrammar"),
        ("gemma4", "gemma4", "xgrammar"),
    ],
)
def test_raw_llm_preserves_guides_outside_harmony_xgrammar(
    reasoning_parser, model_type, guided_backend
):
    guided_decoding = GuidedDecodingParams(json_object=True)
    resolved_parser = resolve_raw_guided_decoding_reasoning_parser(
        reasoning_parser, model_type, guided_backend
    )

    # These combinations worked without raw final-content adaptation before
    # bug 6284101; keep the caller's ordinary guide unchanged.
    assert (
        adapt_guided_decoding_params_for_reasoning_parser(guided_decoding, resolved_parser)
        is guided_decoding
    )


def test_plain_model_guided_decoding_is_unchanged():
    guided_decoding = GuidedDecodingParams(json_object=True)
    reasoning_parser = resolve_guided_decoding_reasoning_parser(None, "llama")

    assert (
        adapt_guided_decoding_params_for_reasoning_parser(guided_decoding, reasoning_parser)
        is guided_decoding
    )


def test_reasoning_parser_guided_decoding_uses_sequence_for_normal_parser():
    guided_decoding = GuidedDecodingParams(json_object=True)

    adapted = adapt_guided_decoding_params_for_reasoning_parser(guided_decoding, "qwen3")

    stag = json.loads(adapted.structural_tag)
    assert stag["type"] == "structural_tag"

    fmt = stag["format"]
    assert fmt["type"] == "sequence"
    reasoning_element, content_element = fmt["elements"]
    assert reasoning_element == {
        "type": "tag",
        "begin": "<think>",
        "content": {"type": "any_text"},
        "end": "</think>",
    }
    assert content_element == {
        "type": "json_schema",
        "json_schema": {"type": "object"},
    }


def test_existing_structural_tag_guided_decoding_is_unchanged():
    guided_decoding = GuidedDecodingParams(
        structural_tag='{"type":"structural_tag","format":{"type":"any_text"}}'
    )

    assert (
        adapt_guided_decoding_params_for_reasoning_parser(guided_decoding, HARMONY_REASONING_PARSER)
        is guided_decoding
    )


def test_chat_template_request_override_respects_runtime_policy():
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        chat_template="{{ messages }}",
    )

    with pytest.raises(ValueError, match="chat_template cannot be supplied"):
        ensure_request_chat_template_allowed(request, allow_request_chat_template=False)

    ensure_request_chat_template_allowed(request, allow_request_chat_template=True)


def test_kv_cache_truncate_chat_template_respects_runtime_policy():
    request = KVCacheTruncateRequest(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        chat_template="{{ messages }}",
    )

    with pytest.raises(ValueError, match="chat_template cannot be supplied"):
        ensure_request_chat_template_allowed(request, allow_request_chat_template=False)

    ensure_request_chat_template_allowed(request, allow_request_chat_template=True)


def test_resource_governor_rejects_request_chat_template_by_default():
    governor = ResourceGovernor(
        resource_governor_queue=None,
        tokenizer=None,
        model_config=None,
    )
    request = KVCacheTruncateRequest(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        chat_template="{{ messages }}",
    )

    response = asyncio.run(governor._truncate_kv_cache(request))

    assert response.status_code == 400
    assert json.loads(response.body)["error"]


def test_completion_logprobs_assignment_revalidates():
    request = CompletionRequest(model="test", prompt="hi", logprobs=MAX_TOP_LOGPROBS)
    request.logprobs = MAX_TOP_LOGPROBS + 1

    with pytest.raises(ValueError, match=f"less than or equal to {MAX_TOP_LOGPROBS}"):
        request.to_sampling_params(backend="pytorch")


@pytest.mark.parametrize("field", ["top_p", "min_p", "temperature"])
def test_sampling_params_rejects_nan(field):
    with pytest.raises(ValueError, match=field):
        SamplingParams(**{field: float("nan")})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_p", -0.1),
        ("top_p", 1.1),
        ("min_p", -0.1),
        ("min_p", 1.1),
        ("temperature", -1.0),
    ],
)
def test_sampling_params_rejects_out_of_range(field, value):
    with pytest.raises(ValueError, match=field):
        SamplingParams(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_p", 0.0),
        ("top_p", 0.9),
        ("top_p", 1.0),
        ("min_p", 0.0),
        ("min_p", 0.5),
        ("min_p", 1.0),
        ("temperature", 0.0),
        ("temperature", 1.0),
    ],
)
def test_sampling_params_accepts_in_range_values(field, value):
    assert getattr(SamplingParams(**{field: value}), field) == value


@pytest.mark.parametrize("value", [None, -1])
def test_thinking_token_budget_unlimited_values(value):
    assert SamplingParams(thinking_token_budget=value).thinking_token_budget is None


@pytest.mark.parametrize("value", [-2, True, 1.5, "1"])
def test_thinking_token_budget_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="thinking_token_budget"):
        SamplingParams(thinking_token_budget=value)


def test_chat_thinking_token_budget_request_validation():
    request = ChatCompletionRequest(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        thinking_token_budget=8,
    )

    assert request.to_sampling_params().thinking_token_budget == 8

    with pytest.raises(ValueError, match="thinking_token_budget"):
        ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            thinking_token_budget=True,
        )


def test_thinking_budget_logits_processor_forces_end_token():
    processor = ThinkingBudgetLogitsProcessor(
        thinking_token_budget=2,
        reasoning_start_token_ids=[1],
        reasoning_end_token_ids=[2, 3],
    )
    logits = torch.zeros(1, 1, 8)

    processor(0, logits, [[1, 5, 6]], None, None)

    assert logits[0, 0, 2] == 0
    mask = torch.ones(8, dtype=torch.bool)
    mask[2] = False
    assert torch.isneginf(logits[0, 0, mask]).all()


def test_thinking_budget_logits_processor_completes_partial_end_sequence():
    processor = ThinkingBudgetLogitsProcessor(
        thinking_token_budget=2,
        reasoning_start_token_ids=[1],
        reasoning_end_token_ids=[2, 3],
    )
    logits = torch.zeros(1, 1, 8)

    processor(0, logits, [[1, 5, 6, 2]], None, None)

    assert logits[0, 0, 3] == 0
    mask = torch.ones(8, dtype=torch.bool)
    mask[3] = False
    assert torch.isneginf(logits[0, 0, mask]).all()


def test_thinking_budget_logits_processor_ignores_partial_end_before_budget():
    processor = ThinkingBudgetLogitsProcessor(
        thinking_token_budget=4,
        reasoning_start_token_ids=[1],
        reasoning_end_token_ids=[2, 3],
    )
    logits = torch.zeros(1, 1, 8)

    processor(0, logits, [[1, 5, 6, 2]], None, None)

    assert torch.equal(logits, torch.zeros(1, 1, 8))


def test_thinking_budget_logits_processor_ignores_closed_reasoning_block():
    processor = ThinkingBudgetLogitsProcessor(
        thinking_token_budget=1,
        reasoning_start_token_ids=[1],
        reasoning_end_token_ids=[2],
    )
    logits = torch.zeros(1, 1, 8)

    processor(0, logits, [[1, 5, 2, 6]], None, None)

    assert torch.equal(logits, torch.zeros(1, 1, 8))


def _run(processor, token_ids):
    """Call the processor on a fresh logits row and return it."""
    logits = torch.zeros(1, 1, 8)
    processor(0, logits, token_ids, None, None)
    return logits


def test_thinking_budget_logits_processor_does_not_force_end_twice_on_stale_view():
    """Regression: a stale token view must not re-force the end tag.

    With the overlap scheduler, logits processors run one step
    behind the sampled tokens, so the call right after forcing the end token
    sees token_ids WITHOUT it. A stateless processor forces the end token a
    second time, producing e.g. `</think></think>` — the reasoning parser
    consumes the first tag and the second leaks into message.content.
    """
    processor = ThinkingBudgetLogitsProcessor(
        thinking_token_budget=2,
        reasoning_start_token_ids=[1],
        reasoning_end_token_ids=[2],
    )
    stale = [[1, 5, 6]]  # budget spent, forced end not visible yet

    logits = _run(processor, stale)
    assert logits[0, 0, 2] == 0 and torch.isneginf(logits[0, 0, 3])

    # Next step: the forced end token is still not visible (stale view).
    assert torch.equal(_run(processor, stale), torch.zeros(1, 1, 8))

    # Once the view catches up (end tag present), progress state resets so a
    # later reasoning block is budgeted again.
    assert torch.equal(_run(processor, [[1, 5, 6, 2, 7]]), torch.zeros(1, 1, 8))
    logits = _run(processor, [[1, 5, 6, 2, 7, 1, 8, 9]])
    assert logits[0, 0, 2] == 0 and torch.isneginf(logits[0, 0, 3])


def test_thinking_budget_logits_processor_continues_end_sequence_on_stale_view():
    """Multi-token end sequence must continue, not restart, on a stale view.

    Under the overlap scheduler the processor must continue from its
    recorded progress rather than re-derive from token_ids.
    """
    processor = ThinkingBudgetLogitsProcessor(
        thinking_token_budget=2,
        reasoning_start_token_ids=[1],
        reasoning_end_token_ids=[2, 3],
    )
    stale = [[1, 5, 6]]

    assert _run(processor, stale)[0, 0, 2] == 0

    # Stale view: forced first end token not visible; must force the SECOND
    # end token, not the first again.
    logits = _run(processor, stale)
    assert logits[0, 0, 3] == 0 and torch.isneginf(logits[0, 0, 2])

    # Sequence complete: no further forcing even while the view is stale.
    assert torch.equal(_run(processor, stale), torch.zeros(1, 1, 8))


def test_add_thinking_budget_logits_processor_uses_reasoning_parser_tokens():
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return {
                "<think>": [1],
                "</think>": [2],
            }[text]

    sampling_params = SamplingParams(thinking_token_budget=4)

    add_thinking_budget_logits_processor(
        sampling_params,
        reasoning_parser="qwen3",
        tokenizer=FakeTokenizer(),
    )

    assert isinstance(sampling_params.logits_processor, ThinkingBudgetLogitsProcessor)


@pytest.mark.parametrize("thinking_token_budget", [0, 2])
def test_thinking_budget_text_generation_caps_reasoning_tokens(thinking_token_budget):
    class FakeTokenizer:
        token_to_id = {
            "Question:": 0,
            "<think>": 1,
            "</think>": 2,
            "alpha": 3,
            "beta": 4,
            "gamma": 5,
            "delta": 6,
            "answer": 7,
        }
        id_to_token = {token_id: token for token, token_id in token_to_id.items()}

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [self.token_to_id[token] for token in text.split()]

        def decode(self, token_ids):
            return " ".join(self.id_to_token[token_id] for token_id in token_ids)

    tokenizer = FakeTokenizer()
    sampling_params = SamplingParams(thinking_token_budget=thinking_token_budget)
    add_thinking_budget_logits_processor(
        sampling_params,
        reasoning_parser="qwen3",
        tokenizer=tokenizer,
    )

    prompt_ids = tokenizer.encode("Question:")
    generated_ids = []
    reasoning_plan = tokenizer.encode("alpha beta gamma delta")
    start_id = tokenizer.encode("<think>")[0]
    end_id = tokenizer.encode("</think>")[0]
    answer_id = tokenizer.encode("answer")[0]

    def next_model_token():
        if not generated_ids:
            return start_id
        if end_id in generated_ids:
            return answer_id
        reasoning_tokens = generated_ids[generated_ids.index(start_id) + 1 :]
        if len(reasoning_tokens) < len(reasoning_plan):
            return reasoning_plan[len(reasoning_tokens)]
        return end_id

    for _ in range(8):
        logits = torch.full((1, 1, len(tokenizer.token_to_id)), -100.0)
        logits[0, 0, next_model_token()] = 100.0
        sampling_params.logits_processor(0, logits, [prompt_ids + generated_ids], None, None)
        generated_ids.append(int(torch.argmax(logits[0, 0]).item()))
        if generated_ids[-1] == answer_id:
            break

    start_idx = generated_ids.index(start_id)
    end_idx = generated_ids.index(end_id)
    reasoning_ids = generated_ids[start_idx + 1 : end_idx]
    expected_reasoning_ids = reasoning_plan[:thinking_token_budget]

    assert tokenizer.decode(prompt_ids) == "Question:"
    assert reasoning_ids == expected_reasoning_ids
    assert len(reasoning_ids) <= thinking_token_budget
    assert generated_ids[end_idx + 1 :] == [answer_id]
    assert tokenizer.decode(generated_ids) == tokenizer.decode(
        [start_id, *expected_reasoning_ids, end_id, answer_id]
    )
