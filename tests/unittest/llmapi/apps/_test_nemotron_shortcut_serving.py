# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image
from test_common.llm_data import llm_models_root

from tensorrt_llm._torch.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version

from .openai_server import RemoteOpenAIServer

PRECISIONS = ("nvfp4",)
LABEL_PROMPT = 'Return the label "violet" as a compact JSON object with the key "label".'


@pytest.fixture(scope="class")
def server(request):
    precision, _, mode = request.param.partition("-")
    if get_sm_version() != 103:
        pytest.skip("Nemotron Shortcut serving is qualified on four GB300 GPUs")
    root = llm_models_root()
    if root is None or not (root / "nemotron_shortcut" / precision).is_dir():
        pytest.skip(f"Missing nemotron_shortcut/{precision} under LLM_MODELS_ROOT")
    checkpoint = root / "nemotron_shortcut" / precision

    config = {
        "modeling_v2": "require",
        "env_overrides": {MODELING_V2_VALIDATE_ENV: "1"},
        "trust_remote_code": True,
        "tensor_parallel_size": 4,
        "moe_tensor_parallel_size": 2,
        "moe_expert_parallel_size": 2,
        "max_seq_len": 1024,
        "max_num_tokens": 1024,
        "max_batch_size": 4,
        "disable_overlap_scheduler": False,
        "kv_cache_config": {
            "use_kv_cache_manager_v2": True,
            "enable_block_reuse": False,
        },
        "guided_decoding_backend": "xgrammar",
    }
    if mode == "mtp2":
        config["speculative_config"] = {"decoding_type": "MTP", "max_draft_len": 2}
    args = ["--reasoning_parser", "auto", "--tool_parser", "auto"]
    env = {**os.environ, MODELING_V2_VALIDATE_ENV: "1"}
    with RemoteOpenAIServer(
        str(checkpoint), args, env=env, extra_config=config, wait=False
    ) as endpoint:
        endpoint.wait_for_server(timeout=900)
        with endpoint.get_client() as client:
            yield client, str(checkpoint)


def _image_uri(image_path):
    png = io.BytesIO()
    with Image.open(image_path) as image:
        image.convert("RGB").resize((224, 224)).save(png, format="PNG")
    return "data:image/png;base64," + base64.b64encode(png.getvalue()).decode("ascii")


class _StructuredOutput:
    @pytest.mark.parametrize("thinking", [False, True], ids=["direct", "thinking"])
    def test_json_schema(self, server, thinking):
        client, model = server
        schema = {
            "type": "object",
            "properties": {"label": {"enum": ["violet"]}},
            "required": ["label"],
            "additionalProperties": False,
        }
        # The grammar constrains the answer after </think>. Give the reasoning
        # phase its input explicitly, while direct mode must infer it from the schema.
        prompt = LABEL_PROMPT if thinking else "Reply with the label in JSON."
        result = client.chat.completions.create(
            model=model,
            temperature=0,
            max_completion_tokens=384 if thinking else 64,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "label", "schema": schema},
            },
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )
        choice = result.choices[0]
        assert choice.finish_reason == "stop", result.usage
        assert json.loads(choice.message.content) == {"label": "violet"}


@pytest.mark.parametrize("server", PRECISIONS, indirect=True)
class TestServing(_StructuredOutput):
    @pytest.mark.parametrize("budget", [0, 8])
    def test_thinking_budget(self, server, budget, record_property):
        client, model = server
        result = client.chat.completions.create(
            model=model,
            temperature=0,
            logprobs=True,
            max_completion_tokens=64,
            messages=[{"role": "user", "content": "What is two plus two? Answer with a digit."}],
            extra_body={
                "thinking_token_budget": budget,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
        choice = result.choices[0]
        message = choice.message
        tokens = [item.token for item in choice.logprobs.content]
        record_property("sampled_tokens", json.dumps(tokens))
        # The generation starts inside <think>. Count sampled tokens up to the
        # closing marker, including tokenizers that split the marker across tokens.
        assert "".join(tokens[budget:]).startswith("</think>")
        assert "</think>" not in "".join(tokens[:budget])
        assert bool((message.reasoning_content or "").strip()) == bool(budget)
        assert "4" in message.content and "</think>" not in message.content

    def test_streaming_reasoning(self, server):
        client, model = server
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is two plus two? Answer with a digit."}],
            temperature=0,
            max_completion_tokens=64,
            stream=True,
            extra_body={
                "thinking_token_budget": 8,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
        reasoning = []
        content = []
        with stream:
            for event in stream:
                for choice in event.choices:
                    reasoning.append(getattr(choice.delta, "reasoning_content", None) or "")
                    content.append(choice.delta.content or "")
        assert "".join(reasoning).strip()
        answer = "".join(content)
        assert "4" in answer and "<think>" not in answer and "</think>" not in answer

    def test_image_tool_roundtrip(self, server):
        client, model = server
        data_uri = _image_uri(Path(__file__).parents[4] / "examples/visual_gen/cat_piano.png")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {
                        "type": "text",
                        "text": "Call lookup_label with the animal in the picture. "
                        "Then tell me the exact label returned by the tool.",
                    },
                ],
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_label",
                    "description": "Look up a label for an animal.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "animal": {"type": "string", "enum": ["dog", "cat", "bird", "horse"]}
                        },
                        "required": ["animal"],
                    },
                },
            }
        ]
        options = {
            "model": model,
            "temperature": 0,
            "max_completion_tokens": 128,
            "tools": tools,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        first = client.chat.completions.create(
            messages=messages, tool_choice="required", **options
        ).choices[0]
        assert first.finish_reason == "tool_calls"
        (call,) = first.message.tool_calls
        assert call.id and call.function.name == "lookup_label"
        assert json.loads(call.function.arguments) == {"animal": "cat"}
        messages.extend(
            [
                first.message.model_dump(exclude_none=True),
                {"role": "tool", "tool_call_id": call.id, "content": "GARNET-418"},
            ]
        )
        answer = client.chat.completions.create(messages=messages, tool_choice="none", **options)
        assert "GARNET-418" in answer.choices[0].message.content


@pytest.mark.parametrize("server", [f"{p}-mtp2" for p in PRECISIONS], indirect=True)
class TestMtpServing(_StructuredOutput):
    """Structured output with two MTP draft tokens."""
