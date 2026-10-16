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
"""Literal and checkpoint-oracle checks for DeepSeek-V4.1 chat formatting.

Literal rendering and wrapper checks run without a checkpoint. Differential and
vendor-golden checks require the checkpoint's encoding files.
"""

import copy
import functools
import importlib.util
import json
import os
from pathlib import Path

import pytest

from tensorrt_llm.inputs.utils import apply_chat_template
from tensorrt_llm.tokenizer.deepseek_v41 import DeepseekV41Tokenizer, encode_messages

# Pure-Python rendering checks with no GPU dependency. The l0_cpu stage runs
# pytest with ``-m cpu_only`` and conftest drops any file lacking this marker.
pytestmark = pytest.mark.cpu_only

MODELS_ROOT = Path(os.environ.get("LLM_MODELS_ROOT", "/code/llm-models"))
ENCODING_DIR = MODELS_ROOT / "DeepSeek-V4.1-Flash" / "encoding"

REASONING_PREAMBLE_75 = (
    "<｜System｜>Reasoning Effort: 75 "
    "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
)


class _DummyTokenizer:
    """Stand-in so rendering is testable without checkpoint weights."""

    all_special_tokens = []
    eos_token_id = 1
    pad_token_id = 0
    name_or_path = "dummy"

    def encode(self, text, *args, **kwargs):
        self.last_encoded_text = text
        self.last_encode_kwargs = kwargs
        return [1, 2, 3]


# ---------------------------------------------------------------------------
# Delta 1: the reasoning-effort budget is numeric and emitted by default
# ---------------------------------------------------------------------------


def test_thinking_mode_emits_the_numeric_reasoning_effort_preamble_by_default():
    """``reasoning_effort=None`` means "high" (75), not "omit".

    This is the delta that cost accuracy: V4 emitted nothing here unless the
    caller asked for ``"max"``, so a V4-rendered V4.1 thinking prompt is missing
    a system message the model was trained to see.
    """
    prompt = encode_messages(
        [{"role": "user", "content": "What is 2+2?"}], thinking_mode="thinking"
    )

    assert prompt == (
        "<｜begin▁of▁sentence｜>"
        + REASONING_PREAMBLE_75
        + "<｜User｜>What is 2+2?<｜Assistant｜><think>"
    )


def test_chat_mode_emits_no_reasoning_effort_preamble():
    prompt = encode_messages(
        [{"role": "user", "content": "q"}], thinking_mode="chat", reasoning_effort="max"
    )

    assert "Reasoning Effort" not in prompt
    assert prompt == "<｜begin▁of▁sentence｜><｜User｜>q<｜Assistant｜></think>"


@pytest.mark.parametrize("effort", [0, 101, -1, "medium", "", True])
def test_out_of_range_effort_is_rejected(effort):
    """A silently-clamped budget would be a wrong prompt with no signal."""
    with pytest.raises(ValueError, match="Invalid reasoning effort"):
        encode_messages(
            [{"role": "user", "content": "q"}], thinking_mode="thinking", reasoning_effort=effort
        )


# ---------------------------------------------------------------------------
# Deltas 2 and 3: the system token, and mid-conversation system messages
# ---------------------------------------------------------------------------


def test_leading_system_message_is_preceded_by_the_system_token():
    prompt = encode_messages(
        [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}],
        thinking_mode="chat",
    )

    assert prompt == (
        "<｜begin▁of▁sentence｜><｜System｜>You are terse.<｜User｜>Hi<｜Assistant｜></think>"
    )


def test_mid_conversation_system_message_is_a_generation_boundary():
    """V4.1 treats a mid-conversation system message like a user turn.

    It gets its own ``<|System|>`` token *and* triggers the assistant header, so
    the model is prompted to respond to it. V4 recognised neither.
    """
    prompt = encode_messages(
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "system", "content": "New rule."},
        ],
        thinking_mode="chat",
    )

    assert prompt.endswith("<｜System｜>New rule.<｜Assistant｜></think>")


# ---------------------------------------------------------------------------
# Deltas 4 and 5: DSML tag names and namespace-qualified tool names
# ---------------------------------------------------------------------------

_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]


def test_dsml_tag_names_carry_their_leading_spaces():
    """The DSML tag names' leading spaces are part of the names.

    ``" calls"``/``" invoke"``/``" parameter"`` tokenize differently from V4's
    unspaced ``tool_calls``/``invoke``/``parameter``.
    """
    prompt = encode_messages(
        [
            {"role": "system", "content": "", "tools": _TOOL},
            {"role": "user", "content": "weather?"},
        ],
        thinking_mode="chat",
    )

    assert "<｜DSML｜ calls>" in prompt
    assert '<｜DSML｜ invoke name="$TOOL_NAME">' in prompt
    assert "<｜DSML｜ parameter name=" in prompt
    # The V4 spellings must not appear at all.
    assert "<｜DSML｜tool_calls>" not in prompt
    assert "<｜DSML｜invoke" not in prompt


def test_rendered_tool_call_uses_the_v41_block_and_tag_names():
    prompt = encode_messages(
        [
            {"role": "user", "content": "weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris", "n": 3}',
                        },
                    }
                ],
            },
        ],
        thinking_mode="chat",
    )

    assert '<｜DSML｜ invoke name="get_weather">' in prompt
    assert ('<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>') in prompt
    # Non-string arguments are JSON-encoded and flagged string="false".
    assert ('<｜DSML｜ parameter name="n" string="false">3</｜DSML｜ parameter>') in prompt


def test_tool_names_are_namespace_qualified():
    prompt = encode_messages(
        [
            {
                "role": "system",
                "content": "S",
                "tools": [
                    {
                        "type": "function",
                        "namespace": {"name": "ns", "description": "NS desc"},
                        "function": {
                            "name": "do_it",
                            "description": "D",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
            {"role": "user", "content": "go"},
        ],
        thinking_mode="chat",
    )

    assert '"name": "ns::do_it"' in prompt
    # The namespace description is prepended to the tool description.
    assert "NS desc\\nD" in prompt


# ---------------------------------------------------------------------------
# Delta 6, and the boundaries this backend refuses to guess at
# ---------------------------------------------------------------------------


def test_developer_role_is_rejected():
    """V4 accepted ``developer``; V4.1 has no such role."""
    with pytest.raises(NotImplementedError, match="developer"):
        encode_messages([{"role": "developer", "content": "x"}], thinking_mode="chat")


def test_assistant_reasoning_is_read_only_from_reasoning_content():
    """V4 also accepted a ``reasoning`` key; V4.1 does not.

    Reading it would inject text the checkpoint's own encoder ignores.
    """
    kwargs = dict(thinking_mode="thinking", drop_thinking=False)
    with_legacy_key = encode_messages(
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B", "reasoning": "IGNORED"},
        ],
        **kwargs,
    )
    assert "IGNORED" not in with_legacy_key

    with_v41_key = encode_messages(
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B", "reasoning_content": "KEPT"},
        ],
        **kwargs,
    )
    assert "KEPT" in with_v41_key


@pytest.mark.parametrize("block_type", ["image", "image_url", "input_image"])
def test_image_content_raises_rather_than_being_dropped(block_type):
    """A dropped image is a silently wrong prompt.

    V4.1's vision tower is not part of this backend, so the honest response is to
    refuse rather than to render a text-only prompt that looks fine.
    """
    with pytest.raises(NotImplementedError, match="image input is not supported"):
        encode_messages(
            [{"role": "user", "content": [{"type": block_type, "url": "u"}]}],
            thinking_mode="chat",
        )


def test_invalid_task_is_rejected():
    with pytest.raises(ValueError, match="Invalid DeepSeek-V4.1 task"):
        encode_messages([{"role": "user", "content": "x", "task": "nope"}], thinking_mode="chat")


# ---------------------------------------------------------------------------
# The tokenizer wrapper
# ---------------------------------------------------------------------------


def test_apply_chat_template_maps_thinking_flags_to_thinking_mode():
    tokenizer = DeepseekV41Tokenizer(_DummyTokenizer())
    messages = [{"role": "user", "content": "q"}]

    for kwargs in ({"thinking": True}, {"enable_thinking": True}):
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
        assert "<think>" in prompt
        assert REASONING_PREAMBLE_75 in prompt
    assert "</think>" in tokenizer.apply_chat_template(messages)


@pytest.mark.parametrize(
    "effort, budget", [("low", 50), ("high", 75), ("max", 100), (1, 1), (42, 42), (100, 100)]
)
def test_apply_chat_template_passes_through_reasoning_effort(effort, budget):
    """Named and integer budgets survive both the wrapper and renderer."""
    tokenizer = DeepseekV41Tokenizer(_DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "q"}], thinking=True, reasoning_effort=effort
    )

    assert f"Reasoning Effort: {budget} (range 1-100," in prompt


def test_apply_chat_template_tokenizes_the_rendered_prompt():
    dummy = _DummyTokenizer()
    tokenizer = DeepseekV41Tokenizer(dummy)

    token_ids = tokenizer.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=True)

    assert token_ids == [1, 2, 3]
    assert dummy.last_encoded_text == (
        "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>"
    )
    # The rendering already contains the special tokens as text; re-adding them
    # would double the BOS.
    assert dummy.last_encode_kwargs["add_special_tokens"] is False


def test_tools_argument_is_folded_into_a_leading_system_message():
    tokenizer = DeepseekV41Tokenizer(_DummyTokenizer())

    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "weather?"}], tools=_TOOL)

    assert prompt.startswith("<｜begin▁of▁sentence｜><｜System｜>")
    assert "## Tools" in prompt


def test_server_chat_template_path_uses_v41_custom_tokenizer():
    tokenizer = DeepseekV41Tokenizer(_DummyTokenizer())

    prompt = apply_chat_template(
        model_type="deepseek_v41",
        tokenizer=tokenizer,
        processor=None,
        conversation=[{"role": "user", "content": "hello"}],
        add_generation_prompt=True,
        mm_placeholder_counts=[{}],
        tools=_TOOL,
        chat_template_kwargs={"thinking": True, "reasoning_effort": "max"},
    )

    assert "Reasoning Effort: 100" in prompt
    assert "## Tools" in prompt
    assert prompt.endswith("<｜Assistant｜><think>")


# ---------------------------------------------------------------------------
# Checkpoint differential: the renderer vs the checkpoint's own encoder
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _load_vendor_encoding():
    """Import the checkpoint's ``encoding.py`` under a private module name.

    Loaded by file path rather than by putting the directory on ``sys.path``:
    the module is named ``encoding``, which is generic enough to shadow an
    unrelated import for the rest of the session.
    """
    path = ENCODING_DIR / "encoding.py"
    if not path.exists():
        pytest.skip(
            f"the DeepSeek-V4.1 checkpoint encoder is not available at {path}; "
            f"set LLM_MODELS_ROOT to run the checkpoint differential"
        )
    spec = importlib.util.spec_from_file_location("_dsv41_vendor_encoding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TOOL_NS = [
    {
        "type": "function",
        "namespace": {"name": "ns", "description": "NS desc"},
        "function": {
            "name": "do_it",
            "description": "D",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
_TOOL_CALLS = [
    {
        "id": "c1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Paris", "n": 3}'},
    }
]

# Conversation shapes, chosen to cover each branch of the renderer rather than to
# look realistic: every role, tasks, tool results, wo_eos, response_format and
# structured content blocks.
DIFFERENTIAL_CASES = {
    "single_user": [{"role": "user", "content": "What is 2+2?"}],
    "sys_user": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hi"},
    ],
    "multiturn": [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"},
    ],
    "asst_reason": [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B", "reasoning_content": "because"},
        {"role": "user", "content": "C"},
    ],
    "mid_system": [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "system", "content": "New rule."},
    ],
    "tools_sys": [
        {"role": "system", "content": "", "tools": _TOOL},
        {"role": "user", "content": "weather?"},
    ],
    "tools_namespaced": [
        {"role": "system", "content": "S", "tools": _TOOL_NS},
        {"role": "user", "content": "go"},
    ],
    "tool_call_and_result": [
        {"role": "user", "content": "weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": _TOOL_CALLS},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
    ],
    "two_tool_results": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": _TOOL_CALLS},
        {"role": "tool", "tool_call_id": "c1", "content": "a"},
        {"role": "tool", "tool_call_id": "cX", "content": "b"},
    ],
    "latest_reminder": [
        {"role": "user", "content": "A"},
        {"role": "latest_reminder", "content": "R"},
    ],
    "task_action": [{"role": "user", "content": "A", "task": "action"}],
    "task_query": [{"role": "user", "content": "A", "task": "query"}],
    "wo_eos": [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B", "wo_eos": True},
    ],
    "response_format": [
        {"role": "system", "content": "S", "response_format": {"type": "json_object"}},
        {"role": "user", "content": "go"},
    ],
    "content_blocks": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "p1"}, {"type": "text", "text": "p2"}],
        }
    ],
}


@pytest.mark.parametrize(
    "case_name, thinking_mode, drop_thinking",
    [(name, mode, True) for name in DIFFERENTIAL_CASES for mode in ("chat", "thinking")]
    + [
        (name, "thinking", False)
        for name in (
            "multiturn",
            "asst_reason",
            "mid_system",
            "tool_call_and_result",
            "two_tool_results",
        )
    ],
)
def test_matches_the_checkpoint_encoder(case_name, thinking_mode, drop_thinking):
    """Cover both modes for every shape, plus retained reasoning in prior turns."""
    vendor = _load_vendor_encoding()
    messages = DIFFERENTIAL_CASES[case_name]
    kwargs = dict(thinking_mode=thinking_mode, drop_thinking=drop_thinking)

    expected = vendor.encode_messages(copy.deepcopy(messages), **kwargs)
    actual = encode_messages(copy.deepcopy(messages), **kwargs)

    assert actual == expected


@pytest.mark.parametrize(
    "thinking_mode, add_default_bos_token, with_context",
    [
        (mode, bos, context)
        for mode in ("chat", "thinking")
        for bos, context in ((True, True), (False, True), (False, False))
    ],
)
def test_matches_the_checkpoint_encoder_with_context(
    thinking_mode, add_default_bos_token, with_context
):
    """``context`` shifts message indices and suppresses BOS.

    Both are easy to get subtly wrong -- an off-by-one in the index shift moves
    the reasoning-effort preamble onto the wrong message.
    """
    vendor = _load_vendor_encoding()
    context = (
        [
            {"role": "user", "content": "ctx q"},
            {"role": "assistant", "content": "ctx a", "reasoning_content": "r"},
        ]
        if with_context
        else None
    )
    messages = [{"role": "user", "content": "new q"}]
    kwargs = dict(
        thinking_mode=thinking_mode,
        reasoning_effort=25,
        add_default_bos_token=add_default_bos_token,
        context=copy.deepcopy(context),
    )

    expected = vendor.encode_messages(copy.deepcopy(messages), **kwargs)
    actual = encode_messages(copy.deepcopy(messages), **kwargs)

    assert actual == expected


def _vendor_fixture_ids():
    fixtures = ENCODING_DIR / "tests"
    if not fixtures.is_dir():
        return []
    return sorted(path.stem.split("_")[-1] for path in fixtures.glob("test_input_*.json"))


@pytest.mark.parametrize("case_id", _vendor_fixture_ids())
def test_matches_the_vendor_golden_fixtures(case_id):
    """Reproduce the vendor's checked-in golden prompts byte for byte.

    A stronger pin than the differential above: these files are the vendor's
    declared correct output, so they hold even if our reading of ``encoding.py``
    is wrong in the same way twice.
    """
    fixtures = ENCODING_DIR / "tests"
    golden = (fixtures / f"test_output_{case_id}.txt").read_text()
    case = json.loads((fixtures / f"test_input_{case_id}.json").read_text())

    # Mirror the vendor's own ``load_cases`` normalisation.
    if isinstance(case, dict):
        case = [case]
    elif case and isinstance(case[0], dict) and "role" in case[0]:
        case = [{"messages": case}]
    case = case[0]
    messages = copy.deepcopy(case["messages"])
    if "tools" in case:
        messages[0]["tools"] = case["tools"]

    try:
        actual = encode_messages(
            messages,
            thinking_mode=case.get("thinking_mode") or "chat",
            context=case.get("context"),
            reasoning_effort=case.get("reasoning_effort"),
        )
    except NotImplementedError as exc:
        if "image input is not supported" not in str(exc):
            raise
        pytest.skip(
            f"vendor fixture {case_id} is multimodal; this backend refuses "
            f"image input rather than rendering a text-only prompt from it"
        )

    assert actual == golden
