# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MuseSparkOpen input processor with the prefix-tokenization cache.

With ``enable_tokenization_cache`` the processor's text-only path splices the
token ids of a cached earlier prompt with a freshly tokenized tail. The ids
must equal what the uncached path produces, turn by turn, for the prompt
shapes an agentic conversation has under this checkpoint's chat template:
multi-turn growth, consecutive assistant tool-call messages (whose end token
changes once another message follows), reasoning blocks, tool results, and the
``Current date`` line at the top of the system message.

Runs against the real checkpoint's tokenizer and template, like
``tests/unittest/llmapi/test_musespark_open_tokenizer.py``; it is skipped when
``MUSESPARK_REFERENCE_CKPT`` is unset.
"""

import os

import pytest

from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only

CHECKPOINT = os.environ.get("MUSESPARK_REFERENCE_CKPT", "")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command and return its output. " * 20,
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}, "timeout": {"type": "number"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file from the local filesystem. " * 20,
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        },
    },
]

FILLER = (
    "Line {i}: the build log reports 1234567 bytes copied in 0.25s; "
    "naïve café ünïcode, tabs\tand  double  spaces, and a URL "
    "https://example.com/a?b=c&d=e#f.\n"
)


@pytest.fixture(scope="module")
def tokenizer():
    if not CHECKPOINT or not os.path.isdir(CHECKPOINT):
        pytest.skip("checkpoint not present; set MUSESPARK_REFERENCE_CKPT")
    from tensorrt_llm.tokenizer import load_custom_tokenizer

    return load_custom_tokenizer("musespark_open", CHECKPOINT)


def _processor(tokenizer, enable):
    from tensorrt_llm.inputs.registry import create_input_processor

    return create_input_processor(
        CHECKPOINT, tokenizer, trust_remote_code=True, enable_tokenization_cache=enable
    )


def _render(tokenizer, messages, current_date="2026-10-20"):
    return tokenizer.tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
        tokenize=False,
        current_date=current_date,
    )


def _filler(start, count):
    return "".join(FILLER.format(i=i) for i in range(start, start + count))


def _tool_call(call_id, name, **arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _tool_result(call_id, step):
    return {"role": "tool", "tool_call_id": call_id, "content": _filler(step * 7, 30)}


def _agentic_turns():
    """Message lists of one growing agentic conversation, one per request."""
    messages = [
        {"role": "system", "content": "You are a coding agent. " + "Be careful. " * 400},
        {"role": "user", "content": "Fix the failing test.\n" + _filler(0, 200)},
    ]
    turns = [list(messages)]
    for step in range(12):
        if step % 4 == 1:
            # Two tool calls emitted as consecutive assistant messages. The
            # first one ends with <|eot|> while it is the last message and with
            # <|eom|> once the second follows, so the next prompt does not
            # extend this one character for character.
            messages.append(
                {
                    "role": "assistant",
                    "reasoning_content": f"I should list files (step {step}).",
                    "content": "",
                    "tool_calls": [_tool_call(f"a{step}", "Bash", command=f"ls -la dir{step}")],
                }
            )
            turns.append(list(messages))
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tool_call(f"b{step}", "Read", file_path=f"/src/f{step}.py")],
                }
            )
            messages.append(_tool_result(f"a{step}", step))
            messages.append(_tool_result(f"b{step}", step))
        elif step % 4 == 3:
            # A reasoning block plus a final answer, then a new user turn.
            messages.append(
                {
                    "role": "assistant",
                    "reasoning_content": "Thinking " * 50,
                    "content": f"Done with step {step}.",
                }
            )
            messages.append({"role": "user", "content": f"Next: step{step + 1}" + "x" * step})
        else:
            messages.append(
                {
                    "role": "assistant",
                    "reasoning_content": f"Run the tests ({step}).",
                    "content": "Running the tests now.",
                    "tool_calls": [
                        _tool_call(f"c{step}", "Bash", command="pytest -x", timeout=600)
                    ],
                }
            )
            messages.append(_tool_result(f"c{step}", step))
        turns.append(list(messages))
    return turns


def test_cache_is_wired_only_when_enabled(tokenizer):
    from tensorrt_llm._torch.models.modeling_musespark_open_omni import (
        MuseSparkOpenOmniInputProcessor,
    )

    on = _processor(tokenizer, True)
    off = _processor(tokenizer, False)
    assert isinstance(on, MuseSparkOpenOmniInputProcessor)
    assert on._prefix_token_cache is not None
    assert off._prefix_token_cache is None


def test_agentic_turns_are_id_identical(tokenizer):
    on = _processor(tokenizer, True)
    off = _processor(tokenizer, False)
    params = SamplingParams(add_special_tokens=False)
    turns = _agentic_turns()
    for turn, messages in enumerate(turns):
        prompt = _render(tokenizer, messages)
        expected = tokenizer.encode(prompt, add_special_tokens=False)
        cached, extra = on({"prompt": prompt}, params)
        uncached, _ = off({"prompt": prompt}, params)
        assert extra is None
        assert uncached == expected, f"turn {turn}: uncached path changed"
        assert cached == expected, f"turn {turn}: cached ids differ"
    cache = on._prefix_token_cache
    assert not cache.disabled
    # Every turn but the first and the three that follow a lone assistant
    # tool call extends the previous prompt, so the cache must splice them.
    assert cache.hits >= len(turns) - 4, cache.summary()


def test_date_change_misses_but_stays_identical(tokenizer):
    on = _processor(tokenizer, True)
    params = SamplingParams(add_special_tokens=False)
    turns = _agentic_turns()
    for date, messages in (("2026-10-20", turns[3]), ("2026-10-21", turns[4])):
        prompt = _render(tokenizer, messages, current_date=date)
        cached, _ = on({"prompt": prompt}, params)
        assert cached == tokenizer.encode(prompt, add_special_tokens=False)
    # The date sits at the top of the system message, so the second prompt
    # does not extend the first one.
    assert on._prefix_token_cache.hits == 0


@pytest.mark.parametrize("stride_offset", [1, 7, 31, 97, 1009])
def test_appends_starting_mid_word_are_identical(tokenizer, stride_offset):
    on = _processor(tokenizer, True)
    params = SamplingParams(add_special_tokens=False)
    full = _render(tokenizer, _agentic_turns()[-1])
    for end in range(len(full) // 2, len(full), len(full) // 40 + stride_offset):
        prompt = full[:end]
        cached, _ = on({"prompt": prompt}, params)
        assert cached == tokenizer.encode(prompt, add_special_tokens=False), end
    assert on._prefix_token_cache.hits > 0


def test_cache_is_bypassed_when_special_tokens_are_requested(tokenizer):
    on = _processor(tokenizer, True)
    prompt = _render(tokenizer, _agentic_turns()[2])
    ids, _ = on({"prompt": prompt}, SamplingParams(add_special_tokens=True))
    # The processor never adds special tokens itself; the template carries BOS.
    assert ids == tokenizer.encode(prompt, add_special_tokens=False)
    cache = on._prefix_token_cache
    assert cache.hits + cache.misses == 0
