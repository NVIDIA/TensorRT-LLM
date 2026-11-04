# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guided decoding with llguidance over chat token ids a serving frontend renders itself.

A frontend that renders Mistral chats itself hands the engine token ids with either a JSON
schema (``response_format``) or the Lark tool-call grammar mistral-common's grammar factory writes
(tool_choice auto / required / named). Each case sends the same, in one batch, and reads the
generated token ids.

Output the grammar forbids fails the case: a value outside a one-value enum, a missing reasoning
section under reasoning effort high, a reply with no call under a forced tool choice, a call to
another tool. A reply cut off by ``max_tokens`` before its answer, and the model's choice not to call
under ``auto``, are INCONCLUSIVE.
"""

import json
import time
from pathlib import Path

from tensorrt_llm import SamplingParams
from tensorrt_llm.sampling_params import GuidedDecodingParams

# A value no prompt asks for: only the grammar writes it.
_KEY = "zq-7f3k"
_KEY_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "string", "enum": [_KEY]}},
    "required": ["value"],
    "additionalProperties": False,
}
_CITY_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "population_millions": {"type": "number"}},
    "required": ["city", "population_millions"],
    "additionalProperties": False,
}
_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"enum": ["c", "f"]}},
            "required": ["city", "unit"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}
_RECORD_TOOL = {
    "type": "function",
    "function": {
        "name": "record_key",
        "description": "Store the session key.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string", "enum": [_KEY]}},
            "required": ["key"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}
_WEATHER = "What is the weather in Paris right now, in celsius?"

# name, user message, tools, reasoning effort, constraint (a JSON schema, or a tool choice),
# max_tokens.
_CASES = (
    ("json_low", "Name the capital of France. Answer in JSON.", None, "low", _KEY_SCHEMA, 2048),
    (
        "json_after_reasoning",
        "Which city has the largest metropolitan population? Answer in JSON.",
        None,
        "high",
        _CITY_SCHEMA,
        8192,
    ),
    ("tool_required", "Store the session key now.", [_RECORD_TOOL], "low", "required", 2048),
    ("tool_named", _WEATHER, [_WEATHER_TOOL, _RECORD_TOOL], "low", "record_key", 2048),
    ("tool_auto", _WEATHER, [_WEATHER_TOOL], "high", "auto", 4096),
)


def _prompt(llm, message, tools, effort):
    from mistral_common.protocol.instruct.request import ChatCompletionRequest

    request = ChatCompletionRequest.from_openai(
        messages=[{"role": "user", "content": message}], tools=tools, reasoning_effort=effort
    )
    return {"prompt_token_ids": llm.tokenizer.mistral.encode_chat_completion(request).tokens}


def _tool_grammar(llm, tools, choice):
    from mistral_common.guidance.grammar_factory import GrammarFactory
    from mistral_common.protocol.instruct.tool_calls import NamedToolChoice, Tool, ToolChoiceEnum

    factory = GrammarFactory(llm.tokenizer.mistral)
    native_tools = [Tool.from_openai(tool) for tool in tools]
    if choice in ("auto", "required"):
        mode, force_tool_names = ToolChoiceEnum(choice), False
    else:
        # A named choice, as mistral-common's get_grammar_for_chat_request resolves it.
        name = NamedToolChoice.model_validate(
            {"type": "function", "function": {"name": choice}}
        ).function.name
        native_tools = [tool for tool in native_tools if tool.function.name == name]
        mode, force_tool_names = ToolChoiceEnum.required, True
    # A named choice is exactly one call, as the serving extension resolves it.
    return factory.get_lark_from_jinja(
        factory.select_jinja_template(),
        mode=mode,
        tools=native_tools,
        json_schema=None,
        parallel_tool_calls=choice in ("auto", "required"),
        json_only=False,
        force_tool_names=force_tool_names,
    )


def _split(llm, ids):
    """The reasoning, content and calls of generated ids, or an error string."""
    control = llm.tokenizer.tokenizer.get_control_token
    think, end_think = control("[THINK]"), control("[/THINK]")
    tool_calls, args = control("[TOOL_CALLS]"), control("[ARGS]")
    ids = [i for i in ids if i != llm.tokenizer.eos_token_id]
    reasoning = None
    if ids and ids[0] == think:
        if end_think not in ids:
            return None, "unterminated reasoning section"
        end = ids.index(end_think)
        reasoning, ids = ids[1:end], ids[end + 1 :]
    if think in ids or end_think in ids:
        return None, "reasoning marker outside a leading reasoning section"
    starts = [n for n, i in enumerate(ids) if i == tool_calls]
    content = ids[: starts[0]] if starts else ids
    calls = []
    for start, stop in zip(starts, starts[1:] + [len(ids)]):
        call = ids[start + 1 : stop]
        if args not in call:
            return None, "a call without [ARGS]"
        cut = call.index(args)
        # The tool grammar allows whitespace before the name, as the model writes it under a forced
        # call to a tool the prompt does not ask for.
        name = llm.tokenizer.decode(call[:cut]).strip()
        calls.append((name, llm.tokenizer.decode(call[cut + 1 :])))
    decode = llm.tokenizer.decode
    return {
        "reasoning": None if reasoning is None else decode(reasoning),
        "content": decode(content),
        "calls": calls,
    }, None


def _check(name, constraint, parts, finish_reason):
    """FAIL, INCONCLUSIVE or None (pass) for one case."""
    cut = finish_reason == "length"
    if isinstance(constraint, dict):
        if name == "json_after_reasoning" and parts["reasoning"] is None:
            return "FAIL: no reasoning section under reasoning effort high"
        try:
            answer = json.loads(parts["content"])
        except json.JSONDecodeError:
            if cut:
                return "INCONCLUSIVE: cut off by max_tokens before the JSON"
            return f"FAIL: not JSON: {parts['content']!r}"
        if constraint is _KEY_SCHEMA:
            return None if answer == {"value": _KEY} else f"FAIL: outside the schema: {answer!r}"
        ok = (
            isinstance(answer, dict)
            and set(answer) == {"city", "population_millions"}
            and isinstance(answer["city"], str)
            and isinstance(answer["population_millions"], (int, float))
        )
        return None if ok else f"FAIL: outside the schema: {answer!r}"
    if not parts["calls"]:
        if constraint == "auto":
            return "INCONCLUSIVE: no call under auto"
        if cut:
            return "INCONCLUSIVE: cut off by max_tokens before the forced call"
        return f"FAIL: no call under tool_choice {constraint}"
    for tool, arguments in parts["calls"]:
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return f"FAIL: {tool} arguments are not JSON: {arguments!r}"
        if constraint == "auto":
            ok = (
                tool == "get_weather"
                and isinstance(arguments, dict)
                and set(arguments) == {"city", "unit"}
                and isinstance(arguments["city"], str)
                and arguments["unit"] in ("c", "f")
            )
        else:
            ok = tool == "record_key" and arguments == {"key": _KEY}
        if not ok:
            return f"FAIL: call outside the grammar: {tool} {arguments!r}"
    return None


def evaluate(llm, out: Path):
    started = time.monotonic()
    prompts, params = [], []
    for name, message, tools, effort, constraint, max_tokens in _CASES:
        prompts.append(_prompt(llm, message, tools, effort))
        guided = (
            GuidedDecodingParams(json=constraint)
            if isinstance(constraint, dict)
            else GuidedDecodingParams(grammar=_tool_grammar(llm, tools, constraint))
        )
        params.append(SamplingParams(max_tokens=max_tokens, temperature=0, guided_decoding=guided))
    outputs = llm.generate(prompts, params, use_tqdm=False)
    failures = []
    with (out / "llguidance.jsonl").open("w") as stream:
        for (name, _, _, _, constraint, _), output in zip(_CASES, outputs):
            completion = output.outputs[0]
            parts, error = _split(llm, list(completion.token_ids))
            verdict = (
                f"FAIL: {error}"
                if error
                else _check(name, constraint, parts, completion.finish_reason)
            )
            if verdict:
                failures.append(f"{verdict.split(':')[0]}: {name}: {verdict.split(': ', 1)[1]}")
            stream.write(
                json.dumps(
                    {
                        "name": name,
                        "finish_reason": completion.finish_reason,
                        "verdict": verdict or "PASS",
                        "parts": parts,
                        "text": llm.tokenizer.decode(
                            list(completion.token_ids), skip_special_tokens=False
                        ),
                    }
                )
                + "\n"
            )
    return {"seconds": time.monotonic() - started, "cases": len(_CASES), "failures": failures}
