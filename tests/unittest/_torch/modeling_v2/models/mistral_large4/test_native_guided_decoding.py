# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guided decoding for mistral_large_4: xgrammar over the Tekken vocabulary, on by default.

xgrammar matches a grammar against each token's bytes. A Tekken token is raw bytes that
need not be valid UTF-8, so the vocabulary keeps them as bytes. Control tokens are empty
strings, which xgrammar treats as special: a grammar never selects them, so `[INST]` or
`[IMG]` cannot appear inside a JSON string, and end-of-sequence is the only stop token.

llguidance takes the Tekken tokenizer as mistral-common wraps it: ordinary tokens are
their bytes and control tokens are special, which a Lark grammar selects only by id
(`<[9]>` for `[TOOL_CALLS]`). That is the form of the tool-call and reasoning grammars
mistral-common's grammar factory renders, which a serving frontend may send as a
request's grammar.

These tests run on CPU with the Tekken file bundled with mistral-common, and with the
checkpoint's when MISTRAL_LARGE4_CHECKPOINT_DIR names it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor_creator import get_guided_decoding_config
from tensorrt_llm.llmapi.llm import _default_guided_decoding_backend
from tensorrt_llm.llmapi.llm_args import GuidedDecodingConfig

pytestmark = pytest.mark.cpu_only

_SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}


@pytest.fixture(scope="module", params=["bundled", "checkpoint"])
def tokenizer(request):
    mistral_common = pytest.importorskip("mistral_common")
    from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import MistralTokenizer

    if request.param == "bundled":
        files = sorted((Path(mistral_common.__file__).parent / "data").glob("tekken*.json"))
        if not files:
            pytest.skip("this mistral_common ships no tekken tokenizer file")
        return MistralTokenizer.from_pretrained(str(files[-1]))
    checkpoint = os.environ.get("MISTRAL_LARGE4_CHECKPOINT_DIR")
    if not checkpoint:
        pytest.skip("needs checkpoint tokenizer")
    return MistralTokenizer.from_pretrained(checkpoint)


def test_vocab_is_raw_bytes_with_control_tokens_empty(tokenizer):
    tekken = tokenizer.tokenizer
    info = tokenizer.xgrammar_tokenizer_info()
    vocab = info["encoded_vocab"]
    assert info["stop_token_ids"] == [tokenizer.eos_token_id]
    assert len(vocab) == tekken.n_words
    assert vocab[: tekken.num_special_tokens] == [""] * tekken.num_special_tokens
    pieces = [tekken.id_to_byte_piece(i) for i in range(tekken.num_special_tokens, tekken.n_words)]
    expected = ["" if piece.startswith(b"\x00") else piece for piece in pieces]
    assert vocab[tekken.num_special_tokens :] == expected
    # Some tokens are partial UTF-8 sequences: only bytes represent them.
    undecodable = 0
    for piece in pieces:
        try:
            piece.decode("utf-8")
        except UnicodeDecodeError:
            undecodable += 1
    assert undecodable > 0


def test_mistral_large_4_config_takes_the_byte_vocab(tokenizer):
    config = get_guided_decoding_config("xgrammar", tokenizer, checkpoint_format="mistral_large_4")
    assert config.backend == GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR
    assert config.encoded_vocab == tokenizer.xgrammar_tokenizer_info()["encoded_vocab"]
    assert config.stop_token_ids == [tokenizer.eos_token_id]
    assert get_guided_decoding_config(None, tokenizer, checkpoint_format="mistral_large_4") is None


def test_mistral_large_4_llguidance_config_defers_to_the_tokenizer(tokenizer):
    from tensorrt_llm._torch.pyexecutor.py_executor_creator import _llguidance_tokenizer

    config = get_guided_decoding_config(
        "llguidance", tokenizer, checkpoint_format="mistral_large_4"
    )
    assert config.backend == GuidedDecodingConfig.GuidedDecodingBackend.LLGUIDANCE
    assert config.tokenizer_str is None and config.encoded_vocab is None
    assert config.stop_token_ids == [tokenizer.eos_token_id]
    n_words = tokenizer.tokenizer.n_words
    assert _llguidance_tokenizer(config, tokenizer, "mistral_large_4", n_words) is not None
    xgrammar = get_guided_decoding_config(
        "xgrammar", tokenizer, checkpoint_format="mistral_large_4"
    )
    assert _llguidance_tokenizer(xgrammar, tokenizer, "mistral_large_4", n_words) is None
    with pytest.raises(ValueError, match="xgrammar or llguidance"):
        get_guided_decoding_config("outlines", tokenizer, checkpoint_format="mistral_large_4")


def _matcher(tokenizer, schema):
    from tensorrt_llm._torch.pyexecutor.grammar_matcher import XGrammarMatcherFactory
    from tensorrt_llm.bindings.executor import GuidedDecodingParams

    config = get_guided_decoding_config("xgrammar", tokenizer, checkpoint_format="mistral_large_4")
    factory = XGrammarMatcherFactory(config, vocab_size_padded=tokenizer.tokenizer.n_words)
    return factory.create(
        GuidedDecodingParams(GuidedDecodingParams.GuideType.JSON_SCHEMA, json.dumps(schema))
    )


@pytest.mark.parametrize("text", ['{"city": "Paris"}', '{"city": "巴黎 São Paulo 🙂"}'])
def test_xgrammar_accepts_the_tokens_of_a_valid_answer(tokenizer, text):
    matcher = _matcher(tokenizer, _SCHEMA)
    for token_id in tokenizer.tokenizer.encode(text, bos=False, eos=False):
        assert matcher.accept_token(token_id), token_id
    assert matcher.accept_token(tokenizer.eos_token_id)
    assert matcher.is_terminated()


def test_xgrammar_never_allows_control_tokens_inside_a_string(tokenizer):
    import xgrammar

    tekken = tokenizer.tokenizer
    matcher = _matcher(tokenizer, _SCHEMA)
    for token_id in tekken.encode('{"city": "', bos=False, eos=False):
        assert matcher.accept_token(token_id), token_id
    bitmask = xgrammar.allocate_token_bitmask(1, tekken.n_words)
    matcher.fill_next_token_bitmask(bitmask, 0)
    allowed = [
        token_id
        for token_id in range(tekken.n_words)
        if (int(bitmask[0, token_id // 32]) >> (token_id % 32)) & 1
    ]
    assert allowed, "a JSON string accepts ordinary text"
    assert min(allowed) >= tekken.num_special_tokens


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({}, "xgrammar"),
        ({"guided_decoding_backend": "llguidance"}, "llguidance"),
        ({"speculative_config": object()}, None),
        ({"mm_encoder_only": True}, None),
        ({"checkpoint_format": None}, None),
        ({"checkpoint_format": "mistral_large_3"}, None),
    ],
)
def test_mistral_large_4_serves_structured_outputs_by_default(fields, expected):
    args = SimpleNamespace(
        **{
            "guided_decoding_backend": None,
            "checkpoint_format": "mistral_large_4",
            "speculative_config": None,
            "mm_encoder_only": False,
            **fields,
        }
    )
    assert _default_guided_decoding_backend(args) == expected


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"enum": ["c", "f"]}},
            "required": ["city", "unit"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


@pytest.fixture(scope="module")
def llg(tokenizer):
    pytest.importorskip("llguidance")
    return tokenizer.llguidance_tokenizer(tokenizer.tokenizer.n_words)


def test_llguidance_tokenizer_is_tekken(tokenizer, llg):
    tekken = tokenizer.tokenizer
    assert llg.vocab_size == tekken.n_words
    assert llg.eos_token == tokenizer.eos_token_id
    assert all(llg.is_special_token(i) for i in range(tekken.num_special_tokens))
    # llguidance marks special tokens with a leading 0xFF byte, which no UTF-8 text has; the
    # raw 0xFF byte token reads as special too.
    ordinary = [
        i
        for i in range(tekken.num_special_tokens, 2000)
        if not tekken.id_to_byte_piece(i).startswith(b"\xff")
    ]
    assert len(ordinary) > 900
    assert not any(llg.is_special_token(i) for i in ordinary)
    for text in ("Hello, world!", '{"city": "巴黎 São Paulo 🙂"}'):
        ids = tekken.encode(text, bos=False, eos=False)
        assert llg.tokenize_str(text) == ids
        assert llg.decode_bytes(ids) == text.encode()


def _llg_matcher(tokenizer, llg, guide_type, guide):
    from tensorrt_llm._torch.pyexecutor.grammar_matcher import LLGuidanceMatcherFactory
    from tensorrt_llm.bindings.executor import GuidedDecodingParams

    config = get_guided_decoding_config(
        "llguidance", tokenizer, checkpoint_format="mistral_large_4"
    )
    factory = LLGuidanceMatcherFactory(config, tokenizer.tokenizer.n_words, tokenizer=llg)
    return factory.create(
        GuidedDecodingParams(getattr(GuidedDecodingParams.GuideType, guide_type), guide)
    )


def _tool_grammar(tokenizer, mode, json_schema=None):
    from mistral_common.guidance.grammar_factory import GrammarFactory
    from mistral_common.protocol.instruct.tool_calls import Tool, ToolChoiceEnum

    if not GrammarFactory.is_supported(tokenizer.mistral):
        pytest.skip("mistral-common renders tool grammars for Tekken v11+ only")
    factory = GrammarFactory(tokenizer.mistral)
    tools = [Tool.from_openai(_WEATHER_TOOL)]
    return factory.get_lark_from_jinja(
        factory.select_jinja_template(),
        mode=ToolChoiceEnum(mode),
        tools=tools,
        json_schema=json_schema,
        parallel_tool_calls=True,
        json_only=False,
        force_tool_names=False,
    )


def _control(tokenizer, name):
    return tokenizer.tokenizer.get_control_token(name)


def _consume(matcher, ids):
    for token_id in ids:
        if not matcher.accept_token(token_id):
            return False
    return True


def _allowed(matcher, n_words):
    import torch

    bitmask = torch.zeros(1, (n_words + 31) // 32, dtype=torch.int32)
    matcher.fill_next_token_bitmask(bitmask, 0)
    return {i for i in range(n_words) if (int(bitmask[0, i // 32]) >> (i % 32)) & 1}


def test_llguidance_json_schema_over_tekken(tokenizer, llg):
    matcher = _llg_matcher(tokenizer, llg, "JSON_SCHEMA", json.dumps(_SCHEMA))
    assert _consume(matcher, tokenizer.tokenizer.encode('{"city": "Paris"}', bos=False, eos=False))
    assert matcher.accept_token(tokenizer.eos_token_id)
    matcher = _llg_matcher(tokenizer, llg, "JSON_SCHEMA", json.dumps(_SCHEMA))
    assert _consume(matcher, tokenizer.tokenizer.encode('{"city": "', bos=False, eos=False))
    allowed = _allowed(matcher, tokenizer.tokenizer.n_words)
    assert allowed and min(allowed) >= tokenizer.tokenizer.num_special_tokens


def test_llguidance_tool_call_grammar_from_mistral_common(tokenizer, llg):
    """mistral-common's tool-call grammar opens a call with the [TOOL_CALLS] control
    token and checks the arguments against the strict tool's schema."""
    pytest.importorskip("jinja2")
    tekken = tokenizer.tokenizer
    _tool_grammar(
        tokenizer, "auto"
    )  # skips vocabularies mistral-common renders no tool grammar for
    call = (
        [_control(tokenizer, "[TOOL_CALLS]")]
        + tekken.encode("get_weather", bos=False, eos=False)
        + [_control(tokenizer, "[ARGS]")]
        + tekken.encode('{"city": "Paris", "unit": "c"}', bos=False, eos=False)
    )
    text = tekken.encode("It is sunny.", bos=False, eos=False)
    for mode in ("auto", "required"):
        grammar = _tool_grammar(tokenizer, mode)
        matcher = _llg_matcher(tokenizer, llg, "EBNF_GRAMMAR", grammar)
        assert _control(tokenizer, "[TOOL_CALLS]") in _allowed(matcher, tekken.n_words), mode
        assert _consume(matcher, call), mode
        assert matcher.accept_token(tokenizer.eos_token_id), mode

    auto = _llg_matcher(tokenizer, llg, "EBNF_GRAMMAR", _tool_grammar(tokenizer, "auto"))
    assert _consume(auto, text) and auto.accept_token(tokenizer.eos_token_id)
    required = _llg_matcher(tokenizer, llg, "EBNF_GRAMMAR", _tool_grammar(tokenizer, "required"))
    assert _consume(required, text)
    assert not required.accept_token(tokenizer.eos_token_id), "required needs a call"

    bad_args = _llg_matcher(tokenizer, llg, "EBNF_GRAMMAR", _tool_grammar(tokenizer, "required"))
    prefix = call[: call.index(_control(tokenizer, "[ARGS]")) + 1]
    assert _consume(bad_args, prefix)
    assert not _consume(bad_args, tekken.encode('{"town": "Paris"}', bos=False, eos=False))


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize(
    "parallel_kwargs,allow_parallel",
    [
        ({}, True),
        ({"parallel_tool_calls": None}, True),
        ({"parallel_tool_calls": True}, True),
        ({"parallel_tool_calls": False}, False),
    ],
)
def test_serving_required_and_named_calls_use_native_grammar(
    tokenizer, llg, named, parallel_kwargs, allow_parallel
):
    from tensorrt_llm.serve.extensions.mistral_large_4 import native_tool_guided_decoding
    from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

    _tool_grammar(tokenizer, "required")
    choice = {"type": "function", "function": {"name": "get_weather"}} if named else "required"
    request = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        tools=[_WEATHER_TOOL],
        tool_choice=choice,
        **parallel_kwargs,
    )
    params = native_tool_guided_decoding(request, tokenizer)
    matcher = _llg_matcher(tokenizer, llg, "EBNF_GRAMMAR", params.grammar)
    _consume(matcher, tokenizer.tokenizer.encode("Sunny", bos=False, eos=False))
    assert not matcher.accept_token(tokenizer.eos_token_id)
    matcher = _llg_matcher(tokenizer, llg, "EBNF_GRAMMAR", params.grammar)
    call = (
        [_control(tokenizer, "[TOOL_CALLS]")]
        + tokenizer.tokenizer.encode("get_weather", bos=False, eos=False)
        + [_control(tokenizer, "[ARGS]")]
        + tokenizer.tokenizer.encode('{"city": "Paris", "unit": "c"}', bos=False, eos=False)
    )
    assert _consume(matcher, call)
    assert matcher.accept_token(tokenizer.eos_token_id)
    matcher = _llg_matcher(tokenizer, llg, "EBNF_GRAMMAR", params.grammar)
    assert _consume(matcher, call)
    assert _consume(matcher, call) is (allow_parallel and not named)
    if allow_parallel and not named:
        assert matcher.accept_token(tokenizer.eos_token_id)


def test_llguidance_grammar_rolls_back_draft_tokens(tokenizer, llg):
    """Speculative decoding rolls a rejected draft back off the matcher."""
    pytest.importorskip("jinja2")
    tekken = tokenizer.tokenizer
    matcher = _llg_matcher(tokenizer, llg, "EBNF_GRAMMAR", _tool_grammar(tokenizer, "required"))
    start = _allowed(matcher, tekken.n_words)
    draft = [_control(tokenizer, "[TOOL_CALLS]")] + tekken.encode(
        "get_weather", bos=False, eos=False
    )
    assert _consume(matcher, draft)
    matcher.rollback(len(draft))
    assert _allowed(matcher, tekken.n_words) == start


def test_llguidance_tool_grammar_takes_a_reasoning_section(tokenizer, llg):
    pytest.importorskip("jinja2")
    tekken = tokenizer.tokenizer
    grammar = _tool_grammar(tokenizer, "auto")
    if "[THINK]" not in tekken._special_tokens_reverse_vocab:
        pytest.skip("this Tekken vocabulary has no reasoning tokens")
    matcher = _llg_matcher(tokenizer, llg, "EBNF_GRAMMAR", grammar)
    ids = (
        [_control(tokenizer, "[THINK]")]
        + tekken.encode("The user wants weather.", bos=False, eos=False)
        + [_control(tokenizer, "[/THINK]")]
        + [_control(tokenizer, "[TOOL_CALLS]")]
        + tekken.encode("get_weather", bos=False, eos=False)
        + [_control(tokenizer, "[ARGS]")]
        + tekken.encode('{"city": "Paris", "unit": "f"}', bos=False, eos=False)
    )
    assert _consume(matcher, ids)
    assert matcher.accept_token(tokenizer.eos_token_id)


def _json_matcher(tokenizer, llg, guide_type, guide, *, reasoning_section=True):
    """An llguidance matcher of a JSON response, built as the executor builds it."""
    from tensorrt_llm._torch.pyexecutor.grammar_matcher import LLGuidanceMatcherFactory
    from tensorrt_llm._torch.pyexecutor.py_executor_creator import _llguidance_json_schema_lark
    from tensorrt_llm.bindings.executor import GuidedDecodingParams

    config = get_guided_decoding_config(
        "llguidance", tokenizer, checkpoint_format="mistral_large_4"
    )
    json_schema_lark = (
        _llguidance_json_schema_lark(config, tokenizer, "mistral_large_4")
        if reasoning_section
        else None
    )
    factory = LLGuidanceMatcherFactory(
        config, tokenizer.tokenizer.n_words, tokenizer=llg, json_schema_lark=json_schema_lark
    )
    return factory.create(
        GuidedDecodingParams(getattr(GuidedDecodingParams.GuideType, guide_type), guide)
    )


@pytest.fixture(scope="module")
def reasoning_vocab(tokenizer, llg):
    """Skips vocabularies whose JSON grammar has no reasoning section."""
    pytest.importorskip("jinja2")
    from mistral_common.guidance.grammar_factory import GrammarFactory

    tekken = tokenizer.tokenizer
    if "[THINK]" not in tekken._special_tokens_reverse_vocab:
        pytest.skip("this Tekken vocabulary has no reasoning tokens")
    if not GrammarFactory.is_supported(tokenizer.mistral):
        pytest.skip("mistral-common renders no grammars for this Tekken version")
    if not tekken.version.supports_model_settings:
        pytest.skip("mistral-common reasons before JSON from the model-settings Tekken on")
    return tekken


_JSON_GUIDES = [("JSON_SCHEMA", json.dumps(_SCHEMA)), ("JSON", None)]


@pytest.mark.parametrize("guide_type, guide", _JSON_GUIDES)
def test_llguidance_json_response_may_open_with_a_reasoning_section(
    tokenizer, llg, reasoning_vocab, guide_type, guide
):
    """Reasoning on: [THINK] thoughts [/THINK], then the JSON. Reasoning off: the JSON."""
    tekken = reasoning_vocab
    answer = tekken.encode('{"city": "Paris"}', bos=False, eos=False)
    thought = (
        [_control(tokenizer, "[THINK]")]
        + tekken.encode("The user wants a city as JSON.\nParis fits.", bos=False, eos=False)
        + [_control(tokenizer, "[/THINK]")]
    )
    for ids in (thought + answer, answer):
        matcher = _json_matcher(tokenizer, llg, guide_type, guide)
        assert _consume(matcher, ids)
        assert matcher.accept_token(tokenizer.eos_token_id)


@pytest.mark.parametrize("guide_type, guide", _JSON_GUIDES)
def test_llguidance_json_response_holds_nothing_else(
    tokenizer, llg, reasoning_vocab, guide_type, guide
):
    tekken = reasoning_vocab
    think, end_think = _control(tokenizer, "[THINK]"), _control(tokenizer, "[/THINK]")
    thought = [think] + tekken.encode("Thinking.", bos=False, eos=False) + [end_think]

    prose = _json_matcher(tokenizer, llg, guide_type, guide)
    assert not _consume(prose, tekken.encode('Sure: {"city": "Paris"}', bos=False, eos=False))
    no_answer = _json_matcher(tokenizer, llg, guide_type, guide)
    assert _consume(no_answer, thought)
    assert not no_answer.accept_token(tokenizer.eos_token_id)
    second_thought = _json_matcher(tokenizer, llg, guide_type, guide)
    assert _consume(second_thought, thought)
    assert not second_thought.accept_token(think)
    after_answer = _json_matcher(tokenizer, llg, guide_type, guide)
    assert _consume(after_answer, tekken.encode('{"city": "Paris"}', bos=False, eos=False))
    assert not after_answer.accept_token(think)


def test_llguidance_json_response_never_takes_control_tokens_in_a_string(
    tokenizer, llg, reasoning_vocab
):
    tekken = reasoning_vocab
    matcher = _json_matcher(tokenizer, llg, "JSON_SCHEMA", json.dumps(_SCHEMA))
    thought = (
        [_control(tokenizer, "[THINK]")]
        + tekken.encode("Thinking.", bos=False, eos=False)
        + [_control(tokenizer, "[/THINK]")]
    )
    assert _consume(matcher, thought + tekken.encode('{"city": "', bos=False, eos=False))
    allowed = _allowed(matcher, tekken.n_words)
    assert allowed and min(allowed) >= tekken.num_special_tokens


def test_llguidance_json_response_rolls_back_a_draft_across_the_section_end(
    tokenizer, llg, reasoning_vocab
):
    """A rejected draft that closed the reasoning section and opened the JSON."""
    tekken = reasoning_vocab
    matcher = _json_matcher(tokenizer, llg, "JSON_SCHEMA", json.dumps(_SCHEMA))
    assert _consume(
        matcher,
        [_control(tokenizer, "[THINK]")] + tekken.encode("Thinking.", bos=False, eos=False),
    )
    before = _allowed(matcher, tekken.n_words)
    assert _control(tokenizer, "[/THINK]") in before
    draft = [_control(tokenizer, "[/THINK]")] + tekken.encode('{"city"', bos=False, eos=False)
    assert _consume(matcher, draft)
    matcher.rollback(len(draft))
    assert _allowed(matcher, tekken.n_words) == before


def test_llguidance_json_response_without_the_renderer_starts_at_the_json(
    tokenizer, llg, reasoning_vocab
):
    """Other llguidance models keep the schema alone: [THINK] cannot open the response."""
    matcher = _json_matcher(
        tokenizer, llg, "JSON_SCHEMA", json.dumps(_SCHEMA), reasoning_section=False
    )
    assert _control(tokenizer, "[THINK]") not in _allowed(matcher, reasoning_vocab.n_words)
    matcher = _json_matcher(tokenizer, llg, "JSON_SCHEMA", json.dumps(_SCHEMA))
    assert _control(tokenizer, "[THINK]") in _allowed(matcher, reasoning_vocab.n_words)


def test_llguidance_json_schema_lark_is_mistral_large_4_llguidance_only(tokenizer):
    from tensorrt_llm._torch.pyexecutor.py_executor_creator import _llguidance_json_schema_lark

    llguidance = get_guided_decoding_config(
        "llguidance", tokenizer, checkpoint_format="mistral_large_4"
    )
    xgrammar = get_guided_decoding_config(
        "xgrammar", tokenizer, checkpoint_format="mistral_large_4"
    )
    assert _llguidance_json_schema_lark(llguidance, tokenizer, "mistral_large_4") is not None
    assert _llguidance_json_schema_lark(xgrammar, tokenizer, "mistral_large_4") is None
    assert _llguidance_json_schema_lark(llguidance, tokenizer, "mistral") is None
