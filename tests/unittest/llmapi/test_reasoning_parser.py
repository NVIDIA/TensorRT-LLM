import json
import os

import pytest

from tensorrt_llm.llmapi.reasoning_parser import (ReasoningParserFactory,
                                                  resolve_auto_reasoning_parser)

R1_START, R1_END = "<think>", "</think>"


@pytest.mark.parametrize(("text", "content", "reasoning_context"), [
    ("a b", "", "a b"),
    (f"{R1_END} a b", " a b", ""),
    (f"a {R1_END} b", " b", "a "),
    (f"a b {R1_END}", "", "a b "),
    (f"{R1_START} a {R1_END} b", " b", f"{R1_START} a "),
])
def test_deepseek_r1_reasoning_parser(text: str, content: str,
                                      reasoning_context: str):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "deepseek-r1")
    result = reasoning_parser.parse(text)
    assert result.content == content
    assert result.reasoning_content == reasoning_context


@pytest.mark.parametrize(("delta_texts", "content", "reasoning_context"), [
    (["a", "b"], ["", ""], ["a", "b"]),
    ([R1_END, "a", "b"], ["", "a", "b"], ["", "", ""]),
    (["a", R1_END, "b"], ["", "", "b"], ["a", "", ""]),
    (["a", "b", R1_END], ["", "", ""], ["a", "b", ""]),
    (["a", f"l{R1_END}", "b"], ["", "", "b"], ["a", "l", ""]),
    (["a", f"l{R1_END}r", "b"], ["", "r", "b"], ["a", "l", ""]),
    (["a", f"{R1_END}r", "b"], ["", "r", "b"], ["a", "", ""]),
])
def test_deepseek_r1_reasoning_parser_stream(delta_texts: list, content: list,
                                             reasoning_context: list):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "deepseek-r1")
    for i, delta_text in enumerate(delta_texts):
        result = reasoning_parser.parse_delta(delta_text)
        assert result.content == content[i]
        assert result.reasoning_content == reasoning_context[i]


TOOL_START = "<|tool_calls_section_begin|>"


@pytest.mark.parametrize(
    ("text", "content", "reasoning_context"),
    [
        # Standard <think>...</think> patterns.
        ("a<think>b</think>c", "c", "b"),
        ("<think>a</think>b", "b", "a"),
        ("<think>a", "", "a"),
        ("a", "a", ""),
        ("<think>", "", ""),
        # Interleaved thinking: tool call section implicitly ends reasoning.
        (f"<think>reasoning{TOOL_START}tool_call_data",
         f"{TOOL_START}tool_call_data", "reasoning"),
        # </think> before tool call section: standard end wins.
        (f"<think>reasoning</think>text{TOOL_START}tool_call_data",
         f"text{TOOL_START}tool_call_data", "reasoning"),
        # No <think> tag at all – just content.
        (f"content{TOOL_START}tool_call_data",
         f"content{TOOL_START}tool_call_data", ""),
    ])
def test_kimi_k2_reasoning_parser(text: str, content: str,
                                  reasoning_context: str):
    """Test kimi_k2 reasoning parser (non-streaming).

    Kimi-K2-Thinking generates <think>...</think> tags and may implicitly
    end reasoning via <|tool_calls_section_begin|>.
    """
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser("kimi_k2")
    result = reasoning_parser.parse(text)
    assert result.content == content
    assert result.reasoning_content == reasoning_context


@pytest.mark.parametrize(
    ("delta_texts", "content", "reasoning_context"),
    [
        # Standard streaming cases (same as qwen3).
        (["<think>a", "l</think>r", "b"], ["", "r", "b"], ["a", "l", ""]),
        (["<th", "ink>a</think>b"], ["", "b"], ["", "a"]),
        (["<think>a</th", "ink>b"], ["", "b"], ["a", ""]),
        (["<think>", "a</think>b"], ["", "b"], ["", "a"]),
        (["<think>a</think>", "b"], ["", "b"], ["a", ""]),
        # Interleaved thinking: tool call section implicitly ends reasoning.
        # When the tool token arrives as a full token, the parser buffers it
        # (prefix check) and emits it combined with the next delta.
        (
            ["<think>", "reasoning", TOOL_START, "tool_data"],
            ["", "", "", TOOL_START + "tool_data"],
            ["", "reasoning", "", ""],
        ),
        # Tool section arrives combined with preceding reasoning text.
        (
            ["<think>", "reasoning" + TOOL_START + "tool_data"],
            ["", TOOL_START + "tool_data"],
            ["", "reasoning"],
        ),
        # Partial start-tag at end of delta should be buffered (not leaked).
        (
            ["content<th", "ink>reason</think>after"],
            ["content", "after"],
            ["", "reason"],
        ),
        # Partial tool section tag at end of content after </think>.
        (
            ["<think>reason</think>content<|tool", "_calls_section_begin|>td"],
            ["content", TOOL_START + "td"],
            ["reason", ""],
        ),
    ])
def test_kimi_k2_reasoning_parser_stream(delta_texts: list, content: list,
                                         reasoning_context: list):
    """Test kimi_k2 reasoning parser streaming."""
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser("kimi_k2")
    for i, delta_text in enumerate(delta_texts):
        result = reasoning_parser.parse_delta(delta_text)
        assert result.content == content[i], \
            f"Step {i}: delta={delta_text!r}, expected content={content[i]!r}, got {result.content!r}"
        assert result.reasoning_content == reasoning_context[i], \
            f"Step {i}: delta={delta_text!r}, expected reasoning={reasoning_context[i]!r}, got {result.reasoning_content!r}"


@pytest.mark.parametrize(
    ("parser_name", "delta_texts", "content", "reasoning_context"),
    [
        # Interleaved thinking with reasoning_at_start=True (deepseek-r1, minimax_m2):
        # Model output starts directly with reasoning (no <think> tag).
        # Simulates: reason1</think>text1<think>reason2</think>text2
        (
            "minimax_m2",
            ["reason1", R1_END, "text1", R1_START, "reason2", R1_END, "text2"],
            ["", "", "text1", "", "", "", "text2"],
            ["reason1", "", "", "", "reason2", "", ""],
        ),
        (
            "deepseek-r1",
            ["reason1", R1_END, "text1", R1_START, "reason2", R1_END, "text2"],
            ["", "", "text1", "", "", "", "text2"],
            ["reason1", "", "", "", "reason2", "", ""],
        ),
        # Interleaved thinking with reasoning_at_start=False (qwen3, kimi_k2):
        # Model output contains <think>...</think> tags.
        # Simulates: <think>reason1</think>content1<think>reason2</think>content2
        (
            "qwen3",
            [
                R1_START, "reason1", R1_END, "content1", R1_START, "reason2",
                R1_END, "content2"
            ],
            ["", "", "", "content1", "", "", "", "content2"],
            ["", "reason1", "", "", "", "reason2", "", ""],
        ),
        (
            "kimi_k2",
            [
                R1_START, "reason1", R1_END, "content1", R1_START, "reason2",
                R1_END, "content2"
            ],
            ["", "", "", "content1", "", "", "", "content2"],
            ["", "reason1", "", "", "", "reason2", "", ""],
        ),
        # Kimi-K2 interleaved thinking: reasoning interrupted by tool calls.
        # Simulates: <think>reasoning<|tool_calls_section_begin|>tool_data
        # Note: when TOOL_START arrives as a full token, the parser buffers it
        # (prefix check) and emits it combined with the next delta.
        (
            "kimi_k2",
            [R1_START, "reasoning", TOOL_START, "tool_data"],
            ["", "", "", TOOL_START + "tool_data"],
            ["", "reasoning", "", ""],
        ),
    ],
)
def test_interleaved_thinking_stream(parser_name: str, delta_texts: list,
                                     content: list, reasoning_context: list):
    """Test that streaming parsers correctly handle interleaved thinking.

    Interleaved thinking allows models to reason between tool calls,
    producing multiple <think>...</think> blocks within a single generation.
    The streaming parser must correctly transition between reasoning and
    content modes across multiple think blocks.

    For kimi_k2, reasoning may also be implicitly ended by the tool call
    section token <|tool_calls_section_begin|> without an explicit </think>.
    """
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        parser_name)
    for i, delta_text in enumerate(delta_texts):
        result = reasoning_parser.parse_delta(delta_text)
        assert result.content == content[i], \
            f"Step {i}: delta={delta_text!r}, expected content={content[i]!r}, got {result.content!r}"
        assert result.reasoning_content == reasoning_context[i], \
            f"Step {i}: delta={delta_text!r}, expected reasoning={reasoning_context[i]!r}, got {result.reasoning_content!r}"


@pytest.mark.parametrize(("text", "content", "reasoning_context"), [
    ("a<think>b</think>c", "c", "b"),
    ("<think>a</think>b", "b", "a"),
    ("<think>a", "", "a"),
    ("a", "a", ""),
    ("<think>", "", ""),
])
def test_qwen3_reasoning_parser(text: str, content: str,
                                reasoning_context: str):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser("qwen3")
    result = reasoning_parser.parse(text)
    assert result.content == content
    assert result.reasoning_content == reasoning_context


@pytest.mark.parametrize(("delta_texts", "content", "reasoning_context"), [
    (["<think>a", "l</think>r", "b"], ["", "r", "b"], ["a", "l", ""]),
    (["<th", "ink>a</think>b"], ["", "b"], ["", "a"]),
    (["<think>a</th", "ink>b"], ["", "b"], ["a", ""]),
    (["<think>", "a</think>b"], ["", "b"], ["", "a"]),
    (["<think>a</think>", "b"], ["", "b"], ["a", ""]),
    (["<think>a</th", "ank></th", "ink>b"], ["", "", "b"
                                             ], ["a", "</thank>", ""]),
])
def test_qwen3_reasoning_parser_stream(delta_texts: list, content: list,
                                       reasoning_context: list):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser("qwen3")
    for i, delta_text in enumerate(delta_texts):
        result = reasoning_parser.parse_delta(delta_text)
        assert result.content == content[i]
        assert result.reasoning_content == reasoning_context[i]


@pytest.mark.parametrize(
    ("text", "content", "reasoning_context", "chat_template_kwargs"),
    [
        ("a b", "", "a b", None),
        (f"{R1_END} a b", " a b", "", None),
        (f"a {R1_END} b", " b", "a ", None),
        (f"a b {R1_END}", "", "a b ", None),
        (f"{R1_START} a {R1_END} b", " b", f"{R1_START} a ", None),
        # All without reasoning_context.
        ("a b", "a b", "", {
            "enable_thinking": False
        }),
        (f"{R1_END} a b", f"{R1_END} a b", "", {
            "enable_thinking": False
        }),
        (f"a {R1_END} b", f"a {R1_END} b", "", {
            "enable_thinking": False
        }),
        (f"a b {R1_END}", f"a b {R1_END}", "", {
            "enable_thinking": False
        }),
        # force_nonempty_content swaps reasoning into content when content is
        # empty (reasoning_at_start stays True, so parsing is unchanged).
        ("a b", "a b", "", {
            "force_nonempty_content": True
        }),
        (f"a {R1_END} b", " b", "a ", {
            "force_nonempty_content": True
        }),
        (f"a b {R1_END}", "a b ", "", {
            "force_nonempty_content": True
        }),
        # NVBug 6060281: whitespace-only content after </redacted_thinking> must
        # still trigger the reasoning-to-content swap when force_nonempty_content.
        (f"a {R1_END}\n", "a ", "", {
            "force_nonempty_content": True
        }),
        (f"a {R1_END} \t ", "a ", "", {
            "force_nonempty_content": True
        }),
    ])
def test_nano_v3_reasoning_parser(text: str, content: str,
                                  reasoning_context: str,
                                  chat_template_kwargs: dict):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "nano-v3", chat_template_kwargs)
    result = reasoning_parser.parse(text)
    print(f"text: {text}, result: {result}")
    assert result.content == content
    assert result.reasoning_content == reasoning_context


@pytest.mark.parametrize(
    ("delta_texts", "content", "reasoning_context", "chat_template_kwargs"),
    [
        (["a", "b"], ["", ""], ["a", "b"], None),
        ([R1_END, "a", "b"], ["", "a", "b"], ["", "", ""], None),
        (["a", R1_END, "b"], ["", "", "b"], ["a", "", ""], None),
        (["a", "b", R1_END], ["", "", ""], ["a", "b", ""], None),
        (["a", f"l{R1_END}", "b"], ["", "", "b"], ["a", "l", ""], None),
        (["a", f"l{R1_END}r", "b"], ["", "r", "b"], ["a", "l", ""], None),
        (["a", f"{R1_END}r", "b"], ["", "r", "b"], ["a", "", ""], None),
        # All without reasoning_context.
        (["a", "b"], ["a", "b"], ["", ""], {
            "enable_thinking": False
        }),
        ([R1_END, "a", "b"], ["", f"{R1_END}a", "b"], ["", "", ""], {
            "enable_thinking": False
        }),
        (["a", R1_END, "b"], ["a", "", f"{R1_END}b"], ["", "", ""], {
            "enable_thinking": False
        }),
        (["a", "b", R1_END], ["a", "b", ""], ["", "", ""], {
            "enable_thinking": False
        }),
        (["a", f"l{R1_END}", "b"], ["a", f"l{R1_END}", "b"], ["", "", ""], {
            "enable_thinking": False
        }),
        (["a", f"l{R1_END}r", "b"], ["a", f"l{R1_END}r", "b"], ["", "", ""], {
            "enable_thinking": False
        }),
        (["a", f"{R1_END}r", "b"], ["a", f"{R1_END}r", "b"], ["", "", ""], {
            "enable_thinking": False
        }),
    ])
def test_nano_v3_reasoning_parser_stream(delta_texts: list, content: list,
                                         reasoning_context: list,
                                         chat_template_kwargs: dict):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "nano-v3", chat_template_kwargs)
    for i, delta_text in enumerate(delta_texts):
        result = reasoning_parser.parse_delta(delta_text)
        print(f"delta_text: {delta_text}, result: {result}")
        assert result.content == content[i]
        assert result.reasoning_content == reasoning_context[i]


@pytest.mark.parametrize(("delta_texts", "finish_content", "finish_reasoning",
                          "chat_template_kwargs"), [
                              (["a", "b"], "", "", None),
                              ([R1_END, "a", "b"], "", "", None),
                              (["a", R1_END, "b"], "", "", None),
                              (["a", "b"], "", "", {
                                  "enable_thinking": False
                              }),
                              ([f"{R1_START}a", "b"], "", "", {
                                  "enable_thinking": False
                              }),
                              (["a", "b"], "", "", {
                                  "force_nonempty_content": False
                              }),
                              (["a", "b"], "ab", "", {
                                  "force_nonempty_content": True
                              }),
                              ([R1_END, "a", "b"], "", "", {
                                  "force_nonempty_content": True
                              }),
                          ])
def test_nano_v3_reasoning_parser_finish(delta_texts: list, finish_content: str,
                                         finish_reasoning: str,
                                         chat_template_kwargs: dict):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "nano-v3", chat_template_kwargs)
    for delta_text in delta_texts:
        reasoning_parser.parse_delta(delta_text)
    result = reasoning_parser.finish()
    assert result.content == finish_content
    assert result.reasoning_content == finish_reasoning


# ---------------------------------------------------------------------------
# Auto-detection tests for resolve_auto_reasoning_parser
# ---------------------------------------------------------------------------


def _write_config(model_dir: str, model_type: str):
    """Write a minimal config.json with the given model_type."""
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump({"model_type": model_type}, f)


def _write_tokenizer_config(model_dir: str, chat_template: str):
    """Write a minimal tokenizer_config.json with the given chat_template."""
    with open(os.path.join(model_dir, "tokenizer_config.json"), "w") as f:
        json.dump({"chat_template": chat_template}, f)


# Hybrid Qwen3: chat template contains "enable_thinking" → use "qwen3" parser
_HYBRID_TEMPLATE = (
    "{%- if enable_thinking is not defined %}{% set enable_thinking = true %}"
    "{% endif %}{%- if add_generation_prompt %}{%- if enable_thinking %}"
    "{{- '<|im_start|>assistant\\n<think>\\n' }}{%- else %}"
    "{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}"
    "{%- endif %}{%- endif %}")

# Forced-thinking Qwen3: no "enable_thinking" but has "<think>" → "deepseek-r1"
_FORCED_THINKING_TEMPLATE = ("{%- if add_generation_prompt %}"
                             "{{- '<|im_start|>assistant\\n<think>\\n' }}"
                             "{%- endif %}")

# Forced-non-thinking Qwen3: no "enable_thinking" and no "<think>" → None
_FORCED_NON_THINKING_TEMPLATE = ("{%- if add_generation_prompt %}"
                                 "{{- '<|im_start|>assistant\\n' }}"
                                 "{%- endif %}")


def test_auto_detect_qwen3_hybrid(tmp_path):
    """Hybrid Qwen3 model with enable_thinking toggle → 'qwen3' parser."""
    model_dir = str(tmp_path / "Qwen3-235B-A22B")
    os.makedirs(model_dir)
    _write_config(model_dir, "qwen3_moe")
    _write_tokenizer_config(model_dir, _HYBRID_TEMPLATE)

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "qwen3"


def test_auto_detect_qwen3_forced_thinking(tmp_path):
    """Forced-thinking Qwen3 model → 'deepseek-r1' parser."""
    model_dir = str(tmp_path / "Qwen3-235B-A22B-Thinking-2507")
    os.makedirs(model_dir)
    _write_config(model_dir, "qwen3_moe")
    _write_tokenizer_config(model_dir, _FORCED_THINKING_TEMPLATE)

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "deepseek-r1"


def test_auto_detect_qwen3_forced_non_thinking(tmp_path):
    """Forced-non-thinking Qwen3 model → None (no parser needed)."""
    model_dir = str(tmp_path / "Qwen3-235B-A22B-Instruct-2507")
    os.makedirs(model_dir)
    _write_config(model_dir, "qwen3_moe")
    _write_tokenizer_config(model_dir, _FORCED_NON_THINKING_TEMPLATE)

    result = resolve_auto_reasoning_parser(model_dir)
    assert result is None


def test_auto_detect_qwen3_no_tokenizer_config(tmp_path):
    """Qwen3 model without tokenizer_config.json → falls back to 'qwen3'."""
    model_dir = str(tmp_path / "Qwen3-SomeModel")
    os.makedirs(model_dir)
    _write_config(model_dir, "qwen3")

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "qwen3"


def test_auto_detect_deepseek_r1(tmp_path):
    """DeepSeek R1 model → 'deepseek-r1' parser."""
    model_dir = str(tmp_path / "DeepSeek-R1")
    os.makedirs(model_dir)
    _write_config(model_dir, "deepseek_v3")

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "deepseek-r1"


def test_auto_detect_deepseek_non_r1(tmp_path):
    """DeepSeek non-R1 model → None."""
    model_dir = str(tmp_path / "DeepSeek-V3")
    os.makedirs(model_dir)
    _write_config(model_dir, "deepseek_v3")

    result = resolve_auto_reasoning_parser(model_dir)
    assert result is None


def test_auto_detect_unknown_model(tmp_path):
    """Unknown model type → None."""
    model_dir = str(tmp_path / "SomeUnknownModel")
    os.makedirs(model_dir)
    _write_config(model_dir, "unknown_type")

    result = resolve_auto_reasoning_parser(model_dir)
    assert result is None


def test_auto_detect_no_config(tmp_path):
    """No config.json → None."""
    model_dir = str(tmp_path / "EmptyDir")
    os.makedirs(model_dir)

    result = resolve_auto_reasoning_parser(model_dir)
    assert result is None


def test_auto_detect_gemma4(tmp_path):
    """Gemma 4 model → 'gemma4' parser."""
    model_dir = str(tmp_path / "gemma-4-26B-A4B-it")
    os.makedirs(model_dir)
    _write_config(model_dir, "gemma4")

    result = resolve_auto_reasoning_parser(model_dir)
    assert result == "gemma4"


# ---------------------------------------------------------------------------
# Gemma 4 reasoning parser tests
# ---------------------------------------------------------------------------

G4_OPEN, G4_CLOSE = "<|channel>", "<channel|>"


@pytest.mark.parametrize(
    ("text", "content", "reasoning_content"),
    [
        # No reasoning block: everything is content.
        ("hello world", "hello world", ""),
        # Channel block wrapping reasoning, followed by content.
        (f"{G4_OPEN}thought\nreasoning{G4_CLOSE}answer", "answer",
         "thought\nreasoning"),
        # Content before and after the reasoning block.
        (f"pre{G4_OPEN}r{G4_CLOSE}post", "prepost", "r"),
        # Unterminated channel: remainder treated as reasoning.
        (f"{G4_OPEN}abc", "", "abc"),
        # Multiple interleaved channel blocks.
        (f"a{G4_OPEN}r1{G4_CLOSE}b{G4_OPEN}r2{G4_CLOSE}c", "abc", "r1r2"),
        # Empty reasoning block (e.g. prefilled when enable_thinking=False).
        (f"{G4_OPEN}thought\n{G4_CLOSE}answer", "answer", "thought\n"),
    ],
)
def test_gemma4_reasoning_parser(text: str, content: str,
                                 reasoning_content: str):
    parser = ReasoningParserFactory.create_reasoning_parser("gemma4")
    result = parser.parse(text)
    assert result.content == content
    assert result.reasoning_content == reasoning_content


@pytest.mark.parametrize(
    ("delta_texts", "content", "reasoning_content"),
    [
        # No reasoning: plain content streams through.
        (["a", "b"], ["a", "b"], ["", ""]),
        # Open and close in a single delta.
        ([f"{G4_OPEN}r{G4_CLOSE}c"], ["c"], ["r"]),
        # Delimiters split across deltas.
        ([G4_OPEN, "r", G4_CLOSE, "c"], ["", "", "", "c"], ["", "r", "", ""]),
        # Partial open tag held back until complete.
        (["pre<|cha", "nnel>r<chan", "nel|>post"], ["pre", "", "post"
                                                    ], ["", "r", ""]),
        # Two reasoning blocks interleaved with content.
        (
            [f"{G4_OPEN}r1{G4_CLOSE}c1", f"{G4_OPEN}r2{G4_CLOSE}c2"],
            ["c1", "c2"],
            ["r1", "r2"],
        ),
        # Partial close tag at end of delta buffered.
        ([f"{G4_OPEN}reason<chan", "nel|>tail"], ["", "tail"], ["reason", ""]),
    ],
)
def test_gemma4_reasoning_parser_stream(delta_texts: list, content: list,
                                        reasoning_content: list):
    parser = ReasoningParserFactory.create_reasoning_parser("gemma4")
    for i, delta in enumerate(delta_texts):
        result = parser.parse_delta(delta)
        assert result.content == content[i], (
            f"Step {i}: delta={delta!r} expected content={content[i]!r} "
            f"got {result.content!r}")
        assert result.reasoning_content == reasoning_content[i], (
            f"Step {i}: delta={delta!r} expected reasoning="
            f"{reasoning_content[i]!r} got {result.reasoning_content!r}")


def test_gemma4_reasoning_parser_finish_flushes_buffer():
    """finish() should flush any buffered trailing text."""
    parser = ReasoningParserFactory.create_reasoning_parser("gemma4")
    # Send a partial open tag; parser holds it back.
    parser.parse_delta("some text<|cha")
    # Stream ended mid-tag: the held-back suffix flushes as content.
    result = parser.finish()
    assert result.content == "<|cha"
    assert result.reasoning_content == ""


def test_gemma4_reasoning_parser_finish_unterminated_reasoning():
    """Verify finish() flushes a held-back partial close tag as reasoning.

    When the stream ends mid-channel with a buffered partial close tag, the
    remainder should surface as reasoning content.
    """
    parser = ReasoningParserFactory.create_reasoning_parser("gemma4")
    # Enter reasoning; stream ends with a partial close tag (held back).
    stream = parser.parse_delta(f"{G4_OPEN}reasoning_start<chan")
    assert stream.reasoning_content == "reasoning_start"
    assert stream.content == ""
    # finish() should release the buffered "<chan" as reasoning since we are
    # still inside the channel block.
    result = parser.finish()
    assert result.content == ""
    assert result.reasoning_content == "<chan"
