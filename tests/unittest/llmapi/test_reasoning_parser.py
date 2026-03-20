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
