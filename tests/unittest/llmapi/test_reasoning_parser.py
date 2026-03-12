import pytest

from tensorrt_llm.llmapi.reasoning_parser import (ContentBlock,
                                                  ReasoningParserFactory)

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


# ── Interleaved thinking: parse_to_blocks ──────────────────────────


@pytest.mark.parametrize(
    ("text", "expected_blocks"),
    [
        # deepseek-r1: reasoning starts immediately
        ("think1</think>text1",
         [ContentBlock("thinking", "think1"),
          ContentBlock("text", "text1")]),
        # Only reasoning, no closing tag
        ("all reasoning", [ContentBlock("thinking", "all reasoning")]),
        # Reasoning then text then reasoning (interleaved)
        ("think1</think>text1<think>think2</think>text2", [
            ContentBlock("thinking", "think1"),
            ContentBlock("text", "text1"),
            ContentBlock("thinking", "think2"),
            ContentBlock("text", "text2"),
        ]),
        # Reasoning with empty text between thinking blocks
        ("think1</think><think>think2</think>text2", [
            ContentBlock("thinking", "think1"),
            ContentBlock("thinking", "think2"),
            ContentBlock("text", "text2"),
        ]),
        # Only reasoning with closing tag
        ("all reasoning</think>", [ContentBlock("thinking", "all reasoning")]),
        # Multiple interleaved blocks
        ("t1</think>c1<think>t2</think>c2<think>t3</think>c3", [
            ContentBlock("thinking", "t1"),
            ContentBlock("text", "c1"),
            ContentBlock("thinking", "t2"),
            ContentBlock("text", "c2"),
            ContentBlock("thinking", "t3"),
            ContentBlock("text", "c3"),
        ]),
    ])
def test_deepseek_r1_parse_to_blocks(text: str,
                                     expected_blocks: list[ContentBlock]):
    parser = ReasoningParserFactory.create_reasoning_parser("deepseek-r1")
    blocks = parser.parse_to_blocks(text)
    assert blocks == expected_blocks


@pytest.mark.parametrize(
    ("text", "expected_blocks"),
    [
        # qwen3: reasoning doesn't start at beginning
        ("<think>think1</think>text1",
         [ContentBlock("thinking", "think1"),
          ContentBlock("text", "text1")]),
        # No thinking at all
        ("just text", [ContentBlock("text", "just text")]),
        # Interleaved thinking
        ("<think>think1</think>text1<think>think2</think>text2", [
            ContentBlock("thinking", "think1"),
            ContentBlock("text", "text1"),
            ContentBlock("thinking", "think2"),
            ContentBlock("text", "text2"),
        ]),
        # Only thinking
        ("<think>thinking only", [ContentBlock("thinking", "thinking only")]),
        # Empty thinking
        ("<think></think>text only", [ContentBlock("text", "text only")]),
    ])
def test_qwen3_parse_to_blocks(text: str, expected_blocks: list[ContentBlock]):
    parser = ReasoningParserFactory.create_reasoning_parser("qwen3")
    blocks = parser.parse_to_blocks(text)
    assert blocks == expected_blocks


# ── Interleaved thinking: streaming block type tracking ──────────


@pytest.mark.parametrize(
    ("delta_texts", "expected_block_types"),
    [
        # deepseek-r1 streaming: starts in reasoning, then transitions
        (["think", f"ing{R1_END}text"], ["thinking", "text"]),
        # All reasoning
        (["think", "more"], ["thinking", "thinking"]),
        # Transition to text then back to thinking
        ([f"t1{R1_END}c1", f"{R1_START}t2{R1_END}c2"], ["text", "text"]),
    ])
def test_deepseek_r1_stream_block_type(delta_texts: list,
                                       expected_block_types: list):
    parser = ReasoningParserFactory.create_reasoning_parser("deepseek-r1")
    for i, delta_text in enumerate(delta_texts):
        result = parser.parse_delta(delta_text)
        assert result.current_block_type == expected_block_types[i]


@pytest.mark.parametrize(
    ("delta_texts", "expected_block_types"),
    [
        # qwen3 streaming: starts with think tag
        (["<think>a", f"l{R1_END}r"], ["thinking", "text"]),
        # Text only
        (["hello", "world"], ["text", "text"]),
    ])
def test_qwen3_stream_block_type(delta_texts: list, expected_block_types: list):
    parser = ReasoningParserFactory.create_reasoning_parser("qwen3")
    for i, delta_text in enumerate(delta_texts):
        result = parser.parse_delta(delta_text)
        assert result.current_block_type == expected_block_types[i]
