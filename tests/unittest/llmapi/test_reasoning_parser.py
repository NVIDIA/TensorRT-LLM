import pytest

from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory

R1_START, R1_END = "<think>", "</think>"


@pytest.mark.parametrize(
    ("text", "in_reasoning", "content", "reasoning_context"), [
        ("a b", True, "", "a b"),
        (f"{R1_END} a b", False, " a b", ""),
        (f"a {R1_END} b", False, " b", "a "),
        (f"a b {R1_END}", True, "", "a b "),
        (f"{R1_START} a {R1_END} b", False, " b", f"{R1_START} a "),
    ])
def test_deepseek_r1_reasoning_parser(text: str, in_reasoning: bool,
                                      content: str, reasoning_context: str):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "deepseek-r1")
    result = reasoning_parser.parse(text)
    assert result.in_reasoning == in_reasoning
    assert result.content == content
    assert result.reasoning_content == reasoning_context


@pytest.mark.parametrize(
    ("delta_texts", "in_reasoning", "content", "reasoning_context"), [
        (["a", "b"], [True, True], ["", ""], ["a", "b"]),
        ([R1_END, "a", "b"], [True, False, False], ["", "a", "b"], ["", "", ""
                                                                    ]),
        (["a", R1_END, "b"], [True, True, False], ["", "", "b"], ["a", "", ""]),
        (["a", "b", R1_END], [True, True, True], ["", "", ""], ["a", "b", ""]),
        (["a", f"l{R1_END}", "b"], [True, False, False], ["", "", "b"
                                                          ], ["a", "l", ""]),
        (["a", f"l{R1_END}r", "b"], [True, False, False], ["", "r", "b"
                                                           ], ["a", "l", ""]),
        (["a", f"{R1_END}r", "b"], [True, False, False], ["", "r", "b"
                                                          ], ["a", "", ""]),
    ])
def test_deepseek_r1_reasoning_parser_stream(delta_texts: list,
                                             in_reasoning: list, content: list,
                                             reasoning_context: list):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
        "deepseek-r1")
    for i, delta_text in enumerate(delta_texts):
        result = reasoning_parser.parse_delta(delta_text)
        assert result.in_reasoning == in_reasoning[i]
        assert result.content == content[i]
        assert result.reasoning_content == reasoning_context[i]


@pytest.mark.parametrize(
    ("text", "in_reasoning", "content", "reasoning_context"), [
        ("a<think>b</think>c", False, "c", "b"),
        ("<think>a</think>b", False, "b", "a"),
        ("<think>a", True, "", "a"),
        ("a", False, "a", ""),
        ("<think>", True, "", ""),
    ])
def test_qwen3_reasoning_parser(text: str, in_reasoning: bool, content: str,
                                reasoning_context: str):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser("qwen3")
    result = reasoning_parser.parse(text)
    assert result.in_reasoning == in_reasoning
    assert result.content == content
    assert result.reasoning_content == reasoning_context


@pytest.mark.parametrize(
    ("delta_texts", "in_reasoning", "content", "reasoning_context"), [
        (["<think>a", "l</think>r", "b"], [True, False, False
                                           ], ["", "r", "b"], ["a", "l", ""]),
        (["<th", "ink>a</think>b"], [False, False], ["", "b"], ["", "a"]),
        (["<think>a</th", "ink>b"], [True, False], ["", "b"], ["a", ""]),
        (["<think>", "a</think>b"], [False, False], ["", "b"], ["", "a"]),
        (["<think>a</think>", "b"], [False, False], ["", "b"], ["a", ""]),
    ])
def test_qwen3_reasoning_parser_stream(delta_texts: list, in_reasoning: list,
                                       content: list, reasoning_context: list):
    reasoning_parser = ReasoningParserFactory.create_reasoning_parser("qwen3")
    for i, delta_text in enumerate(delta_texts):
        result = reasoning_parser.parse_delta(delta_text)
        assert result.in_reasoning == in_reasoning[i]
        assert result.content == content[i]
        assert result.reasoning_content == reasoning_context[i]
