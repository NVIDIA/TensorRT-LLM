import pytest

from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory

R1_START, R1_END = "<think>", "</think>"


@pytest.mark.parametrize(
    ("text", "in_reasoning", "content", "reasoning_context"), [
        ("a b", True, None, "a b"),
        (f"{R1_END} a b", False, " a b", None),
        (f"a {R1_END} b", False, " b", "a "),
        (f"a b {R1_END}", True, None, "a b "),
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
        (["a", "b"], [True, True], [None, None], ["a", "b"]),
        ([R1_END, "a", "b"], [True, False, False], [None, "a", "b"
                                                    ], ["", None, None]),
        (["a", R1_END, "b"], [True, True, False], [None, None, "b"
                                                   ], ["a", "", None]),
        (["a", "b", R1_END], [True, True, True], [None, None, None
                                                  ], ["a", "b", ""]),
        (["a", f"l{R1_END}", "b"], [True, True, False], [None, None, "b"
                                                         ], ["a", "l", None]),
        (["a", f"l{R1_END}r", "b"], [True, False, False], [None, "r", "b"
                                                           ], ["a", "l", None]),
        (["a", f"{R1_END}r", "b"], [True, False, False], [None, "r", "b"
                                                          ], ["a", None, None]),
        ([R1_START, "a", f"l{R1_END}r", "b"], [True, True, False, False],
         [None, None, "r", "b"], [R1_START, "a", "l", None]),
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
