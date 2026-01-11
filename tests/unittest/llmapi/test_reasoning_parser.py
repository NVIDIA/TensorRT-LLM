import pytest

from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory

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
