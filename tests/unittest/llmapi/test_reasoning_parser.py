import pytest

from tensorrt_llm.executor.postproc_worker import PostprocArgs, PostprocParams
from tensorrt_llm.executor.result import CompletionOutput
from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory

R1_START, R1_END = "<think>", "</think>"

# ============================================================================
# Tests for PostprocArgs reasoning_parser field
# ============================================================================


def test_postproc_args_has_reasoning_parser_field():
    """Test that PostprocArgs has the reasoning_parser field."""
    args = PostprocArgs()
    assert hasattr(args, 'reasoning_parser')
    assert args.reasoning_parser is None


def test_postproc_args_with_reasoning_parser():
    """Test PostprocArgs can be created with reasoning_parser."""
    args = PostprocArgs(reasoning_parser="qwen3")
    assert args.reasoning_parser == "qwen3"


def test_postproc_params_with_reasoning_parser():
    """Test PostprocParams can carry reasoning_parser via PostprocArgs."""
    args = PostprocArgs(reasoning_parser="deepseek-r1")
    params = PostprocParams(postproc_args=args)
    assert params.postproc_args.reasoning_parser == "deepseek-r1"


# ============================================================================
# Tests for CompletionOutput reasoning_content field
# ============================================================================


def test_completion_output_has_reasoning_content_field():
    """Test that CompletionOutput has the reasoning_content field."""
    output = CompletionOutput(index=0)
    assert hasattr(output, 'reasoning_content')
    assert output.reasoning_content is None


def test_completion_output_with_reasoning_content():
    """Test CompletionOutput can store reasoning_content."""
    output = CompletionOutput(index=0)
    output.text = "The answer is 42."
    output.reasoning_content = "Let me think step by step..."
    assert output.text == "The answer is 42."
    assert output.reasoning_content == "Let me think step by step..."


# ============================================================================
# Tests for reasoning parser integration with result layer
# ============================================================================


class MockGenerationResultBase:
    """Mock class to test _apply_reasoning_parser_if_needed logic."""

    def __init__(self, postproc_params=None):
        self.postproc_params = postproc_params
        self._outputs = [CompletionOutput(index=0)]

    def _apply_reasoning_parser_if_needed(self) -> None:
        """Apply reasoning parser to completed outputs (non-streaming, no postproc workers)."""
        if self.postproc_params is None:
            return
        postproc_args = self.postproc_params.postproc_args
        if postproc_args is None:
            return
        reasoning_parser_name = getattr(postproc_args, 'reasoning_parser', None)
        if not reasoning_parser_name:
            return
        # Avoid re-parsing if already done by postproc workers
        if self.postproc_params.post_processor is not None:
            return
        parser = ReasoningParserFactory.create_reasoning_parser(
            reasoning_parser_name)
        for output in self._outputs:
            parsed = parser.parse(output.text)
            output.text = parsed.content
            output.reasoning_content = parsed.reasoning_content


def test_apply_reasoning_parser_no_postproc_params():
    """Test that no parsing happens when postproc_params is None."""
    result = MockGenerationResultBase(postproc_params=None)
    result._outputs[0].text = "<think>reasoning</think>answer"
    result._apply_reasoning_parser_if_needed()
    # Should remain unchanged
    assert result._outputs[0].text == "<think>reasoning</think>answer"
    assert result._outputs[0].reasoning_content is None


def test_apply_reasoning_parser_no_postproc_args():
    """Test that no parsing happens when postproc_args is None."""
    params = PostprocParams(postproc_args=None)
    result = MockGenerationResultBase(postproc_params=params)
    result._outputs[0].text = "<think>reasoning</think>answer"
    result._apply_reasoning_parser_if_needed()
    # Should remain unchanged
    assert result._outputs[0].text == "<think>reasoning</think>answer"
    assert result._outputs[0].reasoning_content is None


def test_apply_reasoning_parser_no_parser_name():
    """Test that no parsing happens when reasoning_parser is None."""
    args = PostprocArgs(reasoning_parser=None)
    params = PostprocParams(postproc_args=args)
    result = MockGenerationResultBase(postproc_params=params)
    result._outputs[0].text = "<think>reasoning</think>answer"
    result._apply_reasoning_parser_if_needed()
    # Should remain unchanged
    assert result._outputs[0].text == "<think>reasoning</think>answer"
    assert result._outputs[0].reasoning_content is None


def test_apply_reasoning_parser_skips_when_post_processor_set():
    """Test that no parsing happens when post_processor is already set (postproc workers)."""
    args = PostprocArgs(reasoning_parser="qwen3")
    params = PostprocParams(postproc_args=args,
                            post_processor=lambda x, y: x)  # Dummy processor
    result = MockGenerationResultBase(postproc_params=params)
    result._outputs[0].text = "<think>reasoning</think>answer"
    result._apply_reasoning_parser_if_needed()
    # Should remain unchanged because post_processor is set
    assert result._outputs[0].text == "<think>reasoning</think>answer"
    assert result._outputs[0].reasoning_content is None


@pytest.mark.parametrize(
    ("parser_name", "text", "expected_content", "expected_reasoning"), [
        ("qwen3", "<think>step by step</think>42", "42", "step by step"),
        ("qwen3", "<think>let me think</think>The answer is 60",
         "The answer is 60", "let me think"),
        ("qwen3", "no thinking tags here", "no thinking tags here", ""),
        ("deepseek-r1", "thinking process</think>final answer", "final answer",
         "thinking process"),
        ("deepseek-r1", "all reasoning no end tag", "",
         "all reasoning no end tag"),
    ])
def test_apply_reasoning_parser_parses_correctly(parser_name, text,
                                                 expected_content,
                                                 expected_reasoning):
    """Test that reasoning parser correctly parses text with various patterns."""
    args = PostprocArgs(reasoning_parser=parser_name)
    params = PostprocParams(postproc_args=args)
    result = MockGenerationResultBase(postproc_params=params)
    result._outputs[0].text = text
    result._apply_reasoning_parser_if_needed()
    assert result._outputs[0].text == expected_content
    assert result._outputs[0].reasoning_content == expected_reasoning


def test_apply_reasoning_parser_multiple_outputs():
    """Test that reasoning parser is applied to all outputs."""
    args = PostprocArgs(reasoning_parser="qwen3")
    params = PostprocParams(postproc_args=args)
    result = MockGenerationResultBase(postproc_params=params)
    # Add multiple outputs
    result._outputs = [
        CompletionOutput(index=0),
        CompletionOutput(index=1),
    ]
    result._outputs[0].text = "<think>reason1</think>answer1"
    result._outputs[1].text = "<think>reason2</think>answer2"

    result._apply_reasoning_parser_if_needed()

    assert result._outputs[0].text == "answer1"
    assert result._outputs[0].reasoning_content == "reason1"
    assert result._outputs[1].text == "answer2"
    assert result._outputs[1].reasoning_content == "reason2"


# ============================================================================
# Original reasoning parser unit tests
# ============================================================================


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
