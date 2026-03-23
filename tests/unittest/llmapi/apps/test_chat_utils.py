from unittest.mock import MagicMock, patch

import pytest
from transformers import AutoConfig

from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY
from tensorrt_llm.serve.chat_utils import (
    load_chat_template,
    parse_chat_message_content,
    parse_chat_message_content_part,
    parse_chat_messages_coroutines,
)


@pytest.fixture
def mock_mm_data_tracker():
    """Create a mock MultimodalDataTracker for testing."""
    return MagicMock()


class TestParseAssistantMessages:
    """Test suite for assistant role messages."""

    @pytest.mark.parametrize("content", [None, "Hello, how can I help you?"])
    def test_assistant_message_no_tool_calls(
        self,
        mock_mm_data_tracker,
        content,
    ):
        """Test parsing an assistant message with simple string content."""
        message = {"role": "assistant", "content": content}

        result = parse_chat_message_content(message, mock_mm_data_tracker)

        assert result["role"] == "assistant"
        assert result["content"] == (content or "")
        assert result["media"] == []
        assert "tool_calls" not in result

    @pytest.mark.parametrize(
        "arguments",
        [
            # JSON string.
            '{"location": "San Francisco", "unit": "celsius"}',
            # Python dict.
            {"location": "San Francisco", "unit": "celsius"},
        ],
    )
    def test_assistant_message_with_tool_calls_string_arguments(
        self, mock_mm_data_tracker, arguments
    ):
        """Test parsing an assistant message with tool calls where arguments are JSON strings."""
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": arguments,
                    },
                }
            ],
        }

        result = parse_chat_message_content(message, mock_mm_data_tracker)

        assert result == {
            "role": "assistant",
            "content": "",
            "media": [],
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "San Francisco", "unit": "celsius"},
                    },
                }
            ],
        }

    def test_assistant_message_with_empty_tool_arguments(self, mock_mm_data_tracker):
        """Test parsing an assistant message with tool calls that have no arguments."""
        message = {
            "role": "assistant",
            "content": "Foobar",
            "tool_calls": [
                {
                    "id": "call_789",
                    "type": "function",
                    "function": {"name": "get_current_time", "arguments": None},
                }
            ],
        }

        result = parse_chat_message_content(message, mock_mm_data_tracker)

        expected = {
            "role": "assistant",
            "content": "Foobar",
            "media": [],
            "tool_calls": [
                {
                    "id": "call_789",
                    "type": "function",
                    "function": {"name": "get_current_time", "arguments": {}},
                }
            ],
        }
        assert result == expected

    def test_assistant_message_with_multiple_tool_calls(self, mock_mm_data_tracker):
        """Test parsing an assistant message with multiple tool calls."""
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "New York"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": {"timezone": "EST"}},
                },
                {"id": "call_3", "type": "function", "function": {"name": "no_args_function"}},
            ],
        }

        result = parse_chat_message_content(message, mock_mm_data_tracker)

        expected = {
            "role": "assistant",
            "content": "",
            "media": [],
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"location": "New York"}},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": {"timezone": "EST"}},
                },
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {"name": "no_args_function", "arguments": {}},
                },
            ],
        }
        assert result == expected

    @pytest.mark.parametrize(
        "message_extra_fields, expected_reasoning",
        [
            # reasoning field is used directly.
            ({"reasoning": "Let me think step by step..."}, "Let me think step by step..."),
            # reasoning field falls back to reasoning_content.
            ({"reasoning_content": "Let me think step by step..."}, "Let me think step by step..."),
            # reasoning takes priority over reasoning_content.
            (
                {"reasoning": "Primary reasoning.", "reasoning_content": "Fallback reasoning."},
                "Primary reasoning.",
            ),
            # Neither field provided -> key absent.
            ({}, None),
        ],
    )
    def test_assistant_message_reasoning(
        self, mock_mm_data_tracker, message_extra_fields, expected_reasoning
    ):
        """Test parsing assistant messages with various reasoning field combinations."""
        message = {"role": "assistant", "content": "The answer is 42.", **message_extra_fields}

        result = parse_chat_message_content(message, mock_mm_data_tracker)

        assert result["role"] == "assistant"
        assert result["content"] == "The answer is 42."
        assert result["media"] == []
        assert result.get("reasoning_content", None) == expected_reasoning

    def test_assistant_message_with_reasoning_and_tool_calls(self, mock_mm_data_tracker):
        """Test parsing an assistant message with both reasoning and tool calls."""
        message = {
            "role": "assistant",
            "content": None,
            "reasoning_content": "I need to check the weather.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                }
            ],
        }

        result = parse_chat_message_content(message, mock_mm_data_tracker)

        assert result["reasoning_content"] == "I need to check the weather."
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["arguments"] == {"city": "NYC"}


class TestParseToolMessages:
    """Test suite for tool role messages."""

    @pytest.mark.parametrize("content", ["The weather in San Francisco is 72°F and sunny.", None])
    def test_tool_message_with_tool_call_id(self, mock_mm_data_tracker, content):
        """Test parsing a tool message with tool_call_id."""
        message = {"role": "tool", "content": (content or ""), "tool_call_id": "call_123"}

        result = parse_chat_message_content(message, mock_mm_data_tracker)

        expected = {**message, "media": []}
        assert result == expected

    def test_tool_message_without_tool_call_id(self, mock_mm_data_tracker):
        """Test parsing a tool message without tool_call_id."""
        message = {
            "role": "tool",
            "content": "Database query completed successfully.",
        }

        result = parse_chat_message_content(message, mock_mm_data_tracker)

        expected = {**message, "media": []}
        assert result == expected


# ruff: noqa: E501
TEMPLATE_CHATML = """{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\\n'}}{% endif %}{% endfor %}
{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\\n' }}{% endif %}"""


@pytest.fixture
def chat_template_path(tmp_path):
    """Return the path to the chat template."""
    temp_file_path = tmp_path / "chat_template.jinja"
    with open(temp_file_path, "w") as f:
        f.write(TEMPLATE_CHATML)
    return temp_file_path


class TestLoadChatTemplate:
    """Test suite for loading chat templates."""

    def test_load_chat_template_from_path(self, chat_template_path):
        """Test loading a chat template from a path."""
        template = load_chat_template(chat_template_path)
        assert template == TEMPLATE_CHATML

    def test_load_chat_template_from_string(self):
        """Test loading a chat template from a string."""
        text = "Hello, how can I help you?"
        template = load_chat_template(text, is_literal=True)
        assert template == text

    def test_load_chat_template_from_none(self):
        """Test loading a chat template from None."""
        template = load_chat_template(None)
        assert template is None

    def test_load_chat_template_from_path_with_invalid_path(self):
        """Test loading a chat template from a path with an invalid path."""
        with pytest.raises(ValueError, match="looks like a file path"):
            load_chat_template("invalid/path/to/chat_template.jinja")

    def test_jinjalike_literal(self):
        """Test loading a chat template from a jinja-like string."""
        template = "{{ messages }}"
        template_content = load_chat_template(template)
        assert template_content == template


_MM_MODEL_TYPE = "qwen3_vl"
_IMG_PLACEHOLDER = MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder(
    model_type=_MM_MODEL_TYPE, modality="image"
)
_VIDEO_PLACEHOLDER = MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder(
    model_type=_MM_MODEL_TYPE, modality="video"
)


class TestMultimodalPlaceholderCounts:
    """Verify per-message multimodal placeholder counts.

    Regression test: previously, image/video counts leaked between messages,
    causing later text-only messages to report stale placeholder counts.
    """

    @pytest.mark.parametrize(
        "messages, expected_mm_placeholder_counts",
        [
            # Case #1: 2 messages with 1 image each, 3rd message is text-only.
            (
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {"type": "image_url", "image_url": {"url": "foo"}},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "And this one?"},
                            {"type": "image_url", "image_url": {"url": "bar"}},
                        ],
                    },
                    {"role": "user", "content": "No image here, just text"},
                ],
                [{_IMG_PLACEHOLDER: 1}, {_IMG_PLACEHOLDER: 1}, {}],
            ),
            # Case #2: first and last message have one image each, 2nd is text-only.
            (
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {"type": "image_url", "image_url": {"url": "foo"}},
                        ],
                    },
                    {"role": "user", "content": "No image here, just text"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "And this one?"},
                            {"type": "image_url", "image_url": {"url": "bar"}},
                        ],
                    },
                ],
                [{_IMG_PLACEHOLDER: 1}, {}, {_IMG_PLACEHOLDER: 1}],
            ),
            # Case #3: 1st message with several images, 2nd without any, 3rd with a video.
            (
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What do these images have in common?"},
                            {"type": "image_url", "image_url": {"url": "foo1"}},
                            {"type": "image_url", "image_url": {"url": "foo2"}},
                            {"type": "image_url", "image_url": {"url": "foo3"}},
                        ],
                    },
                    {"role": "user", "content": "No image here, just text"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the image and the video."},
                            {"type": "image_url", "image_url": {"url": "bar"}},
                            {"type": "video_url", "video_url": {"url": "baz"}},
                        ],
                    },
                ],
                [{_IMG_PLACEHOLDER: 3}, {}, {_IMG_PLACEHOLDER: 1, _VIDEO_PLACEHOLDER: 1}],
            ),
            # Case #4: 1st message with image and video, 2nd with several videos, last is text-only.
            (
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the image and the video."},
                            {"type": "image_url", "image_url": {"url": "bar"}},
                            {"type": "video_url", "video_url": {"url": "baz"}},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What do these videos have in common?"},
                            {"type": "video_url", "video_url": {"url": "foo1"}},
                            {"type": "video_url", "video_url": {"url": "foo2"}},
                            {"type": "video_url", "video_url": {"url": "foo3"}},
                        ],
                    },
                    {"role": "user", "content": "No image here, just text"},
                ],
                [{_IMG_PLACEHOLDER: 1, _VIDEO_PLACEHOLDER: 1}, {_VIDEO_PLACEHOLDER: 3}, {}],
            ),
        ],
    )
    def test_per_message_counts(self, messages, expected_mm_placeholder_counts):
        mock_config = MagicMock(spec=AutoConfig)
        mock_config.model_type = _MM_MODEL_TYPE

        _, _, mm_placeholder_counts = parse_chat_messages_coroutines(messages, mock_config, None)

        assert mm_placeholder_counts == expected_mm_placeholder_counts


class CustomError(Exception):
    pass


class TestMultimodalLoadErrorPropagation:
    """Verify that errors from multimodal loading propagate."""

    @pytest.fixture
    def mm_tracker(self):
        tracker = MagicMock()
        tracker._multimodal_server_config.media_io_kwargs = None
        return tracker

    @pytest.mark.parametrize(
        "part, loader_path",
        [
            (
                {"type": "image_url", "image_url": {"url": "http://bad-url/img.png"}},
                "tensorrt_llm.serve.chat_utils.async_load_image",
            ),
            (
                {"type": "video_url", "video_url": {"url": "http://bad-url/vid.mp4"}},
                "tensorrt_llm.serve.chat_utils.async_load_video",
            ),
            (
                {"type": "audio_url", "audio_url": {"url": "http://bad-url/aud.wav"}},
                "tensorrt_llm.serve.chat_utils.async_load_audio",
            ),
            (
                {"type": "image_embeds", "image_embeds": {"data": "notbase64"}},
                "tensorrt_llm.serve.chat_utils.load_base64_image_embeds",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_loader_exception_propagates(self, mm_tracker, part, loader_path):
        """Exceptions from async loaders must propagate, not be swallowed."""
        with patch(loader_path, side_effect=CustomError):
            result = parse_chat_message_content_part(part, mm_tracker)
            assert result is not None
            with pytest.raises(CustomError):
                await result["data"]
