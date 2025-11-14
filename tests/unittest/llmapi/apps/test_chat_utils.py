from unittest.mock import MagicMock

import pytest

from tensorrt_llm.serve.chat_utils import parse_chat_message_content


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
