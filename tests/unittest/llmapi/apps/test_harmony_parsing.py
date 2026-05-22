# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for harmony adapter parsing logic.

Covers two fixes:
  1. fe9e1a33 — Fix harmony parsers for agentic coding use cases (#12045)
     - Tool calls on analysis channel, non-functions recipients, prefix stripping,
       reasoning_content field, remaining tokens on done.
  2. b18b2cee — Grouping deltas within one streaming interval (#12292)
     - _merge_consecutive_deltas, cached_tokens in usage info, stream_options support.
"""

import json
from unittest.mock import Mock, patch

import pytest

try:
    from tensorrt_llm.serve.harmony_adapter import (
        HarmonyAdapter,
        HarmonyStreamState,
        _create_response_message,
        _create_usage_info,
        get_harmony_adapter,
        handle_streaming_response,
    )
    from tensorrt_llm.serve.openai_protocol import StreamOptions, _logit_bias_to_embedding_bias
    from tensorrt_llm.serve.openai_server import OpenAIServer

    _harmony_available = True
except (ImportError, ModuleNotFoundError):
    _harmony_available = False

pytestmark = pytest.mark.skipif(not _harmony_available, reason="harmony_adapter not importable")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stream_state(**overrides) -> HarmonyStreamState:
    """Create a HarmonyStreamState with a mocked parser for testing."""
    adapter = HarmonyAdapter(harmony_input=False, harmony_output=False)
    state = adapter.create_stream_state(
        request_id="test-req", available_tools=None, tool_choice=None
    )
    # Replace the real parser with a mock so tests can set attributes freely
    state.parser = Mock()
    state.parser.current_channel = overrides.get("channel", "final")
    state.parser.current_recipient = overrides.get("recipient", None)
    state.parser.last_content_delta = overrides.get("delta", "")
    # generated_channels must be a real list for _check_channel_valid
    state.generated_channels = overrides.get("generated_channels", [])
    return state


def _make_mock_harmony_message(recipient=None, content_text="", content_type=None, channel=None):
    """Build a mock openai_harmony.Message for _parse_tool_call_from_harmony_message."""
    from unittest.mock import Mock as MockObj

    # We use a real TextContent if available, otherwise a mock with .text
    try:
        from openai_harmony import TextContent

        content_obj = TextContent(content_text)
    except (ImportError, TypeError):
        content_obj = MockObj()
        content_obj.text = content_text

    msg = MockObj()
    msg.recipient = recipient
    msg.content = [content_obj]
    msg.content_type = content_type
    msg.channel = channel
    return msg


def _make_mock_result(
    token_ids_diff=None, token_ids=None, finish_reason="stop", stop_reason=None, cached_tokens=0
):
    """Build a mock GenerationResult for handle_streaming_response tests."""
    output = Mock()
    output.token_ids_diff = token_ids_diff or []
    output.token_ids = token_ids or [1, 2, 3]
    output.finish_reason = finish_reason
    output.stop_reason = stop_reason

    result = Mock()
    result.outputs = [output]
    result._done = True
    result.cached_tokens = cached_tokens
    return result


# ===========================================================================
# Fix 1  —  fe9e1a33: Agentic coding use cases
# ===========================================================================


class TestStreamingToolCallOnAnalysisChannel:
    """Verify _create_delta_from_parser_state handles functions.* on analysis channel."""

    def test_functions_tool_call_on_analysis_channel(self):
        """Tool call on analysis channel should produce tool_calls, not reasoning.

        Raw harmony output from model:
          <|start|>assistant<|channel|>analysis to=functions.get_weather
          <|constrain|>json<|message|>{"city":"SF"}<|call|>

        Before the fix, the analysis branch ran first and returned this as
        {"reasoning": '{"city":"SF"}'} — the tool call was lost.
        """
        state = _make_stream_state(
            channel="analysis", recipient="functions.get_weather", delta='{"city":"SF"}'
        )
        result = state._create_delta_from_parser_state()

        assert result is not None
        assert "tool_calls" in result
        assert "reasoning" not in result
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"
        assert result["tool_calls"][0]["function"]["arguments"] == '{"city":"SF"}'

    def test_functions_tool_call_on_commentary_channel(self):
        """Regression: functions.* on commentary still produces tool_calls.

        Raw harmony output from model:
          <|start|>assistant<|channel|>commentary to=functions.get_weather
          <|constrain|>json<|message|>{"city":"SF"}<|call|>
        """
        state = _make_stream_state(
            channel="commentary", recipient="functions.get_weather", delta='{"city":"SF"}'
        )
        result = state._create_delta_from_parser_state()

        assert result is not None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_functions_tool_call_on_analysis_strips_prefix(self):
        """Verify functions. prefix is stripped from function name in delta.

        Raw harmony output from model:
          <|start|>assistant<|channel|>analysis to=functions.my_tool
          <|constrain|>json<|message|>{}<|call|>

        The recipient is "functions.my_tool" but the OpenAI API expects
        just "my_tool" in function.name.
        """
        state = _make_stream_state(channel="analysis", recipient="functions.my_tool", delta="{}")
        result = state._create_delta_from_parser_state()

        assert result["tool_calls"][0]["function"]["name"] == "my_tool"

    def test_functions_tool_call_filtered_when_not_in_available_tools(self):
        """Filtered tool call returns None when function not in available set.

        Raw harmony output from model:
          <|start|>assistant<|channel|>analysis to=functions.get_weather
          <|constrain|>json<|message|>{}<|call|>

        The user only provided tool_choice with "other_tool", so
        get_weather should be filtered out.
        """
        state = _make_stream_state(
            channel="analysis", recipient="functions.get_weather", delta="{}"
        )
        state.should_filter_tools = True
        state.available_tools = {"other_tool"}

        result = state._create_delta_from_parser_state()
        assert result is None

    def test_analysis_without_recipient_returns_reasoning(self):
        """Analysis with no recipient produces plain reasoning delta.

        Raw harmony output from model:
          <|start|>assistant<|channel|>analysis<|message|>Let me think...<|end|>

        No "to=" in the header means this is pure reasoning, not a tool call.
        """
        state = _make_stream_state(channel="analysis", recipient=None, delta="Let me think...")
        result = state._create_delta_from_parser_state()

        assert result is not None
        assert "reasoning" in result
        assert result["reasoning"] == "Let me think..."
        assert "tool_calls" not in result


class TestStreamingNonFunctionsToolCall:
    """Verify handling of non-functions.* recipients on commentary channel.

    Per the harmony spec, built-in tools (python, browser) normally use the
    **analysis** channel:
      <|start|>assistant<|channel|>analysis to=python<|message|>...<|call|>

    However, in real-world usage the model can also emit these tool calls on
    the **commentary** channel with bare recipient names.  The code handles
    both:
    - functions.* recipients on either channel (the first check)
    - bare recipients (python, browser.search, etc.) on commentary (this path)
    """

    def test_non_functions_recipient_on_commentary(self):
        """Bare recipient on commentary produces tool_calls with original name.

        Raw harmony output from model:
          <|start|>assistant<|channel|>commentary to=python<|message|>
          print("hello")<|call|>

        The canonical spec routes python through the analysis channel, but
        the model can emit it on commentary in practice.
        """
        state = _make_stream_state(channel="commentary", recipient="python", delta='print("hello")')
        result = state._create_delta_from_parser_state()

        assert result is not None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "python"
        assert result["tool_calls"][0]["function"]["arguments"] == 'print("hello")'

    def test_non_functions_recipient_filtered(self):
        """Filtered non-functions recipient returns None.

        Raw harmony output from model:
          <|start|>assistant<|channel|>commentary to=python<|message|>
          import os<|call|>

        User only declared "browser.search" as an available tool, so the
        python tool call should be filtered out.
        """
        state = _make_stream_state(channel="commentary", recipient="python", delta="code")
        state.should_filter_tools = True
        state.available_tools = {"browser.search"}  # python not included

        result = state._create_delta_from_parser_state()
        assert result is None

    def test_commentary_assistant_recipient_returns_preamble(self):
        """Commentary with recipient=assistant returns preamble content.

        Raw harmony output from model:
          <|start|>assistant<|channel|>commentary<|message|>
          **Action plan**:
          1. Generate HTML file
          2. Start the server<|end|>

        The spec shows preambles with no "to=" directive.  The parser may
        report current_recipient as "assistant" (the default) or None for
        these messages.  This test covers the "assistant" case.
        """
        state = _make_stream_state(
            channel="commentary", recipient="assistant", delta="Action plan:"
        )
        result = state._create_delta_from_parser_state()

        assert result is not None
        assert "content" in result
        assert "tool_calls" not in result or result.get("tool_calls") == []

    def test_commentary_no_recipient_returns_preamble(self):
        """Commentary with no recipient returns preamble content.

        Raw harmony output from model:
          <|start|>assistant<|channel|>commentary<|message|>
          Step 1: ...<|end|>

        No "to=" in the header — this is a plain commentary preamble.
        The parser reports current_recipient=None for this case.
        """
        state = _make_stream_state(channel="commentary", recipient=None, delta="Step 1: ...")
        result = state._create_delta_from_parser_state()

        assert result is not None
        assert "content" in result
        assert "tool_calls" not in result or result.get("tool_calls") == []


class TestParseToolCallFromHarmonyMessage:
    """Test _parse_tool_call_from_harmony_message prefix stripping and content_type handling."""

    @pytest.fixture
    def adapter(self):
        try:
            return HarmonyAdapter(harmony_input=False, harmony_output=False)
        except Exception as e:
            pytest.skip(f"Cannot create HarmonyAdapter: {e}")

    def test_parse_functions_prefixed_recipient_strips_prefix(self, adapter):
        """Verify functions. prefix is stripped with json content_type.

        Raw harmony output from model (complete non-streaming message):
          <|start|>assistant<|channel|>commentary to=functions.get_weather
          <|constrain|>json<|message|>{"city":"SF"}<|call|>

        After parse_messages_from_completion_tokens, this becomes a Message
        with recipient="functions.get_weather", content_type="<|constrain|>json",
        content=[TextContent('{"city":"SF"}')].
        """
        msg = _make_mock_harmony_message(
            recipient="functions.get_weather",
            content_text='{"city":"SF"}',
            content_type="<|constrain|>json",
        )
        result = adapter._parse_tool_call_from_harmony_message(msg)

        assert result is not None
        assert result["function"]["name"] == "get_weather"
        assert result["function"]["arguments"] == '{"city":"SF"}'
        assert result["type"] == "function"
        assert result["id"].startswith("call_")

    def test_parse_code_content_type_strips_prefix(self, adapter):
        """Verify functions. prefix is stripped with code content_type.

        The "code" content_type is typically used by the built-in python tool
        on the analysis channel:
          <|start|>assistant<|channel|>analysis to=python
          code<|message|>x = 1 + 1<|call|>

        However, the model can also emit functions.* recipients with code
        content_type.  Before the fix, the code branch used the raw recipient
        string as the function name (e.g., "functions.run_python" instead of
        "run_python").  This test verifies the prefix is stripped on that path.
        """
        msg = _make_mock_harmony_message(
            recipient="functions.run_python",
            content_text="x = 1 + 1",
            content_type="code",
        )
        result = adapter._parse_tool_call_from_harmony_message(msg)

        assert result is not None
        assert result["function"]["name"] == "run_python"
        assert result["function"]["arguments"] == "x = 1 + 1"

    def test_parse_non_prefixed_recipient_unchanged(self, adapter):
        """Non-prefixed recipient like browser.search passes through unchanged.

        Raw harmony output from model:
          <|start|>assistant<|channel|>analysis to=browser.search
          <|constrain|>json<|message|>{"query":"test"}<|call|>

        Per the harmony spec, built-in browser recipients use dotted names
        (browser.search, browser.open, browser.find) and typically appear
        on the analysis channel.  They do NOT use the "functions." namespace.

        Note: we use <|constrain|>json here to exercise the json-parsing path
        in _parse_tool_call_from_harmony_message; the spec says built-in tools
        typically omit <|constrain|>, but the code handles both.
        """
        msg = _make_mock_harmony_message(
            recipient="browser.search",
            content_text='{"query":"test"}',
            content_type="<|constrain|>json",
        )
        result = adapter._parse_tool_call_from_harmony_message(msg)

        assert result is not None
        assert result["function"]["name"] == "browser.search"

    def test_parse_assistant_recipient_returns_none(self, adapter):
        """Recipient=assistant is rejected as not a tool call.

        Raw harmony output from model:
          <|start|>assistant<|channel|>commentary<|message|>hello<|end|>

        Preamble messages have no "to=" or "to=assistant" — either way, the
        parser reports recipient="assistant" and the function should return
        None since this is content, not a tool invocation.
        """
        msg = _make_mock_harmony_message(
            recipient="assistant",
            content_text="hello",
            content_type="<|constrain|>json",
        )
        result = adapter._parse_tool_call_from_harmony_message(msg)
        assert result is None

    def test_parse_no_recipient_returns_none(self, adapter):
        """No recipient at all should be rejected.

        Raw harmony output from model:
          <|start|>assistant<|channel|>commentary<|message|>hello<|end|>

        No "to=" in header — this is a preamble, not a tool call.
        """
        msg = _make_mock_harmony_message(
            recipient=None,
            content_text="hello",
            content_type="<|constrain|>json",
        )
        result = adapter._parse_tool_call_from_harmony_message(msg)
        assert result is None


class TestNonStreamingToolCallOnAnalysisChannel:
    """Test harmony_output_to_openai message grouping for tool calls."""

    @pytest.fixture
    def adapter(self):
        try:
            return HarmonyAdapter(harmony_input=False, harmony_output=False)
        except Exception as e:
            pytest.skip(f"Cannot create HarmonyAdapter: {e}")

    def _make_harmony_msg(self, channel, recipient, text, content_type=None):
        return _make_mock_harmony_message(
            recipient=recipient, content_text=text, content_type=content_type, channel=channel
        )

    def test_nonstreaming_functions_tool_on_analysis_channel(self, adapter):
        """Functions.* tool call on analysis channel appears in tool_calls.

        Raw harmony output from model (full turn):
          <|start|>assistant<|channel|>analysis to=functions.get_weather
          <|constrain|>json<|message|>{"city":"SF"}<|call|>

        This is the key scenario the fix addresses — before the fix, tool
        calls on the analysis channel were ignored in non-streaming mode
        because only the commentary branch checked for tool calls.
        """
        msg = self._make_harmony_msg(
            channel="analysis",
            recipient="functions.get_weather",
            text='{"city":"SF"}',
            content_type="<|constrain|>json",
        )
        # Patch parse_messages_from_completion_tokens to return our mock message
        with patch.object(
            adapter.encoding, "parse_messages_from_completion_tokens", return_value=[msg]
        ):
            result = adapter.harmony_output_to_openai(
                harmony_output_tokens=[1, 2, 3],
                available_tools=None,
                tool_choice=None,
            )

        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_nonstreaming_functions_tool_on_commentary_channel(self, adapter):
        """Regression: functions.* tool call on commentary appears in tool_calls.

        Raw harmony output from model (full turn):
          <|start|>assistant<|channel|>commentary to=functions.search
          <|constrain|>json<|message|>{"q":"test"}<|call|>
        """
        msg = self._make_harmony_msg(
            channel="commentary",
            recipient="functions.search",
            text='{"q":"test"}',
            content_type="<|constrain|>json",
        )
        with patch.object(
            adapter.encoding, "parse_messages_from_completion_tokens", return_value=[msg]
        ):
            result = adapter.harmony_output_to_openai(
                harmony_output_tokens=[1, 2, 3],
                available_tools=None,
                tool_choice=None,
            )

        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "search"

    def test_nonstreaming_non_functions_tool_on_commentary(self, adapter):
        """Non-functions recipient on commentary is parsed as tool call.

        Raw harmony output from model:
          <|start|>assistant<|channel|>commentary to=python
          code<|message|>x = 1<|call|>

        The spec routes the built-in python tool through the analysis
        channel, but the model can emit it on commentary in practice.
        The code handles both channels for functions.* recipients, and
        commentary for bare recipients like "python".
        """
        msg = self._make_harmony_msg(
            channel="commentary",
            recipient="python",
            text="x = 1",
            content_type="code",
        )
        with patch.object(
            adapter.encoding, "parse_messages_from_completion_tokens", return_value=[msg]
        ):
            result = adapter.harmony_output_to_openai(
                harmony_output_tokens=[1, 2, 3],
                available_tools=None,
                tool_choice=None,
            )

        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "python"

    def test_nonstreaming_analysis_without_recipient_is_reasoning(self, adapter):
        """Analysis without recipient produces reasoning, not tool_calls.

        Raw harmony output from model (full turn with reasoning then final):
          <|start|>assistant<|channel|>analysis<|message|>
          Let me think about this...<|end|>
          <|start|>assistant<|channel|>final<|message|>The answer is 42.<|return|>

        The analysis message (no "to=") is pure chain-of-thought — it should
        map to the "reasoning" field, not be mistaken for a tool call.
        """
        msg = self._make_harmony_msg(
            channel="analysis",
            recipient=None,
            text="Let me think about this...",
            content_type=None,
        )
        with patch.object(
            adapter.encoding, "parse_messages_from_completion_tokens", return_value=[msg]
        ):
            result = adapter.harmony_output_to_openai(
                harmony_output_tokens=[1, 2, 3],
                available_tools=None,
                tool_choice=None,
            )

        assert "reasoning" in result
        assert "Let me think" in result["reasoning"]
        assert "tool_calls" not in result or len(result.get("tool_calls", [])) == 0


class TestReasoningContentField:
    """Verify reasoning_content is set alongside reasoning in deltas and messages."""

    def test_streaming_delta_has_reasoning_content(self):
        """Streaming reasoning delta sets both reasoning and reasoning_content.

        Raw harmony output from model (streaming, token by token):
          <|start|>assistant<|channel|>analysis<|message|>step 1: think<|end|>

        The StreamableParser emits current_channel="analysis" with
        last_content_delta="step 1: think".  The adapter converts this to
        {"reasoning": "step 1: think"} and then the SSE formatter must set
        BOTH reasoning and reasoning_content on the DeltaMessage for client
        compatibility (some clients read reasoning_content, others reasoning).
        """
        adapter = HarmonyAdapter(harmony_input=False, harmony_output=False)
        request_id = "test-reasoning-content"
        adapter.create_stream_state(request_id=request_id, available_tools=None, tool_choice=None)
        try:
            with patch.object(
                adapter,
                "stateful_stream_harmony_tokens_to_openai_deltas",
                return_value=[{"reasoning": "step 1: think"}],
            ):
                responses, _ = adapter.create_openai_streaming_response(
                    request_id=request_id,
                    tokens=[1, 2, 3],
                    available_tools=None,
                    model_name="test-model",
                    tool_choice=None,
                )

            assert len(responses) == 1
            data = json.loads(responses[0].replace("data: ", "").strip())
            delta = data["choices"][0]["delta"]
            assert delta["reasoning"] == "step 1: think"
            assert delta["reasoning_content"] == "step 1: think"
        finally:
            adapter.cleanup_stream_state(request_id)

    def test_create_response_message_has_reasoning_content(self):
        """Non-streaming response message sets both reasoning and reasoning_content.

        Raw harmony output from model (complete non-streaming turn):
          <|start|>assistant<|channel|>analysis<|message|>
          Let me calculate...<|end|>
          <|start|>assistant<|channel|>final<|message|>
          The answer is 42.<|return|>

        After harmony_output_to_openai parses this, the result dict has
        reasoning="Let me calculate..." and content="The answer is 42.".
        _create_response_message must propagate reasoning to BOTH fields.
        """
        parsed = {
            "role": "assistant",
            "content": "The answer is 42.",
            "reasoning": "Let me calculate...",
        }
        result = _create_response_message(parsed)

        assert result["reasoning"] == "Let me calculate..."
        assert result["reasoning_content"] == "Let me calculate..."

    def test_create_response_message_without_reasoning(self):
        """No reasoning in input means neither reasoning field appears.

        Raw harmony output from model (no analysis channel at all):
          <|start|>assistant<|channel|>final<|message|>Hello!<|return|>

        Model went straight to final — no chain-of-thought reasoning.
        """
        parsed = {"role": "assistant", "content": "Hello!"}
        result = _create_response_message(parsed)

        assert "reasoning" not in result
        assert "reasoning_content" not in result


class TestRemainingTokensOnDone:
    """Verify handle_streaming_response processes leftover tokens on done=True."""

    def test_done_with_remaining_tokens_processes_them(self):
        """Remaining tokens on done=True are processed before finish_reason chunk.

        Raw harmony output from model (full turn, showing which tokens arrive
        in the final batch):
          <|start|>assistant<|channel|>final<|message|>Hello world tail<|return|>
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^
          already streamed in prior iterations                   token_ids_diff
                                                                 on done=True

        The last batch of tokens (encoding " tail<|return|>") arrives in
        output.token_ids_diff when done=True.  Before the fix, these tokens
        were dropped — the stream jumped straight to the finish_reason chunk,
        losing the final content.
        """
        mock_result = _make_mock_result(token_ids_diff=[10, 20, 30])
        harmony_adapter = get_harmony_adapter()
        request_id = "test-done-remaining"
        harmony_adapter.create_stream_state(
            request_id=request_id, available_tools=None, tool_choice=None
        )
        try:
            with patch.object(
                harmony_adapter,
                "create_openai_streaming_response",
                return_value=(
                    ['data: {"choices":[{"index":0,"delta":{"content":"tail"}}]}\n\n'],
                    False,
                ),
            ) as mock_create:
                responses = handle_streaming_response(
                    tools=[],
                    tool_choice=None,
                    result=mock_result,
                    model="test-model",
                    request_id=request_id,
                    done=True,
                    num_prompt_tokens=5,
                    first_iteration=False,
                )

            # create_openai_streaming_response should have been called
            # with the remaining tokens
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args
            assert call_kwargs[1]["tokens"] == [10, 20, 30]

            # The content delta should appear before the final chunk
            assert any("tail" in r for r in responses)
            # The last data chunk should have a finish_reason
            last_data = [r for r in responses if r.startswith("data: ") and "finish_reason" in r]
            assert len(last_data) >= 1
        finally:
            if request_id in harmony_adapter._stream_states:
                harmony_adapter.cleanup_stream_state(request_id)

    def test_done_with_empty_tokens_skips_processing(self):
        """Empty token_ids_diff on done=True skips token processing.

        Does not call create_openai_streaming_response — only emits the final chunk.

        Raw harmony output from model:
          <|start|>assistant<|channel|>final<|message|>Hello<|return|>

        All content tokens were already streamed in prior iterations.
        The final callback arrives with done=True but token_ids_diff=[]
        because there are no new tokens — only the stop signal.
        """
        mock_result = _make_mock_result(token_ids_diff=[])
        harmony_adapter = get_harmony_adapter()
        request_id = "test-done-empty"
        harmony_adapter.create_stream_state(
            request_id=request_id, available_tools=None, tool_choice=None
        )
        try:
            with patch.object(harmony_adapter, "create_openai_streaming_response") as mock_create:
                responses = handle_streaming_response(
                    tools=[],
                    tool_choice=None,
                    result=mock_result,
                    model="test-model",
                    request_id=request_id,
                    done=True,
                    num_prompt_tokens=5,
                    first_iteration=False,
                )

            # Should NOT have been called because token_ids_diff is empty
            mock_create.assert_not_called()

            # Should still have the final finish_reason chunk + usage
            assert len(responses) >= 1
        finally:
            if request_id in harmony_adapter._stream_states:
                harmony_adapter.cleanup_stream_state(request_id)

    def test_done_first_iteration_prepends_role_delta(self):
        """First iteration with done=True prepends role=assistant delta.

        Raw harmony output from model (very short response, all in one batch):
          <|start|>assistant<|channel|>final<|message|>hi<|return|>

        The entire response fits in a single executor callback with done=True.
        Since this is the first (and only) iteration, the OpenAI streaming
        protocol requires the first SSE event to carry role="assistant".
        """
        mock_result = _make_mock_result(token_ids_diff=[10, 20])
        harmony_adapter = get_harmony_adapter()
        request_id = "test-done-first-iter"
        harmony_adapter.create_stream_state(
            request_id=request_id, available_tools=None, tool_choice=None
        )
        try:
            with patch.object(
                harmony_adapter,
                "create_openai_streaming_response",
                return_value=(
                    ['data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'],
                    False,
                ),
            ):
                responses = handle_streaming_response(
                    tools=[],
                    tool_choice=None,
                    result=mock_result,
                    model="test-model",
                    request_id=request_id,
                    done=True,
                    num_prompt_tokens=5,
                    first_iteration=True,
                )

            # First data chunk should have role="assistant"
            first_data = json.loads(responses[0].replace("data: ", "").strip())
            assert first_data["choices"][0]["delta"]["role"] == "assistant"
        finally:
            if request_id in harmony_adapter._stream_states:
                harmony_adapter.cleanup_stream_state(request_id)


# ===========================================================================
# Fix 2  —  b18b2cee: Grouping deltas within one streaming interval
# ===========================================================================


class TestMergeConsecutiveDeltas:
    """Pure unit tests for HarmonyStreamState._merge_consecutive_deltas."""

    @staticmethod
    def merge(deltas):
        return HarmonyStreamState._merge_consecutive_deltas(deltas)

    def test_empty_list(self):
        """Empty input → empty output.

        Occurs when the token batch contains only special tokens
        (e.g., <|start|>assistant<|channel|>) that produce no content deltas.
        """
        assert self.merge([]) == []

    def test_single_delta_unchanged(self):
        """A single delta should pass through untouched.

        Raw harmony (single token in analysis channel):
          <|start|>assistant<|channel|>analysis<|message|>a

        Parser emits one delta: {"reasoning": "a"}.  Nothing to merge with.
        """
        assert self.merge([{"reasoning": "a"}]) == [{"reasoning": "a"}]

    def test_merge_consecutive_reasoning(self):
        """Two consecutive reasoning-only deltas should merge into one.

        Raw harmony (two content tokens within one analysis message):
          <|start|>assistant<|channel|>analysis<|message|>ab...

        Parser emits per-token: {"reasoning": "a"}, {"reasoning": "b"}.
        After merge: {"reasoning": "ab"} — one SSE event instead of two.
        """
        result = self.merge([{"reasoning": "a"}, {"reasoning": "b"}])
        assert result == [{"reasoning": "ab"}]

    def test_merge_three_consecutive_reasoning(self):
        """Three consecutive reasoning deltas should all merge.

        Raw harmony (three content tokens in analysis):
          <|start|>assistant<|channel|>analysis<|message|>abc...

        Parser emits: {"reasoning":"a"}, {"reasoning":"b"}, {"reasoning":"c"}.
        Merge collapses all three into one.
        """
        result = self.merge(
            [
                {"reasoning": "a"},
                {"reasoning": "b"},
                {"reasoning": "c"},
            ]
        )
        assert result == [{"reasoning": "abc"}]

    def test_merge_consecutive_content(self):
        """Two consecutive content-only deltas should merge.

        Raw harmony (two tokens in final channel):
          <|start|>assistant<|channel|>final<|message|>hello world...

        Parser emits: {"content": "hello "}, {"content": "world"}.
        """
        result = self.merge([{"content": "hello "}, {"content": "world"}])
        assert result == [{"content": "hello world"}]

    def test_merge_content_with_tool_calls_key(self):
        """Content deltas with matching key sets (including tool_calls=[]) merge.

        Preamble content deltas carry tool_calls=[] and should still merge — this happens with preamble content.

        Raw harmony (commentary preamble, multiple tokens):
          <|start|>assistant<|channel|>commentary<|message|>Step 1 done<|end|>

        Parser emits preamble deltas with empty tool_calls:
          {"content": "Step 1", "tool_calls": []},
          {"content": " done", "tool_calls": []}
        Both have identical key sets → safe to merge.
        """
        result = self.merge(
            [
                {"content": "Step 1", "tool_calls": []},
                {"content": " done", "tool_calls": []},
            ]
        )
        assert len(result) == 1
        assert result[0]["content"] == "Step 1 done"
        assert result[0]["tool_calls"] == []

    def test_no_merge_content_different_keys(self):
        """Content deltas with different key sets should NOT merge.

        This happens when a token batch spans a channel transition within
        one process_token_batch call.  For example, the batch might contain
        the end of a final message and the start of a commentary preamble:

        Raw harmony tokens in one batch:
          ...final content tokens...<|end|>
          <|start|>assistant<|channel|>commentary<|message|>b...

        First delta: {"content": "a"} (from the final channel — no tool_calls key).
        Second delta: {"content": "b", "tool_calls": []} (commentary preamble).
        Different key sets → must not merge.
        """
        result = self.merge(
            [
                {"content": "a"},
                {"content": "b", "tool_calls": []},
            ]
        )
        assert len(result) == 2

    def test_merge_consecutive_tool_calls_same_id(self):
        """Tool call deltas with same id merge by concatenating arguments.

        Raw harmony (tool call arguments arriving token by token):
          <|start|>assistant<|channel|>commentary to=functions.f
          <|constrain|>json<|message|>{"a":1}<|call|>

        Parser emits per-token as the JSON is generated:
          {"tool_calls": [{id:"call_1", function:{arguments:'{"a'}}]}
          {"tool_calls": [{id:"call_1", function:{arguments:'":1}'}}]}
        Merge concatenates arguments: '{"a":1}'.
        """
        result = self.merge(
            [
                {"tool_calls": [{"id": "call_1", "function": {"arguments": '{"a'}}]},
                {"tool_calls": [{"id": "call_1", "function": {"arguments": '":1}'}}]},
            ]
        )
        assert len(result) == 1
        assert result[0]["tool_calls"][0]["function"]["arguments"] == '{"a":1}'

    def test_no_merge_tool_calls_different_ids(self):
        """Tool call deltas with different ids stay separate.

        Raw harmony (parallel tool calls in one turn — each is a separate
        commentary message ending with <|call|>):
          <|start|>assistant<|channel|>commentary to=functions.get_weather
          <|constrain|>json<|message|>{"city":"SF"}<|call|>
          <|start|>assistant<|channel|>commentary to=functions.get_time
          <|constrain|>json<|message|>{"tz":"PST"}<|call|>

        Each tool call gets a unique call_id.  They must stay separate even
        if they appear in the same token batch.
        """
        result = self.merge(
            [
                {"tool_calls": [{"id": "call_1", "function": {"arguments": "a"}}]},
                {"tool_calls": [{"id": "call_2", "function": {"arguments": "b"}}]},
            ]
        )
        assert len(result) == 2

    def test_no_merge_reasoning_with_extra_keys(self):
        """Reasoning delta with extra keys does not merge with pure reasoning.

        This is a synthetic edge case — in practice, a delta won't have both
        "reasoning" and "content".  The guard ensures the merge logic only
        fires for pure single-type deltas (len(delta) == 1).
        """
        result = self.merge(
            [
                {"reasoning": "a", "content": "x"},
                {"reasoning": "b"},
            ]
        )
        assert len(result) == 2

    def test_interleaved_types_not_merged(self):
        """Interleaved delta types stay separate.

        This tests the merge function in isolation with synthetic input.
        In practice, an analysis→final→analysis sequence would be flagged by
        _check_channel_valid, but _merge_consecutive_deltas operates on
        already-produced deltas and must handle any ordering correctly.

        Input: reasoning, content, reasoning — not consecutive same-type.
        """
        result = self.merge(
            [
                {"reasoning": "a"},
                {"content": "b"},
                {"reasoning": "c"},
            ]
        )
        assert len(result) == 3

    def test_mixed_sequence(self):
        """Mixed sequence of reasoning, preamble, and tool call merges correctly.

        Consecutive reasoning, then consecutive preamble content, then a tool call — should produce 3 merged deltas.

        Raw harmony (turn with reasoning, preamble, then tool call):
          <|start|>assistant<|channel|>analysis<|message|>think more<|end|>
          <|start|>assistant<|channel|>commentary<|message|>The answer<|end|>
          <|start|>assistant<|channel|>commentary to=functions.f
          <|constrain|>json<|message|>{}<|call|>

        Tokens within each message produce per-token deltas:
        reasoning×2 (from analysis), content×2 (from commentary preamble —
        note: preamble deltas include tool_calls=[]), tool_calls×1.
        The merge collapses each consecutive run into one.
        """
        result = self.merge(
            [
                {"reasoning": "think "},
                {"reasoning": "more"},
                {"content": "The ", "tool_calls": []},
                {"content": "answer", "tool_calls": []},
                {"tool_calls": [{"id": "c1", "function": {"arguments": "{}"}}]},
            ]
        )
        assert len(result) == 3
        assert result[0] == {"reasoning": "think more"}
        assert result[1] == {"content": "The answer", "tool_calls": []}
        assert "tool_calls" in result[2]

    def test_should_stop_not_merged(self):
        """should_stop deltas are never merged.

        should_stop is emitted by _check_channel_valid when the model
        generates an invalid channel sequence (e.g., analysis → final →
        extra_content).  It's a control signal, not content.
        """
        result = self.merge(
            [
                {"should_stop": "reason1"},
                {"should_stop": "reason2"},
            ]
        )
        assert len(result) == 2

    def test_tool_calls_with_content_key_not_merged(self):
        """tool_calls deltas with extra content key are not merged.

        This is a defensive check — the merge logic requires both deltas
        to be pure tool_calls (no "content" or "reasoning" keys).
        """
        result = self.merge(
            [
                {"tool_calls": [{"id": "c1", "function": {"arguments": "a"}}]},
                {"tool_calls": [{"id": "c1", "function": {"arguments": "b"}}], "content": "x"},
            ]
        )
        assert len(result) == 2


class TestCreateUsageInfoCachedTokens:
    """Test _create_usage_info reports cached_tokens via PromptTokensDetails."""

    def _make_outputs(self, *token_counts):
        """Create mock outputs with given token_id lengths."""
        outputs = []
        for count in token_counts:
            out = Mock()
            out.token_ids = list(range(count))
            outputs.append(out)
        return outputs

    def test_cached_tokens_zero(self):
        """cached_tokens=0 should still populate prompt_tokens_details.

        Scenario: a fresh request with no KV cache hit.  The harmony output
        itself is irrelevant here — this tests the usage info formatting
        that wraps any harmony response.

        Expected OpenAI usage JSON:
          "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                    "total_tokens": 15,
                    "prompt_tokens_details": {"cached_tokens": 0}}
        """
        usage = _create_usage_info(10, self._make_outputs(5), cached_tokens=0)
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == 0

    def test_cached_tokens_positive(self):
        """Positive cached_tokens should be reflected in the details.

        Scenario: a multi-turn conversation where 50 of 100 prompt tokens
        were served from the KV cache (prefix caching hit).

        Expected OpenAI usage JSON:
          "prompt_tokens_details": {"cached_tokens": 50}
        """
        usage = _create_usage_info(100, self._make_outputs(20), cached_tokens=50)
        assert usage.prompt_tokens_details.cached_tokens == 50

    def test_cached_tokens_default(self):
        """When cached_tokens is not passed, it should default to 0.

        Backwards compatibility: callers that don't pass cached_tokens
        (e.g., older code paths) should get cached_tokens=0, not None.
        """
        usage = _create_usage_info(10, self._make_outputs(5))
        assert usage.prompt_tokens_details.cached_tokens == 0

    def test_usage_token_counts_correct_with_cached(self):
        """Token counts are unaffected by cached_tokens value.

        Scenario: 100 prompt tokens (40 cached), two output sequences
        of 30 and 20 tokens (e.g., from beam search / best_of > 1).

        cached_tokens is informational — it should not reduce prompt_tokens
        or affect total_tokens arithmetic.
        """
        usage = _create_usage_info(100, self._make_outputs(30, 20), cached_tokens=40)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50  # 30 + 20
        assert usage.total_tokens == 150  # 100 + 50
        assert usage.prompt_tokens_details.cached_tokens == 40


class TestStreamOptionsUsage:
    """Test handle_streaming_response respects stream_options.include_usage."""

    def test_stream_options_include_usage_true(self):
        """include_usage=True should append a usage chunk at the end.

        Raw harmony output from model:
          <|start|>assistant<|channel|>final<|message|>Hello<|return|>

        OpenAI streaming protocol with stream_options={"include_usage": true}:
        the final SSE event should be a usage-only chunk like:
          data: {"choices":[], "usage":{"prompt_tokens":10, ...}}
        """
        mock_result = _make_mock_result(token_ids_diff=[])
        harmony_adapter = get_harmony_adapter()
        request_id = "test-usage-true"
        harmony_adapter.create_stream_state(
            request_id=request_id, available_tools=None, tool_choice=None
        )
        try:
            stream_opts = StreamOptions(include_usage=True)
            responses = handle_streaming_response(
                tools=[],
                tool_choice=None,
                result=mock_result,
                model="test-model",
                request_id=request_id,
                done=True,
                num_prompt_tokens=10,
                first_iteration=False,
                stream_options=stream_opts,
            )

            # There should be a chunk with "usage" in it
            usage_chunks = [r for r in responses if "usage" in r and "prompt_tokens" in r]
            assert len(usage_chunks) == 1
        finally:
            if request_id in harmony_adapter._stream_states:
                harmony_adapter.cleanup_stream_state(request_id)

    def test_stream_options_include_usage_false(self):
        """include_usage=False should NOT append a usage chunk.

        Raw harmony output from model:
          <|start|>assistant<|channel|>final<|message|>Hello<|return|>

        OpenAI streaming protocol with stream_options={"include_usage": false}:
        the stream should end with the finish_reason chunk only — no usage.
        This reduces SSE payload size when the client doesn't need token counts.
        """
        mock_result = _make_mock_result(token_ids_diff=[])
        harmony_adapter = get_harmony_adapter()
        request_id = "test-usage-false"
        harmony_adapter.create_stream_state(
            request_id=request_id, available_tools=None, tool_choice=None
        )
        try:
            stream_opts = StreamOptions(include_usage=False)
            responses = handle_streaming_response(
                tools=[],
                tool_choice=None,
                result=mock_result,
                model="test-model",
                request_id=request_id,
                done=True,
                num_prompt_tokens=10,
                first_iteration=False,
                stream_options=stream_opts,
            )

            # No chunk should contain usage with prompt_tokens
            usage_chunks = [r for r in responses if "prompt_tokens" in r]
            assert len(usage_chunks) == 0
        finally:
            if request_id in harmony_adapter._stream_states:
                harmony_adapter.cleanup_stream_state(request_id)

    def test_stream_options_none_defaults_to_include(self):
        """stream_options=None defaults to including usage.

        Raw harmony output from model:
          <|start|>assistant<|channel|>final<|message|>Hello<|return|>

        When the client doesn't pass stream_options at all (None), the server
        should default to including usage — this preserves backwards
        compatibility with clients that expect usage in every stream.
        """
        mock_result = _make_mock_result(token_ids_diff=[])
        harmony_adapter = get_harmony_adapter()
        request_id = "test-usage-default"
        harmony_adapter.create_stream_state(
            request_id=request_id, available_tools=None, tool_choice=None
        )
        try:
            responses = handle_streaming_response(
                tools=[],
                tool_choice=None,
                result=mock_result,
                model="test-model",
                request_id=request_id,
                done=True,
                num_prompt_tokens=10,
                first_iteration=False,
                stream_options=None,
            )

            # Default behavior: usage chunk should be present
            usage_chunks = [r for r in responses if "usage" in r and "prompt_tokens" in r]
            assert len(usage_chunks) == 1
        finally:
            if request_id in harmony_adapter._stream_states:
                harmony_adapter.cleanup_stream_state(request_id)


def test_none_tokenizer_num_postprocess_workers():
    server = object.__new__(OpenAIServer)
    server.tokenizer = None
    assert server._vocab_size is None
    with pytest.raises(ValueError, match="logit_bias requires a tokenizer"):
        _logit_bias_to_embedding_bias({"0": 1.0}, vocab_size=None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
