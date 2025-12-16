# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import patch

import pytest

# Try to import the private function for direct testing
try:
    from tensorrt_llm.serve import harmony_adapter

    _check_channel_valid = getattr(harmony_adapter, "_check_channel_valid", None)
except (ImportError, AttributeError):
    _check_channel_valid = None

# Always import the public API
from tensorrt_llm.serve.harmony_adapter import HarmonyAdapter


@pytest.mark.skipif(
    _check_channel_valid is None,
    reason="_check_channel_valid function not available in this version",
)
class TestChannelValidation:
    """Test the _check_channel_valid function."""

    def test_empty_list_first_channel(self):
        """Test adding the first channel to an empty list."""
        generated_channels = []
        result = _check_channel_valid(generated_channels, "analysis")

        assert result is True
        assert generated_channels == ["analysis"]

    def test_different_consecutive_channels(self):
        """Test adding different consecutive channels."""
        generated_channels = ["analysis"]
        result = _check_channel_valid(generated_channels, "final")

        assert result is True
        assert generated_channels == ["analysis", "final"]

    def test_same_consecutive_channel_not_duplicated(self):
        """Test that same consecutive channel is not duplicated."""
        generated_channels = ["analysis"]
        result = _check_channel_valid(generated_channels, "analysis")

        assert result is True
        assert generated_channels == ["analysis"]  # Should not duplicate

    def test_valid_two_channel_sequence(self):
        """Test valid sequence with exactly 2 channels (analysis, final)."""
        generated_channels = ["analysis"]
        _check_channel_valid(generated_channels, "final")

        # Adding another channel when we have analysis and final
        result = _check_channel_valid(generated_channels, "final")

        assert result is True  # Still valid with 2 channels
        assert generated_channels == ["analysis", "final"]

    def test_invalid_three_channel_sequence_with_analysis_and_final(self):
        """Test invalid sequence: analysis, final, and a third channel."""
        generated_channels = ["analysis", "final"]
        result = _check_channel_valid(generated_channels, "commentary")

        # Should return False when we have analysis + final + third channel
        assert result is False
        assert generated_channels == ["analysis", "final", "commentary"]

    def test_invalid_extra_channel_after_complete_message(self):
        """Test detection of extra content after complete message."""
        # Simulate a complete message (analysis -> final)
        generated_channels = ["analysis"]
        assert _check_channel_valid(generated_channels, "final") is True
        assert generated_channels == ["analysis", "final"]

        # Now try to add another channel - should be invalid
        assert _check_channel_valid(generated_channels, "analysis") is False
        assert generated_channels == ["analysis", "final", "analysis"]

    def test_three_channels_without_both_analysis_and_final(self):
        """Test that 3 channels is OK if not both analysis and final are present."""
        # Case 1: No "analysis"
        generated_channels = ["commentary"]
        _check_channel_valid(generated_channels, "final")
        result = _check_channel_valid(generated_channels, "other")

        assert result is True  # Valid because no "analysis"

        # Case 2: No "final"
        generated_channels = ["analysis"]
        _check_channel_valid(generated_channels, "commentary")
        result = _check_channel_valid(generated_channels, "other")

        assert result is True  # Valid because no "final"

    def test_only_final_channel(self):
        """Test sequence with only final channel."""
        generated_channels = []
        result = _check_channel_valid(generated_channels, "final")

        assert result is True
        assert generated_channels == ["final"]

    def test_only_analysis_channel(self):
        """Test sequence with only analysis channel."""
        generated_channels = []
        result = _check_channel_valid(generated_channels, "analysis")

        assert result is True
        assert generated_channels == ["analysis"]

    def test_multiple_different_channels_without_trigger(self):
        """Test multiple channels that don't trigger the invalid condition."""
        generated_channels = []

        # Add multiple channels, but not both analysis and final
        assert _check_channel_valid(generated_channels, "preamble") is True
        assert _check_channel_valid(generated_channels, "commentary") is True
        result = _check_channel_valid(generated_channels, "other")

        assert len(generated_channels) == 3
        assert result is True  # Valid because neither analysis nor final

    def test_analysis_final_exact_sequence(self):
        """Test the exact sequence: analysis -> final (should be valid)."""
        generated_channels = []

        assert _check_channel_valid(generated_channels, "analysis") is True
        assert _check_channel_valid(generated_channels, "final") is True
        assert len(generated_channels) == 2

        # This is still valid - exactly 2 channels

    def test_final_analysis_then_third(self):
        """Test sequence: final -> analysis -> third (should be invalid)."""
        generated_channels = []

        assert _check_channel_valid(generated_channels, "final") is True
        assert _check_channel_valid(generated_channels, "analysis") is True
        # Now adding a third channel with both final and analysis present
        result = _check_channel_valid(generated_channels, "other")

        assert result is False
        assert len(generated_channels) == 3


class TestHarmonyStreamStateChannelValidation:
    """Test channel validation in HarmonyStreamState context."""

    @pytest.fixture
    def harmony_adapter(self):
        """Create HarmonyAdapter instance (uses real encoding)."""
        # Use the real HarmonyAdapter with real encoding
        # This is necessary because HarmonyStreamState requires a real PyHarmonyEncoding
        try:
            adapter = HarmonyAdapter(harmony_input=False, harmony_output=False)
            return adapter
        except Exception as e:
            pytest.skip(f"Cannot create HarmonyAdapter: {e}")

    def test_stream_state_tracks_generated_channels(self, harmony_adapter):
        """Test that HarmonyStreamState properly tracks generated channels."""
        # Create stream state through the public API
        request_id = "test-request-123"
        stream_state = harmony_adapter.create_stream_state(
            request_id=request_id, available_tools=None, tool_choice=None
        )

        # Verify initial state
        assert stream_state.generated_channels == []
        assert stream_state.request_id == request_id

        # Test that we can manually manipulate the channel list for validation
        # This simulates what happens during parsing
        stream_state.generated_channels = ["analysis", "final"]
        assert len(stream_state.generated_channels) == 2
        assert "analysis" in stream_state.generated_channels
        assert "final" in stream_state.generated_channels

    def test_stream_state_cleanup(self, harmony_adapter):
        """Test that stream state is properly cleaned up."""
        request_id = "test-request-456"

        # Create stream state
        _stream_state = harmony_adapter.create_stream_state(
            request_id=request_id, available_tools=None, tool_choice=None
        )

        # Verify it exists in the adapter's state
        assert request_id in harmony_adapter._stream_states

        # Clean up
        harmony_adapter.cleanup_stream_state(request_id)

        # Verify it's removed
        assert request_id not in harmony_adapter._stream_states

    def test_create_openai_streaming_response_with_should_stop_signal(self, harmony_adapter):
        """Test that create_openai_streaming_response properly handles should_stop signal."""
        import json

        request_id = "test-request-789"

        # Create stream state
        _stream_state = harmony_adapter.create_stream_state(
            request_id=request_id, available_tools=None, tool_choice=None
        )

        # Mock the stateful_stream method to return a delta with should_stop
        with patch.object(
            harmony_adapter, "stateful_stream_harmony_tokens_to_openai_deltas"
        ) as mock_stream:
            mock_stream.return_value = [{"should_stop": "Repeated message"}]

            responses, should_stop = harmony_adapter.create_openai_streaming_response(
                request_id=request_id,
                tokens=[1, 2, 3],
                available_tools=None,
                model_name="test-model",
                tool_choice=None,
            )

            # Should return should_stop=True
            assert should_stop is True
            assert len(responses) > 0

            # Check that the response contains finish_reason="stop"
            response_data = json.loads(responses[0].replace("data: ", "").strip())
            assert response_data["choices"][0]["finish_reason"] == "stop"

    def test_create_openai_streaming_response_without_stop_signal(self, harmony_adapter):
        """Test that valid sequences don't trigger should_stop."""
        import json

        request_id = "test-request-999"

        # Create stream state
        _stream_state = harmony_adapter.create_stream_state(
            request_id=request_id, available_tools=None, tool_choice=None
        )

        # Mock the stateful_stream method to return valid deltas without should_stop
        with patch.object(
            harmony_adapter, "stateful_stream_harmony_tokens_to_openai_deltas"
        ) as mock_stream:
            mock_stream.return_value = [
                {"reasoning": "some reasoning"},
                {"content": "some content"},
            ]

            responses, should_stop = harmony_adapter.create_openai_streaming_response(
                request_id=request_id,
                tokens=[1, 2, 3],
                available_tools=None,
                model_name="test-model",
                tool_choice=None,
            )

            # Should return should_stop=False
            assert should_stop is False
            assert len(responses) == 2  # Two deltas

            # Check that responses don't have finish_reason (or it's None)
            for response_str in responses:
                response_data = json.loads(response_str.replace("data: ", "").strip())
                # finish_reason may be absent (excluded) or explicitly None
                finish_reason = response_data["choices"][0].get("finish_reason")
                assert finish_reason is None


class TestHandleStreamingResponse:
    """Test handle_streaming_response integration with channel validation.

    Real-world bug scenario:
    -----------------------
    Before PR #7849, when the model generated invalid channel sequences,
    it would continue generating, resulting in massive repetition like:

    "Could you provide information? I captured... Could you provide information?
     I captured... Could you provide information? [repeated hundreds of times]"

    After PR #7849, when invalid channels are detected:
    1. should_stop=True is returned
    2. result.abort() is called
    3. Response is properly terminated: "Could you provide information?"
    """

    @pytest.fixture
    def mock_result(self):
        """Create a mock GenerationResult."""
        from unittest.mock import Mock as MockObject

        result = MockObject()

        # Create a proper output mock
        output = MockObject()
        output.token_ids_diff = [1, 2, 3]
        output.token_ids = [10, 20, 30, 40, 50]  # For usage calculation
        output.finish_reason = "length"
        output.stop_reason = None

        result.outputs = [output]
        result.abort = MockObject()  # Track if abort() was called
        return result

    def test_handle_streaming_response_aborts_on_invalid_channels(self, mock_result):
        """Test that handle_streaming_response() calls result.abort() when invalid channel sequence is detected.

        This prevents the bug where the model generates infinite repetition
        by properly stopping generation when the channel validator detects
        patterns like ["analysis", "final", "extra_content"].
        """
        from tensorrt_llm.serve.harmony_adapter import (
            get_harmony_adapter,
            handle_streaming_response,
        )

        request_id = "test-abort-request"

        # Get the real harmony adapter
        harmony_adapter = get_harmony_adapter()

        # Create stream state for this request
        harmony_adapter.create_stream_state(
            request_id=request_id, available_tools=None, tool_choice=None
        )

        try:
            # Mock create_openai_streaming_response to return should_stop=True
            # This simulates detecting an invalid channel sequence
            with patch.object(
                harmony_adapter, "create_openai_streaming_response"
            ) as mock_create_response:
                # Simulate invalid channel detection
                mock_create_response.return_value = (
                    [  # responses with valid content
                        'data: {"choices":[{"index":0,"delta":{"content":"Could you provide information?"}}]}\n\n'
                    ],
                    True,  # should_stop=True (invalid channel detected!)
                )

                # Call handle_streaming_response
                responses = handle_streaming_response(
                    tools=[],
                    tool_choice=None,
                    result=mock_result,
                    model="test-model",
                    request_id=request_id,
                    done=False,  # Not done yet, still streaming
                    num_prompt_tokens=10,
                )

                # CRITICAL ASSERTION: result.abort() should be called
                # This is what prevents the infinite repetition bug
                mock_result.abort.assert_called_once()

                # Verify responses were generated before abort
                assert len(responses) > 0

                # Verify cleanup happened (stream state removed)
                assert request_id not in harmony_adapter._stream_states

        finally:
            # Cleanup in case test failed
            if request_id in harmony_adapter._stream_states:
                harmony_adapter.cleanup_stream_state(request_id)

    def test_handle_streaming_response_continues_on_valid_channels(self, mock_result):
        """Test that handle_streaming_response() does NOT abort when valid channel sequences are processed.

        This ensures normal streaming continues for valid patterns like
        ["analysis"] or ["analysis", "final"].
        """
        from tensorrt_llm.serve.harmony_adapter import (
            get_harmony_adapter,
            handle_streaming_response,
        )

        request_id = "test-continue-request"

        # Get the real harmony adapter
        harmony_adapter = get_harmony_adapter()

        # Create stream state for this request
        harmony_adapter.create_stream_state(
            request_id=request_id, available_tools=None, tool_choice=None
        )

        try:
            # Mock create_openai_streaming_response to return should_stop=False
            # This simulates normal, valid streaming
            with patch.object(
                harmony_adapter, "create_openai_streaming_response"
            ) as mock_create_response:
                # Simulate valid channel sequence
                mock_create_response.return_value = (
                    [  # Normal streaming responses
                        'data: {"choices":[{"index":0,"delta":{"reasoning":"thinking..."}}]}\n\n',
                        'data: {"choices":[{"index":0,"delta":{"content":"response"}}]}\n\n',
                    ],
                    False,  # should_stop=False (valid channels!)
                )

                # Call handle_streaming_response
                responses = handle_streaming_response(
                    tools=[],
                    tool_choice=None,
                    result=mock_result,
                    model="test-model",
                    request_id=request_id,
                    done=False,
                    num_prompt_tokens=10,
                )

                # CRITICAL ASSERTION: result.abort() should NOT be called
                # Streaming should continue normally
                mock_result.abort.assert_not_called()

                # Verify responses were generated
                assert len(responses) > 0

                # Stream state should still exist (not cleaned up yet)
                assert request_id in harmony_adapter._stream_states

        finally:
            # Cleanup
            if request_id in harmony_adapter._stream_states:
                harmony_adapter.cleanup_stream_state(request_id)


@pytest.mark.skipif(
    _check_channel_valid is None,
    reason="_check_channel_valid function not available in this version",
)
class TestIntegrationChannelValidation:
    """Integration tests for channel validation in the full streaming flow."""

    def test_channel_validation_order_matters(self):
        """Test that order of channels matters for validation."""
        # Test 1: analysis -> final -> extra (invalid)
        channels1 = []
        _check_channel_valid(channels1, "analysis")
        _check_channel_valid(channels1, "final")
        result1 = _check_channel_valid(channels1, "extra")
        assert result1 is False

        # Test 2: extra -> analysis -> final (invalid)
        channels2 = []
        _check_channel_valid(channels2, "extra")
        _check_channel_valid(channels2, "analysis")
        _check_channel_valid(channels2, "final")
        result2 = _check_channel_valid(channels2, "another")
        assert result2 is False

    def test_repeated_same_channel_still_counts_as_two(self):
        """Test that repeated channels (after deduplication) still trigger validation."""
        channels = []

        # Add analysis multiple times (but only counts once due to deduplication)
        _check_channel_valid(channels, "analysis")
        _check_channel_valid(channels, "analysis")  # Deduplicated
        assert channels == ["analysis"]

        # Add final
        _check_channel_valid(channels, "final")
        assert channels == ["analysis", "final"]

        # Add third channel - should be invalid
        result = _check_channel_valid(channels, "other")
        assert result is False

    def test_edge_case_exactly_two_channels(self):
        """Test edge case with exactly 2 channels (analysis, final) - should be valid."""
        channels = []

        assert _check_channel_valid(channels, "analysis") is True
        assert len(channels) == 1

        assert _check_channel_valid(channels, "final") is True
        assert len(channels) == 2

        # Calling again with same channel shouldn't add more
        assert _check_channel_valid(channels, "final") is True
        assert len(channels) == 2  # Still 2, not 3

    def test_neither_analysis_nor_final_allows_many_channels(self):
        """Test that without both analysis and final, many channels are allowed."""
        channels = []

        for i in range(10):
            result = _check_channel_valid(channels, f"channel_{i}")
            assert result is True

        assert len(channels) == 10

    def test_only_one_of_analysis_or_final_allows_many_channels(self):
        """Test that with only analysis (not final) or only final (not analysis), many channels allowed."""
        # Only analysis
        channels1 = []
        _check_channel_valid(channels1, "analysis")
        for i in range(10):
            result = _check_channel_valid(channels1, f"channel_{i}")
            assert result is True

        # Only final
        channels2 = []
        _check_channel_valid(channels2, "final")
        for i in range(10):
            result = _check_channel_valid(channels2, f"channel_{i}")
            assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
