# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import re
import time
import traceback
import uuid
from typing import Any, List, Literal, Tuple

from openai_harmony import (Author, Conversation, DeveloperContent,
                            HarmonyEncodingName, HarmonyError, Message,
                            ReasoningEffort, Role, StreamableParser,
                            SystemContent, TextContent, ToolDescription,
                            load_harmony_encoding)

from tensorrt_llm.executor import GenerationResult
from tensorrt_llm.logger import logger

# yapf: disable
from .openai_protocol import (ChatCompletionMessageParam,
                              ChatCompletionResponse,
                              ChatCompletionResponseChoice,
                              ChatCompletionResponseStreamChoice,
                              ChatCompletionStreamResponse,
                              ChatCompletionToolsParam, ChatMessage,
                              DeltaFunctionCall, DeltaMessage, DeltaToolCall,
                              UsageInfo)

# yapf: enable


def _check_channel_valid(generated_channels: List[str], channel: str) -> bool:

    if len(generated_channels) == 0 or generated_channels[-1] != channel:
        generated_channels.append(channel)

    logger.debug(f"generated_channels: {generated_channels}")
    if "analysis" in generated_channels and "final" in generated_channels and len(
            generated_channels) > 2:
        return False

    return True


class HarmonyStreamState:
    """
    Maintains harmony parsing state for a single request across multiple token batches.
    This is essential because vLLM provides incremental token batches, but harmony
    parsing requires continuous state for channel transitions and tool calls.
    """

    def __init__(self,
                 request_id: str,
                 encoding,
                 available_tools: list[dict[str, Any]] | None = None,
                 tool_choice: str | None = None):
        self.request_id = request_id
        self.encoding = encoding
        self.parser = StreamableParser(encoding, role=Role.ASSISTANT)

        # Tool filtering - same logic as non-streaming case
        self.available_tools = set()
        self.should_filter_tools = False  # Track if we should filter tools

        if tool_choice == "none":
            # tool_choice="none" means suppress ALL tool calls, regardless of available_tools
            self.should_filter_tools = True
            self.available_tools = set()  # Empty set means no tools are allowed
        elif available_tools is not None:
            # Normal case: filter based on available tools
            self.should_filter_tools = True
            self.available_tools = {
                tool.get("function", {}).get("name", "") if tool.get(
                    "name", None) is None else tool.get("name")
                for tool in available_tools
            }
            self.available_tools.discard("")
        # else: available_tools is None and tool_choice != "none" means no filtering (allow all tools)

        # Persistent state across token batches
        self.tool_calls = {}  # tool_call_id -> {id, name, arguments, index}
        self.tool_call_index = 0
        self.tokens_processed = 0

        # Track channel states for token preservation
        self.has_preamble_content = False
        self.current_channel_state = None  # "analysis", "commentary_preamble", "commentary_tool", "final"
        self.generated_channels = [
        ]  # Track generated channels to avoid generating too many messages
        self.channel_started = False  # Track if we've sent opening token for current channel

        # Track sent arguments for tool call streaming deltas
        self.sent_tool_arguments = {}  # tool_call_id -> sent_arguments_length

        logger.debug(f"Created HarmonyStreamState for request {request_id}")

    def get_parser(self) -> StreamableParser:
        return self.parser

    def process_token_batch(self, tokens: list[int]) -> list[dict[str, Any]]:
        """
        Process a batch of tokens while maintaining parsing state.
        Returns OpenAI-compatible deltas for this batch.
        """
        deltas = []
        self.tokens_processed += len(tokens)

        for token in tokens:
            # Store previous state for transition detection
            prev_channel = self.parser.current_channel
            prev_recipient = self.parser.current_recipient

            # Process the token
            self.parser.process(token)

            # Detect channel/recipient transitions AFTER processing each token
            channel_changed = prev_channel != self.parser.current_channel
            recipient_changed = prev_recipient != self.parser.current_recipient

            if channel_changed or recipient_changed:
                # Mark any active tool calls as completed if we're leaving a tool call
                if prev_channel == "commentary" and prev_recipient and "functions." in str(
                        prev_recipient):
                    func_name = str(prev_recipient).split("functions.")[-1]
                    for tool_id, tool_info in self.tool_calls.items():
                        if tool_info["name"] == func_name and tool_info.get(
                                "active", True):
                            tool_info["active"] = False

                # Send closing token for previous channel
                closing_delta = self._create_closing_token_delta()
                if closing_delta:
                    deltas.append(closing_delta)

                # Reset channel state for new channel
                self.channel_started = False
                self.current_channel_state = None

            # Process content deltas (only generate if there's actual content)
            if self.parser.last_content_delta:
                delta = self._create_delta_from_parser_state()
                if delta:
                    deltas.append(delta)

        return deltas

    def process_token_batch_to_messages(self,
                                        tokens: list[int]) -> list[Message]:
        """
        Process a batch of tokens while maintaining parsing state.
        Returns OpenAI Messages for Responses API
        """
        self.tokens_processed += len(tokens)

        for token in tokens:
            # Store previous state for transition detection
            prev_channel = self.parser.current_channel
            prev_recipient = self.parser.current_recipient

            # Process the token
            self.parser.process(token)

            # Detect channel/recipient transitions AFTER processing each token
            channel_changed = prev_channel != self.parser.current_channel
            recipient_changed = prev_recipient != self.parser.current_recipient

            if channel_changed or recipient_changed:
                # Mark any active tool calls as completed if we're leaving a tool call
                if prev_channel == "commentary" and prev_recipient and "functions." in str(
                        prev_recipient):
                    func_name = str(prev_recipient).split("functions.")[-1]
                    for tool_id, tool_info in self.tool_calls.items():
                        if tool_info["name"] == func_name and tool_info.get(
                                "active", True):
                            tool_info["active"] = False

                # Reset channel state for new channel
                self.channel_started = False
                self.current_channel_state = None

        return self.parser.messages

    def _create_closing_token_delta(self) -> dict[str, Any] | None:
        """Create closing token delta for channel transition."""
        if not self.current_channel_state or not self.channel_started:
            return None

        if self.current_channel_state == "commentary_preamble":
            return {"content": "<|end|>", "tool_calls": []}
        elif self.current_channel_state == "final":
            return None  # No closing token needed for final content

        return None

    def _create_delta_from_parser_state(self) -> dict[str, Any] | None:
        """Create OpenAI delta from current parser state."""
        if not self.parser.last_content_delta:
            return None

        if not _check_channel_valid(self.generated_channels,
                                    self.parser.current_channel):
            return {"should_stop": "Repeated message"}

        if self.parser.current_channel == "analysis":
            # Analysis channel -> reasoning (no token wrapping needed)
            self.current_channel_state = "analysis"
            return {"reasoning": self.parser.last_content_delta}

        elif self.parser.current_channel == "commentary":
            if self.parser.current_recipient and "functions." in str(
                    self.parser.current_recipient):
                # Tool call in commentary channel
                func_name = str(
                    self.parser.current_recipient).split("functions.")[-1]
                self.current_channel_state = "commentary_tool"

                # Check if tool is allowed
                if self.should_filter_tools and func_name not in self.available_tools:
                    logger.debug("Request %s: tool %s not in available tools",
                                 self.request_id, func_name)
                    return None

                # Get or create tool call
                tool_id = self._get_or_create_tool_call(func_name)

                # Accumulate arguments
                self.tool_calls[tool_id][
                    "arguments"] += self.parser.last_content_delta

                # Create tool call delta - return only the new content delta, not accumulated
                return {
                    "tool_calls": [{
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": self.parser.
                            last_content_delta  # Only the new content delta
                        },
                        "index": self.tool_calls[tool_id]["index"]
                    }]
                }
            else:
                # Commentary preamble -> content with token preservation
                self.has_preamble_content = True
                self.current_channel_state = "commentary_preamble"

                # Send opening token if this is the first content in this channel
                if not self.channel_started:
                    self.channel_started = True
                    return {
                        "content":
                        f"<|channel|>commentary<|message|>{self.parser.last_content_delta}",
                        "tool_calls": []
                    }
                else:
                    return {
                        "content": self.parser.last_content_delta,
                        "tool_calls": []
                    }

        elif self.parser.current_channel == "final":
            # Final channel -> content with token preservation
            self.current_channel_state = "final"

            # Send raw content directly (no special tokens for final channel)
            if self.has_preamble_content:
                return {
                    "content": self.parser.last_content_delta,
                    "tool_calls": []
                }
            else:
                return {"content": self.parser.last_content_delta}
        else:
            logger.debug("Request %s: no delta generated for channel=%s",
                         self.request_id, self.parser.current_channel)
            return None

    def _get_or_create_tool_call(self, func_name: str) -> str:
        """Get existing tool call ID or create new one for function."""
        # Look for existing tool call with same function name that's still being constructed
        # (tool calls are completed when they receive an <|end|> or <|call|> token)
        for tool_id, tool_info in self.tool_calls.items():
            if tool_info["name"] == func_name and tool_info.get("active", True):
                return tool_id

        # Create new tool call
        tool_id = f"call_{uuid.uuid4().hex[:8]}"
        self.tool_calls[tool_id] = {
            "id": tool_id,
            "name": func_name,
            "arguments": "",
            "index": self.tool_call_index,
            "active": True
        }
        self.tool_call_index += 1
        logger.debug("Request %s: created new tool call %s for function %s",
                     self.request_id, tool_id, func_name)
        return tool_id

    def get_debug_info(self) -> dict[str, Any]:
        """Get debug information about the parser state."""
        return {
            "request_id":
            self.request_id,
            "tokens_processed":
            self.tokens_processed,
            "current_channel":
            self.parser.current_channel,
            "current_recipient":
            str(self.parser.current_recipient)
            if self.parser.current_recipient else None,
            "tool_calls":
            self.tool_calls,
            "last_content_delta":
            self.parser.last_content_delta,
            "current_channel_state":
            self.current_channel_state,
            "generated_channels":
            self.generated_channels,
            "channel_started":
            self.channel_started,
            "has_preamble_content":
            self.has_preamble_content,
            "available_tools":
            self.available_tools,
            "should_filter_tools":
            self.should_filter_tools
        }

    def finalize_request(self) -> dict[str, Any] | None:
        """
        Finalize the request and return any remaining closing token delta.
        Call this when the request is complete to ensure proper token closure.
        """
        return self._create_closing_token_delta()


class HarmonyAdapter:
    """
    Stateless adapter for converting between OpenAI API chat format and Harmony format.

    Mapping strategy:
    - Commentary preamble: tool_calls: [] (empty array)
    - Commentary tool call: tool_calls: [...] (populated array)
    - Final content: no tool_calls field
    - Analysis: reasoning field

    Parameters:
    - harmony_input: If True, expect harmony format input (no conversion)
    - harmony_output: If True, return harmony format output (no conversion)
    - default_reasoning_effort: Default reasoning effort level ("low", "medium", "high")
    """

    def __init__(
            self,
            harmony_input: bool = False,
            harmony_output: bool = False,
            default_reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH):
        self.encoding = load_harmony_encoding(
            HarmonyEncodingName.HARMONY_GPT_OSS)
        self.harmony_input = harmony_input
        self.harmony_output = harmony_output
        self.default_reasoning_effort = default_reasoning_effort

        # Stateful stream parsers for requests (request_id -> HarmonyStreamState)
        self._stream_states: dict[str, HarmonyStreamState] = {}

        # Special token mappings for harmony format parsing with tiktoken
        self._harmony_special_tokens = {
            "<|start|>": 200006,
            "<|end|>": 200007,
            "<|message|>": 200008,
            "<|channel|>": 200005,
            "<|return|>": 200002,
            "<|call|>": 200012,
            "<|refusal|>": 200013,
            "<|constrain|>": 200009,
        }

    def get_stream_state(self, request_id: str) -> HarmonyStreamState | None:
        return self._stream_states.get(request_id, None)

    def get_stop_tokens(self) -> list[int]:
        """
        Return the list of stop token IDs for Harmony format.
        Includes: <|return|> <|call|>.
        Note: <|end|> is not a stop token, it is a message separator.
        """
        return self.encoding.stop_tokens_for_assistant_actions()

    def collect_content(
        self,
        message: ChatCompletionMessageParam,
    ) -> str | None:
        """Collect and return the current content from the stream state."""
        content = message.get("content")
        contents: list[str] = []
        if content:
            # Str
            if isinstance(content, str):
                contents.append(content.strip())
            # list[ChatCompletionContentPartTextParam]
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        contents.append(part["text"].strip())
            return "\n".join(contents)
        return None

    def _safe_decode_utf8(self,
                          tokens: list[int],
                          fallback_prefix: str = "") -> str:
        """
        Safely decode tokens with proper error handling.

        Now uses the new encoding.decode() method which gracefully handles UTF-8 issues
        by replacing invalid sequences with replacement characters (�) instead of failing.
        This eliminates the need for complex tiktoken fallbacks in most cases.

        Args:
            tokens: List of token IDs to decode
            fallback_prefix: Prefix to add to fallback representation

        Returns:
            Decoded string (may contain � for invalid UTF-8 sequences)
        """
        if not tokens:
            return ""

        return self.encoding.decode(tokens)

    def harmony_system_message(self,
                               reasoning_effort: ReasoningEffort | None = None,
                               system_instructions: list[str] = []) -> Message:
        """Create system content with configurable reasoning effort."""
        effort = reasoning_effort or self.default_reasoning_effort
        content = "\n".join(system_instructions)
        system_content = SystemContent.new()\
            .with_model_identity(content or "You are ChatGPT, a large language model trained by OpenAI.")\
            .with_reasoning_effort(effort)\
            .with_knowledge_cutoff("2024-06")\
            .with_conversation_start_date(time.strftime("%Y-%m-%d"))\
            .with_required_channels(["analysis", "commentary", "final"])
        return Message.from_role_and_content(Role.SYSTEM, system_content)

    def harmony_developer_message(
        self,
        instructions: list[str],
        tools: list[dict[str, Any]] | None,
    ) -> Message:
        """Create developer content with tools and instructions."""
        dev_content = DeveloperContent.new()
        # Add instructions if available - convert list to string
        if instructions:
            instructions_text = "\n".join(instructions)
            dev_content = dev_content.with_instructions(instructions_text)

        # Add tools if available - convert OpenAI tools to ToolDescription objects
        if tools:
            tool_descriptions = []
            for tool in tools:
                func_def = tool.get("function", {})
                func_name = func_def.get("name", "")
                func_description = func_def.get("description", "")
                func_parameters = func_def.get("parameters", {})

                if func_name:
                    tool_desc = ToolDescription.new(func_name,
                                                    func_description,
                                                    parameters=func_parameters)
                    tool_descriptions.append(tool_desc)

            if tool_descriptions:
                dev_content = dev_content.with_function_tools(tool_descriptions)
        return Message.from_role_and_content(Role.DEVELOPER, dev_content)

    def _extract_preamble_and_final_content(
        self,
        content: str | None,
    ) -> tuple[str, str]:
        """
        Parse content to extract preamble and final content.
        Returns (preamble_content, final_content).
        """
        if not content:
            return "", ""

        # Extract preamble using harmony format
        preamble_content = self._extract_between_tokens(
            content, "<|channel|>commentary<|message|>", "<|end|>")

        # Final content is everything that's not preamble
        final_content = content
        if preamble_content:
            # Remove the preamble part from final content
            preamble_pattern = re.escape(
                f"<|channel|>commentary<|message|>{preamble_content}<|end|>")
            final_content = re.sub(preamble_pattern, "", content).strip()

        return preamble_content, final_content

    def _extract_between_tokens(self, text: str, start_token: str,
                                end_token: str) -> str:
        """Extract content between start and end tokens."""
        if not text or start_token not in text:
            return ""

        pattern = re.escape(start_token) + r"(.*?)" + re.escape(end_token)
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def harmony_assistant_message(
            self, assistant_msg: ChatCompletionMessageParam,
            external_tools: set[str],
            should_filter_external_tools: bool) -> list[Message]:
        """Convert assistant message from OpenAI API chat format to Harmony format."""

        content = self.collect_content(assistant_msg)
        tool_calls = assistant_msg.get("tool_calls", [])
        reasoning_content = assistant_msg.get("reasoning_content", "")

        messages: list[Message] = []

        preamble_content, final_content = self._extract_preamble_and_final_content(
            content)
        # Add reasoning content as analysis channel if it does not have final content
        if final_content == "" and reasoning_content != "":
            messages.append(
                Message.from_role_and_content(
                    Role.ASSISTANT, reasoning_content).with_channel("analysis"))

        preamble_content, final_content = self._extract_preamble_and_final_content(
            content)

        if "tool_calls" in assistant_msg:
            if not tool_calls:  # Empty array = commentary preamble + final content
                # Add preamble content to commentary channel if present
                if preamble_content.strip():
                    messages.append(
                        Message.from_role_and_content(
                            Role.ASSISTANT,
                            preamble_content).with_channel("commentary"))

                # Add final content to final channel if present
                if final_content.strip():
                    messages.append(
                        Message.from_role_and_content(
                            Role.ASSISTANT,
                            final_content).with_channel("final"))
            else:  # Populated array = commentary tool calls
                # Add preamble if present
                if preamble_content.strip():
                    messages.append(
                        Message.from_role_and_content(
                            Role.ASSISTANT,
                            preamble_content).with_channel("commentary"))

                ## TODO: Remove tool calls filter, since built-in tools are not available
                # Filter tool calls
                filtered_calls = self._filter_tool_calls(
                    tool_calls, external_tools, should_filter_external_tools)

                # Add individual tool calls
                for tool_call in filtered_calls:
                    func_name = tool_call.get("function", {}).get("name", "")
                    arguments = tool_call.get("function",
                                              {}).get("arguments", "{}")

                    # Tool call rendering in harmony format
                    # Both header orders are supported per partner v9 update (Format B is what is used):
                    # Format A: <|channel|>commentary to=functions.calculate <|constrain|>json<|message|>
                    # Format B: to=functions.calculate<|channel|>commentary <|constrain|>json<|message|>
                    # The library handles both automatically
                    messages.append(
                        Message.from_role_and_content(
                            Role.ASSISTANT,
                            arguments).with_channel("commentary").
                        with_recipient(f"functions.{func_name}"
                                       ).with_content_type("<|constrain|>json"))

                # Add final content if present
                if final_content.strip():
                    messages.append(
                        Message.from_role_and_content(
                            Role.ASSISTANT,
                            final_content).with_channel("final"))
        else:  # No tool_calls field = final content
            if final_content.strip() or content.strip():
                # Use final content if available, otherwise use raw content
                actual_content = final_content if final_content.strip(
                ) else content
                messages.append(
                    Message.from_role_and_content(
                        Role.ASSISTANT, actual_content).with_channel("final"))

        return messages

    def openai_to_harmony_tokens(
            self,
            openai_messages: list[ChatCompletionMessageParam],
            available_tools: list[dict[str, Any]] | None = None,
            reasoning_effort: ReasoningEffort | None = None,
            tool_choice: str | None = None) -> list[int]:
        """
        Convert OpenAI API chat messages to Harmony token sequence.

        Args:
            openai_messages: List of OpenAI chat messages
            available_tools: Optional list of available tools
            explicit_reasoning_effort: Optional explicit reasoning effort that takes precedence over system message parsing
            tool_choice: Optional tool choice ("auto", "none", etc.)

        Reasoning effort precedence:
        1. explicit_reasoning_effort parameter (highest priority)
        2. System message parsing ("reasoning: medium" or "reasoning_effort: low")
        3. self.default_reasoning_effort (fallback)

        Tool choice behavior:
        - "auto" (default): Model decides whether to use tools based on available_tools
        - "none": Tools are completely removed before sending to model
        - Any other value: Falls back to "auto" behavior

        Format notes:
        1. Both header orders are supported:
           - <|start|>assistant<|channel|>commentary to=functions.calculate <|constrain|>json<|message|>
           - Current in rendering: <|start|>assistant to=functions.calculate<|channel|>commentary <|constrain|>json<|message|>

        2. Stop token:
           - Expected: <|call|> for tool calls
           - Occasionally, <|return|> for tool calls
           - Should use <|return|> only for final responses, <|call|> for tool calls

        TODO: Check stripping of CoT from multi-turn conversations
        """
        try:
            if self.harmony_input:
                # Input is already in harmony format
                if isinstance(openai_messages, str):
                    # If input is a string, it should already be tokenized using an external tokenizer
                    # The harmony library doesn't provide string-to-tokens encoding
                    raise ValueError(
                        "harmony_input=True with string input is not supported. "
                        "The harmony library doesn't provide string-to-tokens encoding. "
                        "Please provide either: "
                        "1. A Conversation object, or "
                        "2. Pre-tokenized input as list[int], or "
                        "3. Use harmony_input=False for OpenAI format conversion"
                    )
                elif isinstance(openai_messages, Conversation):
                    # Input is a Conversation object
                    tokens = self.encoding.render_conversation_for_completion(
                        openai_messages, Role.ASSISTANT)
                elif isinstance(openai_messages, list) and all(
                        isinstance(x, int) for x in openai_messages):
                    # Input is already tokenized
                    tokens = openai_messages
                else:
                    # Input should be a Conversation object when harmony_input=True
                    raise ValueError(
                        "harmony_input=True expects either a Conversation object or pre-tokenized list[int]. "
                        f"Got: {type(openai_messages)}")
                return tokens

            # Handle tool_choice parameter
            if tool_choice == "none":
                # Don't pass any tools to harmony model - model won't see any tool definitions
                available_tools = None
            # For "auto" or any other value, use normal behavior (pass tools as-is)

            # Extract available external tool names
            external_tools = set()
            should_filter_external_tools = False
            if available_tools is not None:
                should_filter_external_tools = True
                external_tools = {
                    tool.get("function", {}).get("name", "")
                    for tool in available_tools
                }
                external_tools.discard("")

            # Collect system instructions from OpenAI messages
            system_instructions: list[str] = []
            dev_instructions: list[str] = []
            conversation_messages: list[ChatCompletionMessageParam] = [
            ]  # Store user/assistant/tool messages
            tool_call_map = {}  # Map tool_call_id to function name

            # collect system instructions and conversation messages
            for msg in openai_messages:
                role = msg["role"]

                # Collect system instructions for system message
                if role == "system":
                    content = self.collect_content(msg)
                    if content:
                        system_instructions.append(content.strip())
                # Collect developer instructions
                elif role == "developer":
                    content = self.collect_content(msg)
                    if content:
                        dev_instructions.append(content.strip())
                elif role == "assistant":
                    # Track tool calls for later mapping
                    tool_calls = msg.get("tool_calls", [])
                    for tool_call in tool_calls:
                        tool_call_id = tool_call.get("id")
                        func_name = tool_call.get("function", {}).get("name")
                        if tool_call_id and func_name:
                            tool_call_map[tool_call_id] = func_name
                    conversation_messages.append(msg)
                else:
                    # Store non-system messages for later processing
                    conversation_messages.append(msg)

            harmony_messages: list[Message] = []

            # Add system message: reasoning effort, knowledge cutoff, conversation start date, required channels
            harmony_messages.append(
                self.harmony_system_message(reasoning_effort,
                                            system_instructions))

            # Developer message with instructions and tools
            harmony_messages.append(
                self.harmony_developer_message(dev_instructions,
                                               available_tools))

            # Process conversation messages (user/assistant/tool)
            for msg in conversation_messages:
                role = msg["role"]

                if role == "user":
                    user_msg = Message.from_role_and_content(
                        Role.USER,
                        self.collect_content(msg) or "")
                    harmony_messages.append(user_msg)
                elif role == "assistant":
                    if reasoning := msg.get("reasoning", None):
                        # Add reasoning COT for tool calling
                        cot_message = Message.from_role_and_content(
                            Role.ASSISTANT, reasoning).with_channel("analysis")
                        harmony_messages.append(cot_message)
                    assistant_messages = self.harmony_assistant_message(
                        msg, external_tools, should_filter_external_tools)
                    harmony_messages.extend(assistant_messages)
                elif role == "tool":
                    # Convert tool responses to proper harmony tool format
                    # OpenAI format: {"role": "tool", "tool_call_id": "...", "content": "...", "name": "..."}
                    # Harmony format: Author.new(Role.TOOL, "tool_name") with recipient="assistant"

                    tool_call_id = msg.get("tool_call_id")
                    tool_content = self.collect_content(msg) or ""

                    # Get the actual function name from the tool_call_map
                    if tool_call_id and tool_call_id in tool_call_map:
                        tool_name = tool_call_map[tool_call_id]
                    else:
                        # Fallback to name field or default
                        tool_name = msg.get("name", "tool")

                    # Add namespace prefix if missing
                    if tool_name and not "." in tool_name:
                        tool_name = f"functions.{tool_name}"

                    tool_author = Author.new(Role.TOOL, tool_name)
                    tool_message = Message.from_author_and_content(
                        tool_author, tool_content)
                    # IMPORTANT: Tool messages must have recipient="assistant" according to harmony.md
                    tool_message = tool_message.with_recipient(
                        "assistant").with_channel("commentary")
                    harmony_messages.append(tool_message)
                else:
                    logger.warning(f"Unknown message role: {role}")

            conversation = Conversation.from_messages(harmony_messages)
            tokens = self.encoding.render_conversation_for_completion(
                conversation,
                Role.ASSISTANT,
            )
            return tokens

        except Exception as e:
            logger.error(
                f"Failed to convert OpenAI messages to harmony tokens: {e}")
            logger.debug(
                "Falling back to error - harmony library doesn't provide string encoding"
            )
            raise RuntimeError(
                f"Failed to convert messages to harmony tokens: {e}. "
                "The harmony library doesn't provide string-to-tokens encoding for fallback."
            )

    def openai_to_harmony_prompt(
            self,
            openai_messages: list[ChatCompletionMessageParam],
            available_tools: list[dict[str, Any]] | None = None,
            reasoning_effort: ReasoningEffort | None = None,
            tool_choice: str | None = None) -> str:
        """
        Convert OpenAI API chat messages to Harmony prompt string.
        """
        tokens = self.openai_to_harmony_tokens(
            openai_messages,
            available_tools,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice)
        return self._safe_decode_utf8(tokens, "HARMONY_TOKENS: ")

    def _apply_harmony_to_openai_mapping(self, analysis_content: str,
                                         commentary_preambles: list[str],
                                         tool_calls: list[dict[str, Any]],
                                         final_content: str) -> dict[str, Any]:
        """
        Apply Harmony to OpenAI mapping with content preservation.
        Preserves both preamble and final content for round-trip fidelity.
        """

        content_parts = []

        # Add preamble content using proper harmony format
        if commentary_preambles:
            preamble_text = "\n".join(commentary_preambles).strip()
            if preamble_text:
                content_parts.append(
                    f"<|channel|>commentary<|message|>{preamble_text}<|end|>")

        # Add final content directly (no wrapping needed)
        if final_content.strip():
            content_parts.append(final_content.strip())

        # Combine content
        combined_content = "\n".join(content_parts) if content_parts else ""

        # Start with base fields
        result = {"role": "assistant"}

        # Add reasoning content first if present
        if analysis_content.strip():
            result["reasoning"] = analysis_content.strip()

        # Add content
        if tool_calls:
            # Has tool calls - include tool_calls array
            result["content"] = combined_content if combined_content else None
            result["tool_calls"] = tool_calls
        elif commentary_preambles:
            # Has preambles but no tool calls - empty array mapping
            result["content"] = combined_content if combined_content else ""
            result["tool_calls"] = []
        else:
            # Final content only - no tool_calls field
            result["content"] = combined_content if combined_content else ""

        return result

    def _parse_tool_call_from_harmony_message(
            self, msg: Message) -> dict[str, Any] | None:
        """Parse tool call from harmony message."""
        msg_recipient = getattr(msg, 'recipient', None)
        msg_content = getattr(msg, 'content', [])
        msg_content_type = getattr(msg, 'content_type', None)

        if not msg_recipient or msg_recipient == "assistant":
            return None

        # # Clean up function name by removing constraint tokens that may be incorrectly included
        # # The harmony library sometimes includes ><|constrain|>json in the recipient
        # if ">" in function_name:
        #     function_name = function_name.split(">")[0]

        # Extract arguments from message content
        function_call_args = ""
        for content in msg_content:
            if isinstance(content, TextContent):
                function_call_args += content.text
            elif hasattr(content, 'text'):
                function_call_args += content.text
            else:
                function_call_args += str(content)

        if msg_content_type and "<|constrain|>json" in msg_content_type:
            try:
                function_name = str(msg_recipient).split("functions.")[-1]
                # Validate JSON
                json.loads(function_call_args)
                return {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": function_call_args
                    }
                }
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse tool call arguments as JSON: %s",
                    function_call_args)
                return None
        elif msg_content_type and "code" in msg_content_type:
            function_name = str(msg_recipient)
            return {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": function_call_args
                }
            }
        else:
            logger.warning(
                f"Unsupported message content type for tool call: {msg_content_type}"
            )
            return None

    def _strip_incomplete_messages(self, tokens: list[int]) -> list[int]:
        """
        Strip incomplete messages from the end of a token sequence.

        An incomplete message is one that starts with <|start|> but doesn't have
        a corresponding <|message|> token, indicating the message header is incomplete.
        This ensures render_conversation_for_completion can work properly.

        Args:
            tokens: List of token IDs potentially ending with incomplete messages

        Returns:
            List of token IDs with incomplete messages removed from the end
        """
        if not tokens:
            return tokens

        # Special token IDs from the harmony format
        START_TOKEN = 200006  # <|start|>
        MESSAGE_TOKEN = 200008  # <|message|>

        # Make a copy to avoid modifying the original
        clean_tokens = tokens[:]

        while clean_tokens:
            # Find the last <|start|> token
            last_start_idx = None
            for i in range(len(clean_tokens) - 1, -1, -1):
                if clean_tokens[i] == START_TOKEN:
                    last_start_idx = i
                    break

            # If no <|start|> token found, we're done
            if last_start_idx is None:
                break

            # Check if there's a <|message|> token after the last <|start|>
            has_message_token = False
            for i in range(last_start_idx + 1, len(clean_tokens)):
                if clean_tokens[i] == MESSAGE_TOKEN:
                    has_message_token = True
                    break
                # If we encounter another <|start|> before <|message|>, this is incomplete
                if clean_tokens[i] == START_TOKEN:
                    break

            # If no <|message|> token after the last <|start|>, this is incomplete
            if not has_message_token:
                # Remove everything from the last <|start|> onwards
                clean_tokens = clean_tokens[:last_start_idx]
            else:
                # The last message is complete, we're done
                break

        return clean_tokens

    def harmony_output_to_openai(
            self,
            harmony_output_tokens: list[int],
            available_tools: list[dict[str, Any]] | None = None,
            tool_choice: str | None = None) -> dict[str, Any]:
        """
        Parse Harmony model output tokens and convert to OpenAI API response format. Non-streaming.
        Returns a single message dict.
        """
        if self.harmony_output:
            # Output should remain in harmony format
            return {
                "role":
                "assistant",
                "content":
                self._safe_decode_utf8(harmony_output_tokens,
                                       "HARMONY_OUTPUT: ")
            }

        # Extract available external tool names and set filtering behavior
        external_tools = set()
        should_filter_external_tools = False

        if tool_choice == "none":
            # tool_choice="none" means suppress ALL tool calls, regardless of available_tools
            should_filter_external_tools = True
            external_tools = set()  # Empty set means no tools are allowed
        elif available_tools is not None:
            # Normal case: filter based on available tools
            should_filter_external_tools = True
            external_tools = {
                tool.get("function", {}).get("name", "")
                for tool in available_tools
            }
            external_tools.discard("")
        # else: available_tools is None and tool_choice != "none" means no filtering (allow all)

        # First, try to decode the raw tokens to see what we're working with
        try:
            # Strip incomplete messages and stop tokens before parsing
            # The harmony library's parse_messages_from_completion_tokens expects clean tokens without stop tokens
            # and complete message structures.
            clean_tokens = self._strip_incomplete_messages(
                harmony_output_tokens)

            # Use parse_messages_from_completion_tokens for non-streaming complete parsing
            try:
                harmony_messages = self.encoding.parse_messages_from_completion_tokens(
                    clean_tokens, role=Role.ASSISTANT)
            except (HarmonyError, UnicodeDecodeError,
                    ValueError) as parse_error:
                logger.warning(
                    "Failed to parse harmony messages from tokens: %s",
                    parse_error)
                logger.debug("Problematic clean tokens (%d): %s",
                             len(clean_tokens), clean_tokens)
                # Fallback to raw text parsing
                raise RuntimeError(f"Harmony parsing failed: {parse_error}"
                                   )  # This will be caught by outer try-catch

            # Group messages by type
            analysis_content = ""
            commentary_preambles = []
            tool_calls = []
            final_content = ""
            generated_channels = []

            for msg in harmony_messages:
                msg_channel = getattr(msg, 'channel', None)
                msg_recipient = getattr(msg, 'recipient', None)
                msg_content = getattr(msg, 'content', [])

                if not _check_channel_valid(generated_channels, msg_channel):
                    continue

                if msg_channel == "analysis":
                    for content in msg_content:
                        if isinstance(content, TextContent):
                            analysis_content += content.text
                        elif hasattr(content, 'text'):
                            analysis_content += content.text
                        else:
                            analysis_content += str(content)

                elif msg_channel == "commentary":
                    if msg_recipient and msg_recipient != 'assistant':
                        # Tool call
                        tool_call = self._parse_tool_call_from_harmony_message(
                            msg)
                        if tool_call and self._is_tool_call_allowed(
                                tool_call, external_tools,
                                should_filter_external_tools):
                            tool_calls.append(tool_call)
                    else:
                        # Preamble
                        for content in msg_content:
                            if isinstance(content, TextContent):
                                commentary_preambles.append(content.text)
                            elif hasattr(content, 'text'):
                                commentary_preambles.append(content.text)
                            else:
                                commentary_preambles.append(str(content))

                elif msg_channel == "final":
                    for content in msg_content:
                        if isinstance(content, TextContent):
                            final_content += content.text
                        elif hasattr(content, 'text'):
                            final_content += content.text
                        else:
                            final_content += str(content)

                elif msg_channel is None:
                    # Handle messages without explicit channel - might be final content
                    for content in msg_content:
                        if isinstance(content, TextContent):
                            final_content += content.text
                        elif hasattr(content, 'text'):
                            final_content += content.text
                        else:
                            final_content += str(content)

            # Apply Harmony to OpenAI mapping with content preservation
            result = self._apply_harmony_to_openai_mapping(
                analysis_content, commentary_preambles, tool_calls,
                final_content)

            return result

        except Exception as e:
            raw_text = self._safe_decode_utf8(harmony_output_tokens,
                                              "HARMONY _OUTPUT: ")
            logger.warning("Failed to parse harmony output: %s. Raw output: %s",
                           e, raw_text)
            logger.debug("Detailed error: %s", traceback.format_exc())

            # Check if raw_text contains a decode error (fallback content)
            if "HARMONY_OUTPUT:" in raw_text:
                # The raw text itself couldn't be decoded, create a safe fallback
                fallback_content = f"[Response processing failed - {len(harmony_output_tokens)} tokens generated]"
                logger.warning(
                    "Raw text decoding also failed, using fallback content")
            else:
                # Raw text was decoded successfully, use it
                fallback_content = raw_text

            return {
                "role": "assistant",
                "content": fallback_content,
                "_harmony_parsing_failed":
                True  # Internal flag to indicate harmony parsing failed
            }

    def stream_harmony_tokens_to_openai_deltas(
            self,
            tokens: list[int],
            available_tools: list[dict[str, Any]] | None = None,
            tool_choice: str | None = None) -> list[dict[str, Any]]:
        """
        Convert harmony tokens to OpenAI streaming deltas. DEPRECATED - Use stateful streaming instead.

        This method processes tokens in a single batch and attempts to generate deltas for streaming.
        However, without state tracking, it may produce inconsistent or incomplete results.
        Use create_openai_streaming_response() for production streaming.
        """
        if self.harmony_output:
            # Return harmony format directly
            return [{
                "content":
                self._safe_decode_utf8(tokens, "HARMONY_TOKENS: ")
            }]

        parser = StreamableParser(self.encoding, role=Role.ASSISTANT)

        # State tracking for streaming
        current_tool_call_id = None
        current_tool_call_name = None
        tool_call_buffer = ""

        # Channel transition tracking for special tokens
        previous_channel = None
        commentary_preamble_started = False
        final_started = False

        deltas = []

        for token in tokens:
            parser.process(token)

            # Handle channel transitions for special tokens
            if parser.current_channel != previous_channel:
                # Channel transition detected

                # Close previous channel if needed
                if previous_channel == "commentary" and commentary_preamble_started:
                    # Close preamble with harmony end token
                    deltas.append({"content": "<|end|>", "tool_calls": []})
                    commentary_preamble_started = False
                elif previous_channel == "final" and final_started:
                    # No closing token needed for final content
                    final_started = False

                # Open new channel if needed
                if parser.current_channel == "commentary" and not (
                        parser.current_recipient
                        and "functions." in str(parser.current_recipient)):
                    # Starting commentary preamble with harmony format
                    deltas.append({
                        "content": "<|channel|>commentary<|message|>",
                        "tool_calls": []
                    })
                    commentary_preamble_started = True
                elif parser.current_channel == "final":
                    # No opening token needed for final content
                    final_started = True

                previous_channel = parser.current_channel

            # Process content deltas
            if parser.last_content_delta:
                if parser.current_channel == "analysis":
                    # Analysis -> reasoning_content (no special tokens)
                    deltas.append(
                        {"reasoning_content": parser.last_content_delta})

                elif parser.current_channel == "commentary":
                    if parser.current_recipient and "functions." in str(
                            parser.current_recipient):
                        # Tool call
                        func_name = parser.current_recipient.split(
                            "functions.")[-1]

                        if current_tool_call_name != func_name:
                            # New tool call
                            current_tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
                            current_tool_call_name = func_name
                            tool_call_buffer = ""

                        tool_call_buffer += parser.last_content_delta

                        deltas.append({
                            "tool_calls": [{
                                "id": current_tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": func_name,
                                    "arguments": tool_call_buffer
                                }
                            }]
                        })
                    else:
                        # Preamble content (already wrapped with special tokens)
                        deltas.append({
                            "content": parser.last_content_delta,
                            "tool_calls": []
                        })

                elif parser.current_channel == "final":
                    # Final content (already wrapped with special tokens)
                    deltas.append({"content": parser.last_content_delta})

        # Close any remaining open channels
        if commentary_preamble_started:
            deltas.append({"content": "<|end|>", "tool_calls": []})
        # No closing needed for final content

        return deltas

    # TODO: Implement stateful streaming parser for production use
    # Check multi-byte utf-8 tokens (like emojis) parsing
    #
    def stateful_stream_harmony_tokens_to_openai_deltas(
            self,
            request_id: str,
            tokens: list[int],
            available_tools: list[dict[str, Any]] | None = None,
            tool_choice: str | None = None) -> list[dict[str, Any]]:
        """
        Process tokens using stateful parsing.

        This method maintains persistent state across multiple calls for the same request,
        ensuring proper channel transitions and tool call handling.

        Args:
            request_id: Request ID to maintain state per request
            tokens: New tokens from this iteration (not cumulative)
            available_tools: Available tools for filtering

        Returns:
            List of OpenAI-compatible delta dictionaries
        """
        stream_state = self._stream_states.get(request_id, None)
        if stream_state is None:
            stream_state = self.create_stream_state(request_id, available_tools,
                                                    tool_choice)

        # decoded_tokens = self.encoding.decode_utf8(tokens)
        # logger.info(">> DECODED TOKENS: %r", decoded_tokens)

        try:
            deltas = stream_state.process_token_batch(tokens)
            # logger.info(">> GENERATED DELTAS: %s", deltas)
            return deltas
        except (HarmonyError, UnicodeDecodeError, ValueError):
            logger.error(
                f"Streaming: Failed to process token batch of {len(tokens)} tokens for request {request_id}",
            )
            logger.debug("Problematic streaming tokens: %s", tokens)

            # Return empty deltas to continue processing
            return []

    def stateful_stream_harmony_tokens_to_openai_messages(
            self,
            request_id: str,
            tokens: list[int],
            available_tools: list[dict[str, Any]] | None = None,
            tool_choice: str | None = None) -> list[Message]:
        """
        Process tokens using stateful parsing.

        This method maintains persistent state across multiple calls for the same request,
        ensuring proper channel transitions and tool call handling.

        Args:
            request_id: Request ID to maintain state per request
            tokens: New tokens from this iteration
            available_tools: Available tools for filtering

        Returns:
            List of OpenAI Messages
        """
        stream_state = self._stream_states.get(request_id, None)
        if stream_state is None:
            stream_state = self.create_stream_state(request_id, available_tools,
                                                    tool_choice)

        try:
            messages = stream_state.process_token_batch_to_messages(tokens)
            return messages
        except (HarmonyError, UnicodeDecodeError, ValueError):
            logger.error(
                f"Streaming: Failed to process token batch of {len(tokens)} tokens for request {request_id}",
            )
            logger.debug(f"Problematic streaming tokens: {tokens}")

            return []

    def create_openai_streaming_response(
            self,
            request_id: str,
            tokens: list[int],
            available_tools: list[dict[str, Any]] | None = None,
            model_name: str = "harmony-model",
            tool_choice: str | None = None) -> Tuple[list[str], bool]:
        """
        Create properly formatted OpenAI streaming responses from harmony tokens.

        Args:
            request_id: Request ID for state tracking
            tokens: New tokens from this iteration
            available_tools: Available tools for filtering
            model_name: Model name for response

        Returns:
            List of properly formatted streaming response strings
        """
        # Get harmony deltas with error handling
        try:
            harmony_deltas = self.stateful_stream_harmony_tokens_to_openai_deltas(
                request_id, tokens, available_tools, tool_choice)
        except Exception as e:
            logger.warning(
                f"Error creating harmony deltas for request {request_id}: {e}")
            logger.debug(f"Problematic tokens: {tokens}")
            raise e

        responses = []

        # Track if this is the first delta for this request to set role
        stream_state = self._stream_states.get(request_id)

        # Now process the actual harmony deltas (all with role=null)
        for harmony_delta in harmony_deltas:
            # Convert harmony delta to proper OpenAI DeltaMessage
            delta_message = DeltaMessage(
                role=None)  # Always null for content deltas

            # Handle reasoning content
            if "reasoning" in harmony_delta:
                delta_message.reasoning = harmony_delta["reasoning"]
                # tool_calls will use default factory (empty list)

            # Handle regular content
            elif "content" in harmony_delta:
                delta_message.content = harmony_delta["content"]
                # tool_calls will use default factory (empty list)

            # Handle tool calls
            elif "tool_calls" in harmony_delta:
                # Convert tool calls to proper OpenAI streaming format
                tool_calls = []
                for tool_call in harmony_delta["tool_calls"]:
                    tool_call_id = tool_call.get("id")
                    delta_arguments = tool_call.get("function",
                                                    {}).get("arguments", "")

                    # Track what we've already sent for this tool call (only if stream_state exists)
                    if stream_state and tool_call_id and tool_call_id not in stream_state.sent_tool_arguments:
                        # First time seeing this tool call - send full info including arguments
                        stream_state.sent_tool_arguments[tool_call_id] = True

                        delta_tool_call = DeltaToolCall(
                            id=tool_call_id,
                            type="function",
                            index=tool_call.get("index", 0),
                            function=DeltaFunctionCall(
                                name=tool_call.get("function", {}).get("name"),
                                arguments=
                                delta_arguments  # Use the actual delta arguments
                            ))
                    elif stream_state and tool_call_id:
                        # Subsequent calls - send the delta arguments directly
                        delta_tool_call = DeltaToolCall(
                            id=None,  # null for subsequent calls
                            type="function",
                            index=tool_call.get("index", 0),
                            function=DeltaFunctionCall(
                                name=None,  # null for subsequent calls
                                arguments=
                                delta_arguments  # Use delta arguments directly
                            ))
                    else:
                        # No stream state - send delta arguments directly
                        delta_tool_call = DeltaToolCall(
                            id=tool_call_id,
                            type="function",
                            index=tool_call.get("index", 0),
                            function=DeltaFunctionCall(
                                name=tool_call.get("function", {}).get("name"),
                                arguments=delta_arguments))

                    tool_calls.append(delta_tool_call)

                delta_message.tool_calls = tool_calls

            # Default case - set all fields to appropriate null values
            else:
                delta_message.content = None
                delta_message.reasoning_content = None
                # tool_calls will use default factory (empty list)

            should_stop = ("should_stop" in harmony_delta)

            # Create the streaming response
            choice = ChatCompletionResponseStreamChoice(
                index=0,
                delta=delta_message,
                logprobs=None,
                finish_reason="stop" if should_stop else None,
                stop_reason=None)

            stream_response = ChatCompletionStreamResponse(model=model_name,
                                                           choices=[choice],
                                                           usage=None)

            # Convert to string
            response_json = stream_response.model_dump_json(exclude_none=True)
            responses.append(f"data: {response_json}\n\n")

            if should_stop:
                return responses, should_stop

        return responses, False

    def create_stream_state(
            self,
            request_id: str,
            available_tools: list[dict[str, Any]] | None = None,
            tool_choice: str | None = None) -> HarmonyStreamState:
        """
        Create a stateful harmony stream parser for a request.
        This maintains state across multiple token batches for proper streaming.
        """
        if request_id in self._stream_states:
            logger.warning(
                "Stream state already exists for request %s, replacing",
                request_id)

        stream_state = HarmonyStreamState(
            request_id=request_id,
            encoding=self.encoding,
            available_tools=available_tools,
            tool_choice=tool_choice,
        )
        self._stream_states[request_id] = stream_state
        return stream_state

    def cleanup_stream_state(self, request_id: str) -> None:
        """
        Clean up stream state for a completed/aborted request.
        Call this when a request finishes to free memory.
        """
        if request_id in self._stream_states:
            del self._stream_states[request_id]
            logger.debug(f"Cleaned up stream state for request {request_id}")

    def get_stream_debug_info(self, request_id: str) -> dict[str, Any] | None:
        """Get debug information for a request's stream state."""
        stream_state = self._stream_states.get(request_id)
        return stream_state.get_debug_info() if stream_state else None

    def _filter_tool_calls(
            self, tool_calls: list[dict[str, Any]], external_tools: set[str],
            should_filter_external_tools: bool) -> list[dict[str, Any]]:
        """Filter tool calls based on availability and suppression settings."""
        filtered = []

        for tool_call in tool_calls:
            func_name = tool_call.get("function", {}).get("name", "")

            # Filter unavailable external tools
            if should_filter_external_tools and func_name not in external_tools:
                logger.debug("Filtered unavailable tool call: %s", func_name)
                continue

            filtered.append(tool_call)

        return filtered

    def _is_tool_call_allowed(self, tool_call: dict[str, Any],
                              available_tools: set[str],
                              should_filter_external_tools: bool) -> bool:
        """Check if tool call is allowed based on availability and suppression settings."""
        function_name = tool_call.get("function", {}).get("name", "")
        # Built-in tools would be called occasionally
        if function_name == "python" or "browser" in function_name or "container" in function_name:
            return True

        # Filter unavailable external tools
        if should_filter_external_tools and function_name not in available_tools:
            return False

        return True


_SERVE_HARMONY_ADAPTER: HarmonyAdapter = None


def get_harmony_adapter():
    global _SERVE_HARMONY_ADAPTER
    if _SERVE_HARMONY_ADAPTER is None:
        _SERVE_HARMONY_ADAPTER = HarmonyAdapter()

    return _SERVE_HARMONY_ADAPTER


def handle_streaming_response(tools: List[ChatCompletionToolsParam],
                              tool_choice: str, result: GenerationResult,
                              model: str, request_id: str, done: bool,
                              num_prompt_tokens: int) -> List[str]:
    first_iteration = True
    output = result.outputs[0]

    # Convert tools to dictionary format for harmony adapter (standard pattern)
    tools_dict = None
    harmony_adapter = get_harmony_adapter()
    if tools:
        tools_dict = [tool.model_dump() for tool in tools]

    # Get tool_choice from request - if "none", don't pass tools to parser
    if tool_choice == "none":
        tools_for_parser = None
    else:
        tools_for_parser = tools_dict

    def end_streaming(res):
        # Clean up state
        harmony_adapter.cleanup_stream_state(request_id)

        # Append usage info
        usage_info = _create_usage_info(num_prompt_tokens, result.outputs)

        final_usage_chunk = ChatCompletionStreamResponse(choices=[],
                                                         model=model,
                                                         usage=usage_info)

        final_usage_json = final_usage_chunk.model_dump_json(exclude_none=True)

        res.append(f"data: {final_usage_json}\n\n")

    # Create OpenAI streaming responses
    try:
        res = []
        if done:
            # Send final message with finish_reason
            final_response = ChatCompletionStreamResponse(
                model=model,
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(),
                        finish_reason=output.finish_reason,
                        stop_reason=output.stop_reason)
                ],
            )

            final_response_json = final_response.model_dump_json(
                exclude_none=True)
            res.append(f"data: {final_response_json}\n\n")
            end_streaming(res)
        else:
            responses, should_stop = harmony_adapter.create_openai_streaming_response(
                request_id=request_id,
                tokens=output.token_ids_diff,
                available_tools=tools_for_parser,
                model_name=model,
                tool_choice=tool_choice)
            # Send first response after receiving the first output
            if first_iteration:
                first_iteration = False
                first_delta = DeltaMessage(role="assistant")

                choice = ChatCompletionResponseStreamChoice(index=0,
                                                            delta=first_delta)

                first_response = ChatCompletionStreamResponse(
                    model=model,
                    choices=[choice],
                )

                response_json = first_response.model_dump_json(
                    exclude_none=True)
                res.append(f"data: {response_json}\n\n")

            res.extend(responses)

            if should_stop:
                end_streaming(res)
                result.abort()

        return res

    except Exception as e:
        logger.error(f"Failed to create OpenAI streaming response: {e}")
        logger.debug(f"Streaming error details: {traceback.format_exc()}")
        # Clean up state
        harmony_adapter.cleanup_stream_state(request_id)
        raise e


def handle_non_streaming_response(tools: List[ChatCompletionToolsParam],
                                  tool_choice: str, outputs: List, model: str,
                                  num_prompt_tokens: int):
    """Handle non-streaming response with harmony format."""
    # Parse harmony output to OpenAI format
    # Convert tools to dictionary format for harmony adapter (standard pattern)
    tools_dict = None
    harmony_adapter = get_harmony_adapter()
    if tools:
        tools_dict = [tool.model_dump() for tool in tools]

    # Get tool_choice from request - if "none", don't pass tools to parser
    if tool_choice == "none":
        tools_for_parser = None
    else:
        tools_for_parser = tools_dict

    output = outputs[0]
    parsed_output = harmony_adapter.harmony_output_to_openai(
        output.token_ids, tools_for_parser, tool_choice)

    # CONVERTED OUTPUT (after harmony to openai conversion)
    logger.debug("✅ CONVERTED OUTPUT: %s", json.dumps(parsed_output, indent=2))

    # Create response message
    response_message = _create_response_message(parsed_output)

    # Determine finish reason
    finish_reason = _determine_finish_reason(parsed_output,
                                             output.finish_reason)

    # Create usage info from metrics (RequestOutput doesn't have usage in v1)
    usage_info = _create_usage_info(num_prompt_tokens, outputs)

    # Create response
    response = ChatCompletionResponse(
        model=model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(**response_message),
                finish_reason=finish_reason)
        ],
        usage=usage_info,
    )
    # Optional: Log if harmony parsing failed (for debugging)
    if parsed_output.get('_harmony_parsing_failed'):
        logger.warning("⚠️ Harmony parsing fell back to raw text decoding")
        logger.debug(f"response\n\n{response}\n")

    return response


def _create_response_message(parsed_output: dict[str, Any]) -> dict[str, Any]:
    """Create response message from parsed harmony output."""
    message = {
        "role": parsed_output.get("role", "assistant"),
        "content": parsed_output.get("content")
    }

    # Add tool_calls if present
    if "tool_calls" in parsed_output:
        message["tool_calls"] = parsed_output["tool_calls"]

    # Add reasoning_content if present
    if "reasoning" in parsed_output:
        message["reasoning"] = parsed_output["reasoning"]

    return message


def _determine_finish_reason(parsed_output: dict[str, Any],
                             reason: str | None) -> str | None:
    """Determine finish reason based on parsed output."""
    if "tool_calls" in parsed_output and parsed_output["tool_calls"]:
        return "tool_calls"
    else:
        return reason


def _create_usage_info(num_prompt_tokens, outputs) -> UsageInfo:
    """Create usage info from RequestOutput following serving_chat.py pattern."""
    # Calculate completion tokens from all outputs
    num_generated_tokens = sum(len(output.token_ids) for output in outputs)

    # Create usage info
    usage = UsageInfo(prompt_tokens=num_prompt_tokens,
                      completion_tokens=num_generated_tokens,
                      total_tokens=num_prompt_tokens + num_generated_tokens)
    return usage


def maybe_transform_reasoning_effort(
    reasoning_effort: ReasoningEffort | Literal["low", "medium", "high"] | None
) -> ReasoningEffort | None:
    str_to_effort = {
        "low": ReasoningEffort.LOW,
        "medium": ReasoningEffort.MEDIUM,
        "high": ReasoningEffort.HIGH
    }
    if reasoning_effort and not isinstance(reasoning_effort, ReasoningEffort):
        return str_to_effort[reasoning_effort]
    else:
        return reasoning_effort
