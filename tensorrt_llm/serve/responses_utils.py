# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import os
import time
import uuid
# yapf: disable
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Mapping
from copy import copy
from dataclasses import dataclass
from typing import (Any, Callable, List, Literal, Optional, OrderedDict, Tuple,
                    Union)

from openai.types.responses import (ResponseCompletedEvent,
                                    ResponseContentPartAddedEvent,
                                    ResponseContentPartDoneEvent,
                                    ResponseCreatedEvent,
                                    ResponseCustomToolCall, ResponseErrorEvent,
                                    ResponseFailedEvent,
                                    ResponseFunctionToolCall,
                                    ResponseIncompleteEvent,
                                    ResponseInProgressEvent, ResponseOutputItem,
                                    ResponseOutputItemAddedEvent,
                                    ResponseOutputItemDoneEvent,
                                    ResponseOutputMessage, ResponseOutputText,
                                    ResponseReasoningItem,
                                    ResponseReasoningTextDeltaEvent,
                                    ResponseReasoningTextDoneEvent,
                                    ResponseTextDeltaEvent,
                                    ResponseTextDoneEvent)
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_content_part_added_event import \
    PartReasoningText
from openai.types.responses.response_content_part_done_event import \
    Part as ResponseContentPart
from openai.types.responses.response_content_part_done_event import \
    PartReasoningText as PartReasoningTextDone
from openai.types.responses.response_function_web_search import (
    ActionFind, ActionOpenPage, ActionSearch, ResponseFunctionWebSearch)
from openai.types.responses.response_reasoning_item import Content
from openai.types.responses.tool import FunctionTool, Tool
from openai_harmony import (Author, Conversation, DeveloperContent,
                            HarmonyEncodingName, Message, ReasoningEffort, Role,
                            StreamState, SystemContent, TextContent,
                            ToolDescription, load_harmony_encoding)
from transformers import AutoProcessor, PretrainedConfig

from tensorrt_llm._utils import \
    get_steady_clock_now_in_seconds  # noqa: F401  (re-export)
from tensorrt_llm._utils import AdjustedSteadyClock
from tensorrt_llm.executor import (EngineDeadError, GenerationResult,
                                   RequestError)
from tensorrt_llm.inputs.utils import (async_apply_chat_template,
                                       resolve_hf_chat_template)
from tensorrt_llm.llmapi import SamplingParams
from tensorrt_llm.llmapi.disagg_utils import get_usage_tokens_from_ctx
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.llmapi.reasoning_parser import (BaseReasoningParser,
                                                  ReasoningParserFactory,
                                                  ReasoningParserResult)
from tensorrt_llm.llmapi.thinking_budget import \
    add_thinking_budget_logits_processor
from tensorrt_llm.llmapi.tokenizer import TokenizerBase, TransformersTokenizer
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.chat_utils import (parse_chat_messages_coroutines,
                                           resolve_top_level_model_type)
from tensorrt_llm.serve.openai_protocol import (ChatCompletionMessageParam,
                                                ChatCompletionToolsParam,
                                                FunctionDefinition,
                                                InputTokensDetails,
                                                OpenAIBaseModel,
                                                OutputTokensDetails,
                                                ReasoningAssistantMessage,
                                                ResponseInputOutputItem,
                                                ResponsesRequest,
                                                ResponsesResponse,
                                                ResponseUsage,
                                                StreamingResponsesResponse,
                                                UCompletionRequest,
                                                UCompletionResponse,
                                                to_disaggregated_params)
from tensorrt_llm.serve.responses_web_search import is_web_search_tool
from tensorrt_llm.serve.tool_parser.base_tool_parser import (
    BaseToolParser, warn_if_tool_call_unparsed)
from tensorrt_llm.serve.tool_parser.core_types import ToolCallItem
from tensorrt_llm.serve.tool_parser.tool_parser_factory import ToolParserFactory
from tensorrt_llm.serve.web_search import load_web_search_config
from tensorrt_llm.tokenizer.deepseek_v4 import DeepseekV4Tokenizer
from tensorrt_llm.tokenizer.deepseek_v32 import DeepseekV32Tokenizer

from .harmony_adapter import HarmonyAdapter, get_harmony_adapter

# yapf: enable

# yapf: enable

REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}

# Set TRTLLM_RESPONSES_DEBUG=1 to log each parsed input item and the full
# prompt handed to the model. Off by default: it prints whole
# conversations, so it is a debugging aid rather than something to leave
# enabled on a shared server.
ENABLE_RESPONSES_DEBUG_MSG = os.environ.get("TRTLLM_RESPONSES_DEBUG") == "1"

# The parameter a freeform custom tool is described with; see
# _get_chat_completion_function_tools and _tool_call_output_item.
CUSTOM_TOOL_INPUT_ARG = "input"


@dataclass
class StreamedItem:
    """One reasoning/message output item as the stream published it.

    ``text`` holds every delta emitted for the item, so the final snapshot can
    repeat exactly what the client received (see ``_create_output_content``).
    """
    item_type: str
    item_id: str
    text: str = ""


def _responses_debug_log(msg):
    if ENABLE_RESPONSES_DEBUG_MSG:
        logger.info(msg)


def _is_context_only(request: ResponsesRequest) -> bool:
    """Whether this request is the context half of a disaggregated split.

    Such a request comes from the orchestrator rather than a client, and its
    response is consumed by the orchestrator alone.
    """
    params = request.disaggregated_params
    return params is not None and params.request_type == "context_only"


_harmony_encoding = None


def _random_uuid():
    return str(uuid.uuid4().hex)


def _get_encoding():
    global _harmony_encoding
    if _harmony_encoding is None:
        _harmony_encoding = load_harmony_encoding(
            HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_encoding


def _decode_tokens(
    tokens: list[int],
    tokenizer: Optional[Union[TransformersTokenizer,
                              TokenizerBase]] = None) -> str:
    if tokenizer is not None:
        return tokenizer.decode(tokens)
    return _get_encoding().decode(tokens)


def _parse_response_input(
    input_msg: ResponseInputOutputItem,
    prev_responses: list[Union[ResponseOutputItem, ResponseReasoningItem]]
) -> Message:
    if not isinstance(input_msg, dict):
        input_msg = input_msg.model_dump()

    _responses_debug_log(f"------- Parsing input -----------")
    _responses_debug_log(input_msg)
    _responses_debug_log("")

    if "type" not in input_msg or input_msg["type"] == "message":
        role = input_msg["role"]
        content = input_msg["content"]
        if role == "system":
            # User is trying to set a system message. Change it to:
            # <|start|>developer<|message|># Instructions
            # {instructions}<|end|>
            role = "developer"
            text_prefix = "Instructions:\n"
        else:
            text_prefix = ""
        if isinstance(content, str):
            msg = Message.from_role_and_content(role, text_prefix + content)
        elif isinstance(content, list):
            contents = [
                TextContent(text=text_prefix + c["text"]) for c in content
            ]
            msg = Message.from_role_and_contents(role, contents)
        else:
            logger.warning("Responses API: Invalid input message type")
            msg = None
    elif input_msg["type"] == "function_call_output":
        call_id = input_msg["call_id"]
        call_response: Optional[ResponseFunctionToolCall] = None
        for prev_response in reversed(prev_responses):
            if isinstance(prev_response, ResponseFunctionToolCall
                          ) and prev_response.call_id == call_id:
                call_response = prev_response
                break
        if call_response is None:
            raise ValueError(f"No call message found for {call_id}")
        msg = Message.from_author_and_content(
            Author.new(Role.TOOL, f"functions.{call_response.name}"),
            input_msg["output"])
    elif input_msg["type"] == "reasoning":
        content = input_msg["content"]
        assert len(content) == 1
        msg = Message.from_role_and_content(Role.ASSISTANT, content[0]["text"])
    elif input_msg["type"] == "function_call":
        msg = Message.from_role_and_content(Role.ASSISTANT,
                                            input_msg["arguments"])
        msg = msg.with_channel("commentary")
        msg = msg.with_recipient(f"functions.{input_msg['name']}")
        msg = msg.with_content_type("json")
    else:
        raise ValueError(f"Unknown input type: {input_msg['type']}")
    return msg


class ConversationHistoryStore:

    def __init__(self, resp_capacity: int = 16, max_conversations=32):
        # How many responses can be stored.
        self.response_capacity = resp_capacity
        # How many messages can be stored in a conversation.
        self.conversation_capacity = resp_capacity * 4
        # How many conversations can be stored.
        self.max_conversations = max_conversations

        self.responses_lock = asyncio.Lock()
        # Responses store, responses stored more than response_capacity will be removed in LRU policy.
        self.responses: OrderedDict[str, ResponsesResponse] = OrderedDict()

        self.conversations_lock = asyncio.Lock()
        # Conversations store, conversations stored more than conversation_capacity will be removed in LRU policy.
        self.conversations: OrderedDict[str, Union[
            list[Message], list[ChatCompletionMessageParam]]] = OrderedDict()

        # Map from response id to conversation id. 1 to 1 mapping.
        self.response_to_conversation: dict[str, str] = {}

        # Map from conversation id to response id, which is the latest response in the conversation.
        self.conversation_to_response: dict[str, str] = {}

    async def load_response(self, resp_id: str) -> ResponsesResponse | None:
        _responses_debug_log(
            f"ConversationHistoryStore loading resp: {resp_id}")
        async with self.responses_lock:
            if resp_id not in self.responses:
                return None

            self.responses.move_to_end(resp_id)
            return self.responses.get(resp_id)

    async def store_response(self,
                             resp: ResponsesResponse,
                             resp_msgs: Optional[
                                 Union[list[Message],
                                       list[ChatCompletionMessageParam]]] = [],
                             prev_resp_id: Optional[str] = None) -> None:
        """Store a response and its model-output messages.

        If the previous response ID is provided, the messages are appended to
        that conversation. Otherwise, a new conversation is created.

        Args:
            resp: ResponsesResponse
            resp_msgs: Optional[Union[list[Message], list[ChatCompletionMessageParam]]]
            prev_resp_id: Optional[str]

        Returns:
            None
        """
        resp_id = resp.id
        _responses_debug_log(
            f"ConversationHistoryStore storing resp: {resp_id}")
        if ENABLE_RESPONSES_DEBUG_MSG:
            _responses_debug_log(f" -> resp_msgs:")
            for msg in resp_msgs:
                _responses_debug_log(f" -> {msg}")

        async with self.responses_lock:
            self.responses[resp_id] = resp
            if len(self.responses) > self.response_capacity:
                self._pop_response()

        async with self.conversations_lock:
            conversation_id: str
            if resp_id in self.response_to_conversation:
                conversation_id = self.response_to_conversation[resp_id]
                self.conversations[conversation_id].extend(resp_msgs)
            elif prev_resp_id is not None:
                if prev_resp_id not in self.response_to_conversation:
                    logger.warning(
                        f"Previous response id {prev_resp_id} not found in conversation store"
                    )

                conversation_id = self.response_to_conversation[prev_resp_id]
                self.conversations[conversation_id].extend(resp_msgs)
            else:
                conversation_id = _random_uuid()
                self.conversations[conversation_id] = resp_msgs

            _responses_debug_log(
                f" * storing at conversation id: {conversation_id}")

            self.response_to_conversation[resp_id] = conversation_id
            self.conversation_to_response[conversation_id] = resp_id
            self._trim_conversation(conversation_id)
            self._update_visited_conversation(conversation_id)

    async def pop_response(self, resp_id: Optional[str] = None) -> bool:
        async with self.responses_lock:
            return self._pop_response(resp_id)

    async def store_messages(self, resp_id: str,
                             msgs: Union[list[Message],
                                         list[ChatCompletionMessageParam]],
                             prev_resp_id: Optional[str]) -> None:
        """
        Store the messages in the conversation store.

        `msgs` should always contains the whole conversation messages, including the previous messages and the new messages.

        Args:
            resp_id: str
            msgs: Union[list[Message], list[ChatCompletionMessageParam]]: The messages to store.
            prev_resp_id: Optional[str]: The previous response id. If not provided, a new conversation will be created.

        Returns:
            None
        """
        _responses_debug_log(f"ConversationHistoryStore storing msg:")
        if ENABLE_RESPONSES_DEBUG_MSG:
            for msg in msgs:
                _responses_debug_log(f" -> {msg}")

        async with self.conversations_lock:
            conversation_id: str
            if prev_resp_id is not None and prev_resp_id in self.response_to_conversation:
                conversation_id = self.response_to_conversation[prev_resp_id]
            else:
                conversation_id = _random_uuid()

            _responses_debug_log(
                f" * storing at conversation: {conversation_id}")
            # A copy: trimming the stored conversation must not shorten the
            # caller's list, which is rendered next.
            self.conversations[conversation_id] = list(msgs)

            self.response_to_conversation[resp_id] = conversation_id
            self.conversation_to_response[conversation_id] = resp_id
            self._trim_conversation(conversation_id)
            self._update_visited_conversation(conversation_id)

    async def get_conversation_history(
            self, resp_id: str
    ) -> Union[list[Message], list[ChatCompletionMessageParam]]:
        _responses_debug_log(f"ConversationHistoryStore getting prev_msgs:")
        _responses_debug_log(f" -> prev_resp_id: {resp_id}")
        async with self.conversations_lock:
            if resp_id in self.response_to_conversation:
                conversation_id = self.response_to_conversation[resp_id]
                _responses_debug_log(
                    f" -> getting conversation_id: {conversation_id}")
                self._update_visited_conversation(conversation_id)
                return self.conversations.get(conversation_id, [])

            return []

    def _update_visited_conversation(self, conversation_id) -> None:
        """Move the visited conversation to the front of the store.

        This function is used to keep the conversation store sorted by the visited time.
        And also remove the least recently visited conversation if the number of conversations exceeds the limit.

        Args:
            conversation_id: str, the id of the conversation to update.

        Returns:
            None
        """
        if conversation_id not in self.conversations:
            return

        self.conversations.move_to_end(conversation_id)
        if len(self.conversations) > self.max_conversations:
            removed_id, _ = self.conversations.popitem(last=False)
            _responses_debug_log(
                f"ConversationHistoryStore Removing conversation {removed_id}")
            removed_resp_id = self.conversation_to_response[removed_id]
            # The responses may have been removed due to response capacity
            if removed_resp_id in self.response_to_conversation:
                self.response_to_conversation.pop(removed_resp_id)
            self.conversation_to_response.pop(removed_id)

    def _pop_conversation(self, resp_id) -> None:
        """Pop the oldest messages from a conversation.

        The conversation is starting by a user message and ending by an assistant message.
        This function is used to keep the number of messages in a conversation within the limit.

        Args:
            resp_id: str, the response id of the conversation to pop.

        Returns:
            None
        """
        conversation_id = self.response_to_conversation.get(resp_id, None)
        if conversation_id is None:
            return

        self._pop_conversation_by_conversation_id(conversation_id)

    def _trim_conversation(self, conversation_id: str) -> None:
        conversation = self.conversations.get(conversation_id)
        if conversation is None:
            return

        while len(conversation) > self.conversation_capacity:
            self._pop_conversation_by_conversation_id(conversation_id)

    def _pop_conversation_by_conversation_id(self,
                                             conversation_id: str) -> None:
        conversation = self.conversations.get(conversation_id)
        if conversation is None or len(conversation) == 0:
            return

        is_harmony_conversation = isinstance(conversation[0], Message)

        def get_first_conversation_range_harmony():
            start_index = 0
            end_index = 0
            for i, msg in enumerate(conversation):
                if msg.author.role == Role.USER:
                    start_index = i
                elif msg.channel == "final":
                    end_index = i
                    break

            return start_index, end_index

        def get_first_conversation_range():
            start_index = 0
            end_index = 0
            for i, msg in enumerate(conversation):
                if msg.get("role", "") == "user":
                    start_index = i
                elif msg.get("role", "") == "assistant":
                    end_index = i
                    break

            return start_index, end_index

        start_index, end_index = 0, 0
        if is_harmony_conversation:
            start_index, end_index = get_first_conversation_range_harmony()
        else:
            start_index, end_index = get_first_conversation_range()

        del conversation[start_index:end_index + 1]

    def _pop_response(self, resp_id: Optional[str] = None) -> bool:
        _responses_debug_log(f"pop response {resp_id}")

        if not self.responses:
            return False

        if resp_id is not None:
            if resp_id not in self.responses:
                return False
            self.responses.pop(resp_id)
        else:
            resp_id, _ = self.responses.popitem(last=False)

        if resp_id in self.response_to_conversation:
            self.response_to_conversation.pop(resp_id)

        return True


def _get_system_message(
    model_identity: Optional[str] = None,
    reasoning_effort: Optional[Literal["high", "medium", "low"]] = None,
    start_date: Optional[str] = None,
    browser_description: Optional[str] = None,
    python_description: Optional[str] = None,
) -> Message:
    sys_msg_content = SystemContent.new()
    if model_identity is not None:
        sys_msg_content = sys_msg_content.with_model_identity(model_identity)
    if reasoning_effort is not None:
        sys_msg_content = sys_msg_content.with_reasoning_effort(
            REASONING_EFFORT[reasoning_effort])
    if start_date:
        sys_msg_content = sys_msg_content.with_conversation_start_date(
            start_date)
    if browser_description is not None:
        sys_msg_content = sys_msg_content.with_tools(browser_description)
    if python_description is not None:
        sys_msg_content = sys_msg_content.with_tools(python_description)
    sys_msg = Message.from_role_and_content(Role.SYSTEM, sys_msg_content)
    return sys_msg


def _get_developer_message(instructions: Optional[str] = None,
                           tools: Optional[list[Tool]] = None) -> Message:
    dev_msg_content = DeveloperContent.new()
    if instructions is not None:
        dev_msg_content = dev_msg_content.with_instructions(instructions)
    if tools is not None:
        function_tools = []
        for tool in tools:
            if tool.type == "code_interpreter" or tool.type.startswith(
                    "web_search"):
                # Built-in tools, described in the system message if enabled.
                pass
            elif tool.type == "function":
                function_tools.append(tool)
            else:
                raise ValueError(f"tool type {tool.type} not supported")
        if function_tools:
            function_tool_descriptions = [
                ToolDescription.new(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                ) for tool in function_tools
            ]
            dev_msg_content = dev_msg_content.with_function_tools(
                function_tool_descriptions)
    dev_msg = Message.from_role_and_content(Role.DEVELOPER, dev_msg_content)
    return dev_msg


def _get_user_message(content: str) -> Message:
    return Message.from_role_and_content(Role.USER, content)


def _construct_harmony_messages(
    request: ResponsesRequest,
    prev_response: Optional[ResponsesResponse],
    prev_msgs: list[Message] = [],
) -> list[Message]:
    """Construct messages from request input, includes conversation history messages if exists."""
    messages: list[Message] = []
    if prev_response is None:
        # New conversation.
        reasoning_effort = (request.reasoning.effort
                            if request.reasoning else None)
        sys_msg = _get_system_message(reasoning_effort=reasoning_effort, )
        messages.append(sys_msg)
        dev_msg = _get_developer_message(request.instructions, request.tools)
        messages.append(dev_msg)
    else:
        messages.extend(prev_msgs)
    # Append the new input.
    # Responses API supports simple text inputs without chat format.
    if isinstance(request.input, str):
        messages.append(_get_user_message(request.input))
    else:
        if prev_response is not None:
            prev_outputs = copy(prev_response.output)
        else:
            prev_outputs = []
        for input_msg in request.input:
            msg = _parse_response_input(input_msg, prev_outputs)
            if msg is not None:
                messages.append(msg)
            # User passes in a a tool call request and its output. We need
            # to add the tool call request to prev_outputs so that the
            # parse_response_input can find the tool call request when
            # parsing the tool call output.
            if isinstance(input_msg, ResponseFunctionToolCall):
                prev_outputs.append(input_msg)
    return messages


def _render_for_completion(messages: list[Message]) -> list[int]:
    conversation = Conversation.from_messages(messages)
    if ENABLE_RESPONSES_DEBUG_MSG:
        _responses_debug_log("Rendering conversation:")
        _responses_debug_log(conversation.to_json())
    token_ids = _get_encoding().render_conversation_for_completion(
        conversation, Role.ASSISTANT)
    return token_ids


def _parse_output_tokens(tokens: list[int]) -> list[Message]:
    return _get_encoding().parse_messages_from_completion_tokens(
        tokens, role=Role.ASSISTANT)


def _parse_output_message_harmony(message: Message) -> list[ResponseOutputItem]:
    """
    Parse a Harmony message into a list of output response items.
    """
    if message.author.role != "assistant":
        # This is a message from a tool to the assistant (e.g., search result).
        # Don't include it in the final output for now. This aligns with
        # OpenAI's behavior on models like o4-mini.
        return []

    output_items: list[ResponseOutputItem] = []
    recipient = message.recipient
    if recipient is not None and recipient.startswith("browser."):
        if len(message.content) != 1:
            raise ValueError("Invalid number of contents in browser message")
        content = message.content[0]
        browser_call = json.loads(content.text)
        # TODO: translate to url properly!
        if recipient == "browser.search":
            action = ActionSearch(
                query=f"cursor:{browser_call.get('query', '')}", type="search")
        elif recipient == "browser.open":
            action = ActionOpenPage(url=f"cursor:{browser_call.get('url', '')}",
                                    type="open_page")
        elif recipient == "browser.find":
            action = ActionFind(pattern=browser_call["pattern"],
                                url=f"cursor:{browser_call.get('url', '')}",
                                type="find")
        else:
            raise ValueError(f"Unknown browser action: {recipient}")
        web_search_item = ResponseFunctionWebSearch(
            id=f"ws_{_random_uuid()}",
            action=action,
            status="completed",
            type="web_search_call",
        )
        output_items.append(web_search_item)
    elif message.channel == "analysis":
        for content in message.content:
            reasoning_item = ResponseReasoningItem(
                id=f"rs_{_random_uuid()}",
                summary=[],
                type="reasoning",
                content=[Content(text=content.text, type="reasoning_text")],
                status=None,
            )
            output_items.append(reasoning_item)
    elif message.channel == "commentary":
        if message.recipient is None:
            pass
        elif message.recipient.startswith("functions."):
            function_name = message.recipient.split(".")[-1]
            for content in message.content:
                response_item = ResponseFunctionToolCall(
                    arguments=content.text,
                    call_id=f"call_{_random_uuid()}",
                    type="function_call",
                    name=function_name,
                    id=f"fc_{_random_uuid()}",
                )
                output_items.append(response_item)
        elif message.recipient.startswith(
                "python") or message.recipient.startswith("browser"):
            for content in message.content:
                reasoning_item = ResponseReasoningItem(
                    id=f"rs_{_random_uuid()}",
                    summary=[],
                    type="reasoning",
                    content=[Content(text=content.text, type="reasoning_text")],
                    status=None,
                )
                output_items.append(reasoning_item)
        else:
            raise ValueError(f"Unknown recipient: {message.recipient}")
    elif message.channel == "final":
        contents = []
        for content in message.content:
            output_text = ResponseOutputText(
                text=content.text,
                annotations=[],  # TODO
                type="output_text",
                logprobs=None,  # TODO
            )
            contents.append(output_text)
        text_item = ResponseOutputMessage(
            id=f"msg_{_random_uuid()}",
            content=contents,
            role=message.author.role,
            status="completed",
            type="message",
        )
        output_items.append(text_item)
    else:
        raise ValueError(f"Unknown channel: {message.channel}")
    return output_items


def finish_reason_mapping(finish_reason: str) -> str:
    match finish_reason:
        case 'stop':
            return 'completed'
        case 'length':
            return 'incomplete'
        case 'timeout':
            return 'failed'
        case 'cancelled':
            return 'cancelled'
        case 'not_finished':
            # A disaggregated context worker handing the request off.
            return 'incomplete'

    raise RuntimeError(
        f"Unhandled finish reason {finish_reason!r} in finish_reason_mapping")


def _item_text(item: dict) -> str:
    """The text carried by an input item, whatever shape it uses."""
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if not text and part.get("type") == "encrypted_content":
                # Some clients carry readable text in an `encrypted_content`
                # part; only a string value is taken as text.
                value = part.get("encrypted_content")
                if isinstance(value, str):
                    text = value
            if text:
                parts.append(text)
        return "\n".join(parts)
    return item.get("text") or ""


_WARNED_ONCE: set[str] = set()


def _warn_once(message: str) -> None:
    """Emit a warning the first time only; later identical calls are dropped."""
    if message not in _WARNED_ONCE:
        _WARNED_ONCE.add(message)
        logger.warning(message)


def _qualified_tool_name(item: dict) -> str:
    """The name a replayed tool call is known by.

    A namespaced call is reported with its namespace in a separate field,
    but the model was offered the qualified name. Replaying the bare name
    shows it a tool that was never on its list, so it cannot match the call
    to the result that follows.
    """
    name = item.get("name") or ""
    namespace = item.get("namespace")
    return f"{namespace}.{name}" if namespace else name


# Responses text parts, which the chat-completions content parser knows as `text`.
_RESPONSES_TEXT_PART_TYPES = frozenset(("input_text", "output_text"))


def _chat_content_parts(content: list) -> list:
    """Rewrite Responses text parts as chat text parts; keep the others."""
    parts = []
    for part in content:
        part_type = part.get("type") if isinstance(part, dict) else getattr(
            part, "type", None)
        if part_type in _RESPONSES_TEXT_PART_TYPES:
            text = part.get("text") if isinstance(part, dict) else getattr(
                part, "text", None)
            parts.append({"type": "text", "text": text or ""})
        else:
            parts.append(part)
    return parts


def _tool_output_content(output):
    """A tool result's payload in the vocabulary the chat parser knows.

    ``output`` is a string or a list of Responses content parts.
    """
    if isinstance(output, list):
        return _chat_content_parts(output)
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    return str(output)


def _render_developer_as_system(
    messages: list[ChatCompletionMessageParam],
    tokenizer: Optional[TokenizerBase],
    processor: Optional[AutoProcessor],
    tools: Optional[list[dict[str, Any]]],
) -> list[ChatCompletionMessageParam]:
    """Render ``developer`` messages as ``system`` where the template lacks them.

    A chat template without a ``developer`` branch renders those messages to
    nothing. The DeepSeek tokenizers render ``developer`` themselves.
    """
    if not any(message.get("role") == "developer" for message in messages):
        return messages
    if isinstance(tokenizer, (DeepseekV32Tokenizer, DeepseekV4Tokenizer)):
        return messages
    template = resolve_hf_chat_template(getattr(tokenizer, "tokenizer",
                                                tokenizer),
                                        processor,
                                        chat_template=None,
                                        tools=tools)
    if not isinstance(template, str) or "developer" in template:
        return messages
    return [{
        **message, "role": "system"
    } if message.get("role") == "developer" else message
            for message in messages]


def _response_output_item_to_chat_completion_message(
    item: Union[dict, ResponseInputOutputItem]
) -> Optional[ChatCompletionMessageParam]:
    if not isinstance(item, dict):
        item = item.model_dump()

    item_type = item.get("type", "")

    match item_type:
        case "":
            if "role" not in item:
                raise ValueError(f"Invalid input message item: {item}")
            content = item.get("content")
            if isinstance(content, list):
                return {**item, "content": _chat_content_parts(content)}
            return item
        case "message" | "reasoning":
            content = item.get("content")
            if isinstance(content, str):
                content = [{"text": content}]
            elif item_type == "reasoning" and not content:
                # Reasoning may carry only a summary, or nothing readable.
                content = item.get("summary")
                if not content:
                    return None
            if not content:
                raise ValueError(
                    f"Input item of type {item_type!r} has empty or missing 'content'"
                )
            # Join every text part. Taking content[0] silently dropped the rest
            # of a multi-part message.
            parts = []
            for part in content:
                text = part.get("text") if isinstance(part, dict) else getattr(
                    part, "text", None)
                if text:
                    parts.append(text)
            text = "".join(parts)
            if item_type == "reasoning":
                # Reasoning is always the assistant's.
                return {"role": "assistant", "reasoning": text}
            # Honour the item's own role. Hardcoding "assistant" here turned
            # the caller's user turn into an assistant turn, so the model was
            # asked to continue its own message with no user message in the
            # prompt at all - which produces fabricated context and leaked
            # template markup rather than an answer. Clients that send
            # structured input items (Codex CLI, the OpenAI SDK) always set a
            # role; a plain string input never reaches this function.
            role = item.get("role") or "assistant"
            return {"role": role, "content": text}
        case "output_text":
            return {"role": "assistant", "content": item.get("text") or ""}
        case "function_call":
            # An assistant message carrying tool_calls, which is how the chat
            # completions path represents a call and what chat templates
            # expect. The deprecated role "function" is rejected outright by
            # some templates - DeepSeek-V4 answers "Unsupported message role:
            # function" - so a conversation dies on the turn *after* the model
            # first calls a tool.
            return {
                "role":
                "assistant",
                "content":
                None,
                "tool_calls": [{
                    "id": item.get("call_id") or item.get("id") or "",
                    "type": "function",
                    "function": {
                        "name": _qualified_tool_name(item),
                        "arguments": item.get("arguments") or "",
                    },
                }],
            }
        case "function_call_output":
            return {
                "role": "tool",
                "content": _tool_output_content(item["output"]),
                "tool_call_id": item["call_id"],
            }
        case "custom_tool_call":
            # The freeform counterpart of function_call. It is replayed as an
            # ordinary tool call, with the payload back under the parameter
            # the tool was described with, so the history the model sees
            # matches the calls it was asked to make. An unhandled item type
            # raises, which would end the conversation on the turn after the
            # model first used a custom tool.
            return {
                "role":
                "assistant",
                "content":
                None,
                "tool_calls": [{
                    "id": item.get("call_id") or item.get("id") or "",
                    "type": "function",
                    "function": {
                        "name":
                        _qualified_tool_name(item),
                        "arguments":
                        json.dumps(
                            {CUSTOM_TOOL_INPUT_ARG: item.get("input") or ""}),
                    },
                }],
            }
        case "custom_tool_call_output":
            # Read defensively: a client that omits either key should get a
            # turn that still renders, not a KeyError surfacing as a 500.
            return {
                "role": "tool",
                "content": _tool_output_content(item.get("output")),
                "tool_call_id": item.get("call_id") or "",
            }
        case "agent_message":
            # A message from another agent in a multi-agent session. It is
            # addressed to this agent, so it is replayed as input rather than
            # as something this agent said.
            return {
                "role": "user",
                "content": _item_text(item),
            }
        case _:
            # A client is free to carry its own item types, and refusing one
            # fails the whole request - which ends the conversation rather
            # than the turn. Anything with text is replayed as input so its
            # content is not silently lost; anything else is dropped with a
            # warning.
            text = _item_text(item)
            if text:
                logger.warning(
                    f"Responses API: replaying unrecognised input item type "
                    f"{item_type!r} as a plain message.")
                return {"role": item.get("role") or "user", "content": text}
            logger.warning(
                f"Responses API: skipping unrecognised input item type "
                f"{item_type!r} with no text content.")
            return None


def _fold_tool_calls_into_open_assistant_turn(
        messages: list[ChatCompletionMessageParam],
        message: ChatCompletionMessageParam, turn_start: int) -> bool:
    """Attach a converted tool-call item to the assistant message before it.

    The N calls of one assistant turn arrive as N ``function_call`` items. Kept
    in one assistant message, as chat completions does, they let the chat
    template bind each tool result to its call by id; templates that align
    results with the last assistant message only (GLM) lose that binding when
    the calls are split across messages.

    Only messages from ``turn_start`` on, converted from this request's input,
    are extended; a tool result or user message in between ends the turn.
    """
    if message.get("role") != "assistant" or not message.get("tool_calls"):
        return False
    if message.get("content") is not None:
        # A message with content of its own is a turn of its own.
        return False
    if len(messages) <= turn_start:
        return False
    last = messages[-1]
    if last.get("role") != "assistant":
        return False
    # Rebuilt: the target's list may be None or owned by the caller.
    last["tool_calls"] = [
        *(last.get("tool_calls") or []), *message["tool_calls"]
    ]
    return True


async def _create_input_messages(
    request: ResponsesRequest,
    prev_msgs: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    return chat_messages_from_responses_input(request, prev_msgs)


def chat_messages_from_responses_input(
    request: ResponsesRequest,
    prev_msgs: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    """Convert a Responses request's instructions, history and input to chat messages."""
    messages: list[ChatCompletionMessageParam] = []
    if request.instructions:
        messages.append({
            "role": "system",
            "content": request.instructions,
        })

    # Prepend the conversation history.
    # Skip the reasoning output, but keep the tool calls a reasoning message
    # carries: the tool results that follow answer them.
    for msg in prev_msgs:
        if "reasoning" not in msg:
            messages.append(msg)
            continue
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            })

    # Append the new input.
    # Responses API supports simple text inputs without chat format.
    if isinstance(request.input, str):
        messages.append({"role": "user", "content": request.input})
    else:
        turn_start = len(messages)
        for inp in request.input:
            message = _response_output_item_to_chat_completion_message(inp)
            if message is None:
                continue
            if _fold_tool_calls_into_open_assistant_turn(
                    messages, message, turn_start):
                continue
            messages.append(message)

    return messages


def _stored_tool_arguments(call) -> str:
    """The arguments to record in conversation history for a tool call.

    A custom tool call carries its payload as freeform text in `input`, not as
    JSON in `arguments`, so reading `arguments` unconditionally raises and
    fails the whole request. It only surfaced with a model that emits
    reasoning, because history is only built on that path.

    The payload goes back under the parameter the tool was described with, so
    what is stored matches what the model was asked to produce.
    """
    arguments = getattr(call, "arguments", None)
    if arguments is not None:
        return arguments
    return json.dumps({CUSTOM_TOOL_INPUT_ARG: getattr(call, "input", "") or ""})


def _stored_tool_name(call) -> str:
    """The name to record, qualified again for a namespaced tool."""
    name = getattr(call, "name", "") or ""
    namespace = getattr(call, "namespace", None)
    return f"{namespace}.{name}" if namespace else name


def _create_output_messages(
        output_contents: dict[str, Any]) -> list[ChatCompletionMessageParam]:
    """
    Convert output contents to chat completion messages for conversation store.

    Reasoning is stored and stripped on replay (_create_input_messages). Tool
    calls go on the reasoning message, else the text message, else a bare
    assistant message.

    Input:
        output_contents: dict[str, str]
        - text_content: Optional[str]
        - reasoning_content: Optional[str]
        - tool_calls: Optional[list[ToolCall]]

    Returns:
        list[ChatCompletionMessageParam]: Chat completion messages for conversation store.
    """
    messages: list[ChatCompletionMessageParam] = []

    tool_calls = output_contents.get("tool_calls") or []
    tool_call_msgs = [{
        "id": call.call_id,
        "function": {
            "arguments": _stored_tool_arguments(call),
            "name": _stored_tool_name(call),
        },
        "type": "function",
    } for call in tool_calls]
    _responses_debug_log(f"tool_call_msgs: {tool_call_msgs}")

    text_content = output_contents.get("text_content", None)
    text_msg: Optional[ChatCompletionMessageParam] = None
    if text_content:
        text_msg = {
            "role": "assistant",
            "content": text_content,
        }
        messages.append(text_msg)

    reasoning_content = output_contents.get("reasoning_content", None)
    if reasoning_content:
        reasoning_msg = ReasoningAssistantMessage(
            role="assistant",
            reasoning=reasoning_content,
        )
        reasoning_msg["tool_calls"] = tool_call_msgs
        messages.append(reasoning_msg)
    elif tool_call_msgs:
        if text_msg is not None:
            text_msg["tool_calls"] = tool_call_msgs
        else:
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": tool_call_msgs,
            })

    return messages


def _get_chat_completion_function_tools(
        tools: Optional[list[Tool]]) -> list[ChatCompletionToolsParam]:
    function_tools: list[ChatCompletionToolsParam] = []
    if tools is None:
        return function_tools

    def as_function(name: str, description: Optional[str],
                    parameters: Optional[Any]) -> ChatCompletionToolsParam:
        return ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name=name,
                description=description,
                # A tool with no schema still has to present an object schema,
                # or the chat template renders a call the model cannot fill in.
                parameters=parameters if parameters is not None else {
                    "type": "object",
                    "properties": {},
                },
            ),
        )

    def custom_parameters() -> dict[str, Any]:
        # A custom tool takes one freeform string rather than JSON arguments -
        # apply_patch is the common case, whose payload is a patch, not an
        # object. The chat template can only describe functions, so it is
        # described as a single named string parameter and the call is turned
        # back into a custom tool call on the way out; see
        # _tool_call_output_item. Without the named parameter the model invents
        # its own argument name and the client rejects the call as an
        # incompatible payload.
        return {
            "type": "object",
            "properties": {
                CUSTOM_TOOL_INPUT_ARG: {
                    "type":
                    "string",
                    "description":
                    "The complete freeform input for this tool, "
                    "passed through verbatim.",
                },
            },
            "required": [CUSTOM_TOOL_INPUT_ARG],
        }

    for tool in tools:
        tool_type = getattr(tool, "type", None)
        if isinstance(tool, FunctionTool):
            function_tools.append(
                as_function(tool.name, tool.description, tool.parameters))
        elif tool_type == "namespace":
            # A namespace groups several function/custom tools under one name.
            # Skipping it drops every tool inside, which is most of an agentic
            # client's toolset: the model then has nothing to call, announces
            # an action and does nothing. Names are qualified with the
            # namespace so two namespaces can define the same tool name.
            for inner in getattr(tool, "tools", None) or []:
                inner_name = getattr(inner, "name", None)
                if not inner_name:
                    continue
                # A custom tool nested in a namespace needs the same freeform
                # schema as a top-level one. It carries no `parameters`, so
                # passing them straight through would describe it with an empty
                # object schema - while _tool_resolution still classifies the
                # tool as custom, so the output path goes looking for
                # CUSTOM_TOOL_INPUT_ARG the prompt never mentioned.
                if getattr(inner, "type", None) == "custom":
                    inner_parameters = custom_parameters()
                else:
                    inner_parameters = getattr(inner, "parameters", None)
                function_tools.append(
                    as_function(
                        f"{tool.name}.{inner_name}",
                        getattr(inner, "description", None) or tool.description,
                        inner_parameters,
                    ))
        elif tool_type in ("custom", ):
            function_tools.append(
                as_function(tool.name, getattr(tool, "description", None),
                            custom_parameters()))
        elif is_web_search_tool(tool):
            # Web search is a *server* tool: the client sends the definition
            # and expects the server to run the query and feed the results
            # back within the same response. The search itself is endpoint-
            # neutral and lives in tensorrt_llm/serve/web_search.py.
            #
            # It is described to the model as an ordinary function because a
            # chat template cannot describe anything else, and the call is
            # intercepted server-side rather than returned to the client.
            #
            # A request that reaches here still carrying web_search should
            # already have been refused by the endpoint - see
            # web_search_rejection_reason and its caller in openai_server.py -
            # because answering without the search the client asked for is a
            # wrong answer the client cannot detect. This branch is the
            # fallback for any other caller of this function: drop the tool
            # rather than describe it, since nothing on this path executes the
            # call yet and the model would emit one the client has no
            # implementation for ("unsupported call: web_search").
            #
            # Warn once per process rather than per request: what it reports is
            # a server-configuration fact, not a property of the request, so
            # repeating it per call only buries the rest of the log.
            if load_web_search_config().enabled:
                _warn_once(
                    "Responses web_search is configured but not yet executed "
                    "on this path; dropping it.")
            else:
                _warn_once(
                    "Responses web_search was requested but no provider is "
                    "configured; dropping it.")
        else:
            logger.warning(
                f"Unsupported tool type: {type(tool)} for non-gpt-oss models, skipping."
            )

    return function_tools


async def _create_input_tokens(
    request: ResponsesRequest,
    prev_response: Optional[ResponsesResponse],
    prev_msgs: list[ChatCompletionMessageParam],
    conversation_store: ConversationHistoryStore,
    enable_store: bool,
    tokenizer: Union[TransformersTokenizer, TokenizerBase],
    model_config: PretrainedConfig,
    processor: AutoProcessor,
) -> Tuple[list[int], Optional[dict[str, list[Any]]]]:
    """
    Create input tokens for the model. Also return the mm data if the model is multimodal.

    Returns:
        Tuple[list[int], Optional[dict[str, list[Any]]]]: Input tokens and mm data.

    """
    messages = await _create_input_messages(
        request=request,
        prev_msgs=prev_msgs,
    )

    if enable_store and request.store:
        await conversation_store.store_messages(request.request_id, messages,
                                                request.previous_response_id)

    tools_dict = [
        tool.model_dump()
        for tool in _get_chat_completion_function_tools(request.tools)
    ]
    messages = _render_developer_as_system(messages, tokenizer, processor,
                                           tools_dict)
    conversation, mm_coroutines, mm_placeholder_counts, _ = parse_chat_messages_coroutines(
        messages, model_config)
    # Carry the request's reasoning configuration into the chat template.
    #
    # Chat templates that support thinking are opt-in: DeepSeek-V4's custom
    # tokenizer only emits the thinking prompt when it is handed
    # thinking=True, and picks the reasoning-effort prefix from
    # reasoning_effort. The chat completions path forwards the caller's
    # chat_template_kwargs, but this path forwarded nothing, so a Responses
    # client asking for reasoning.effort="high" silently got the default
    # non-thinking prompt.
    #
    # It matters beyond effort level: when thinking is not enabled the model
    # can still emit a stray closing think tag, and the reasoning parser -
    # which expects the thinking-mode framing - leaves it in the visible text.
    chat_template_kwargs = reasoning_chat_template_kwargs(request)

    token_task = async_apply_chat_template(
        model_type=resolve_top_level_model_type(model_config),
        tokenizer=tokenizer,
        processor=processor,
        conversation=conversation,
        add_generation_prompt=True,
        tools=tools_dict,
        mm_placeholder_counts=mm_placeholder_counts,
        chat_template_kwargs=chat_template_kwargs or None,
        enable_tokenize=True,
        injected_chat_template_kwargs=reasoning_injected_chat_template_keys(
            request),
    )
    token_ids, (mm_data,
                _mm_embeddings) = await asyncio.gather(token_task,
                                                       mm_coroutines)

    return token_ids, mm_data


async def _create_input_tokens_harmony(
    request: ResponsesRequest,
    prev_response: Optional[ResponsesResponse],
    prev_msgs: list[Message],
    conversation_store: ConversationHistoryStore,
    enable_store: bool,
) -> list[int]:
    messages = _construct_harmony_messages(request,
                                           prev_response,
                                           prev_msgs=prev_msgs)

    if enable_store and request.store:
        # Remove reasoning messages to save token usage during multi-turn conversation
        msgs_to_store = [msg for msg in messages if msg.channel != "analysis"]
        await conversation_store.store_messages(request.request_id,
                                                msgs_to_store,
                                                request.previous_response_id)

    return _render_for_completion(messages)


async def request_preprocess(
    request: ResponsesRequest,
    prev_response: Optional[ResponsesResponse],
    conversation_store: ConversationHistoryStore,
    enable_store: bool,
    use_harmony: bool,
    tokenizer: Optional[Union[TransformersTokenizer, TokenizerBase]] = None,
    model_config: Optional[PretrainedConfig] = None,
    processor: Optional[AutoProcessor] = None,
    reasoning_parser: Optional[str] = None,
) -> tuple[list[int], SamplingParams]:

    sampling_params = request.to_sampling_params(
        default_sampling_params={
            "stop_token_ids":
            get_harmony_adapter().get_stop_tokens() if use_harmony else []
        },
        reasoning_parser=reasoning_parser,
    )

    prev_response_id = request.previous_response_id

    # TODO: better way to enable metrics
    if len(os.getenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", "")) > 0:
        sampling_params.return_perf_metrics = True

    prev_msgs = []
    if enable_store and prev_response_id is not None:
        prev_msgs = await conversation_store.get_conversation_history(
            prev_response_id)

        _responses_debug_log(f"Prev msgs:")
        for msg in prev_msgs:
            _responses_debug_log(f" -> {msg}")

    # A generation worker in disaggregated serving gets the context worker's
    # tokens: rendering again could produce a prompt that was never prefilled.
    pretokenized = request.relayed_prompt_token_ids()
    if pretokenized is not None:
        input_tokens = pretokenized
    elif use_harmony:
        input_tokens = await _create_input_tokens_harmony(
            request=request,
            prev_response=prev_response,
            prev_msgs=prev_msgs,
            conversation_store=conversation_store,
            enable_store=enable_store,
        )

    else:
        input_tokens, _ = await _create_input_tokens(
            request=request,
            prev_response=prev_response,
            prev_msgs=prev_msgs,
            conversation_store=conversation_store,
            enable_store=enable_store,
            tokenizer=tokenizer,
            model_config=model_config,
            processor=processor,
        )

    if ENABLE_RESPONSES_DEBUG_MSG:  # decoding is not free; skip it otherwise
        _responses_debug_log("======= Complete Inputs to model =======")
        _responses_debug_log(_decode_tokens(input_tokens, tokenizer))
        _responses_debug_log("========================================")
    add_thinking_budget_logits_processor(
        sampling_params,
        reasoning_parser=reasoning_parser,
        tokenizer=tokenizer,
    )
    return input_tokens, sampling_params


# TODO(JunyiXu-nv): move to use the same function in postprocess_handlers after multiple post processors are supported
def reasoning_chat_template_kwargs(request) -> dict:
    """Chat-template kwargs implied by a request's reasoning configuration.

    Both the prompt and the reasoning parser need these. DeepSeek-V4's
    tokenizer only emits the thinking prompt when it sees thinking=True, and
    DeepSeekV4ReasoningParser only splits reasoning out of the text when it is
    constructed with the same flag - otherwise it falls back to an identity
    parser and the reasoning, plus its closing tag, stays in the visible
    answer. Deriving both from one place keeps them from drifting apart.
    """
    # Check the type rather than the truthiness of each attribute. A caller
    # that passes a stand-in object - a Mock in a unit test, say - hands back a
    # truthy attribute for any name, so `dict(attr or {})` would call
    # `attr.keys()` and fail with a TypeError instead of falling back.
    raw = getattr(request, "chat_template_kwargs", None)
    kwargs = dict(raw) if isinstance(raw, Mapping) else {}
    reasoning = getattr(request, "reasoning", None)
    effort = getattr(reasoning, "effort",
                     None) if reasoning is not None else None
    if isinstance(effort, str) and effort:
        kwargs.setdefault("reasoning_effort", effort)
        kwargs.setdefault("thinking", True)
    return kwargs


def reasoning_injected_chat_template_keys(request) -> frozenset[str]:
    """Keys `reasoning_chat_template_kwargs` added on the caller's behalf.

    The unused-kwargs guard rejects caller controls the template never reads,
    but a Responses client only asked for `reasoning.effort`; it did not pick
    `reasoning_effort` / `thinking` and cannot drop them when a template (a
    Qwen3 one, say) reads neither. Keys the caller passed explicitly in
    `chat_template_kwargs` stay subject to the guard.
    """
    raw = getattr(request, "chat_template_kwargs", None)
    caller_keys = set(raw) if isinstance(raw, Mapping) else set()
    return frozenset(set(reasoning_chat_template_kwargs(request)) - caller_keys)


def _apply_reasoning_parser(
    reasoning_parser_id: Optional[str],
    output_index: int,
    text: str,
    streaming: bool,
    reasoning_parser_dict: Optional[dict[int, BaseReasoningParser]] = None,
    finished: bool = False,
    chat_template_kwargs: Optional[dict[str, Any]] = None,
) -> Tuple[str, str]:
    reasoning_parser: Optional[BaseReasoningParser] = None
    if reasoning_parser_id is not None:
        if reasoning_parser_dict is not None:
            if output_index not in reasoning_parser_dict:
                reasoning_parser_dict[
                    output_index] = ReasoningParserFactory.create_reasoning_parser(
                        reasoning_parser_id, chat_template_kwargs)

            reasoning_parser = reasoning_parser_dict[output_index]
        else:
            reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
                reasoning_parser_id, chat_template_kwargs)

    if reasoning_parser is not None:
        if not streaming:
            result = reasoning_parser.parse(text)
        else:
            result = reasoning_parser.parse_delta(text)
            if finished:
                finish_result = reasoning_parser.finish()
                result = ReasoningParserResult(
                    content=result.content + finish_result.content,
                    reasoning_content=result.reasoning_content +
                    finish_result.reasoning_content,
                )
        content, reasoning_content = result.content, result.reasoning_content
    else:
        content, reasoning_content = text, ""

    return content, reasoning_content


def _whole_text_tool_calls(
    output: RequestOutput,
    request: ResponsesRequest,
    helper: "ResponsesStreamingEventsHelper",
    reasoning_parser_id: Optional[str],
    tool_parser_id: str,
    tools: list[ChatCompletionToolsParam],
) -> Tuple[str, list[ToolCallItem]]:
    """Tool calls from one whole-text parse at end of stream.

    Also returns the normal text the stream still owes: what that parse reads
    as normal text beyond the message text already streamed. Markup the
    incremental parser withheld is never released this way.
    """
    content, _ = _apply_reasoning_parser(
        reasoning_parser_id,
        output.index,
        output.text,
        streaming=False,
        chat_template_kwargs=reasoning_chat_template_kwargs(request))
    normal_text, calls = _apply_tool_parser(tool_parser_id, tools, output.index,
                                            content, False)
    streamed = "".join(item.text for item in helper.emitted_item_ids
                       if item.item_type == "message")
    owed = normal_text[len(streamed):] if normal_text.startswith(
        streamed) else ""
    return owed, calls


def _effective_tool_parser(tool_parser_id: Optional[str],
                           request: ResponsesRequest) -> Optional[str]:
    """The tool parser to run: none for ``tool_choice="none"``.

    Not parsing keeps any call markup in the visible text verbatim, in the
    stream and the final snapshot alike.
    """
    if request.tool_choice == "none":
        return None
    return tool_parser_id


def _apply_tool_parser(
    tool_parser_id: Optional[str],
    tools: Optional[list[Tool]],
    output_index: int,
    text: str,
    streaming: bool,
    tool_parser_dict: Optional[dict[int, BaseToolParser]] = None,
) -> Tuple[str, list[ToolCallItem]]:
    tool_parser: Optional[BaseToolParser] = None
    if tool_parser_id is not None and tools is not None:
        if tool_parser_dict is not None:
            if output_index not in tool_parser_dict:
                tool_parser_dict[
                    output_index] = ToolParserFactory.create_tool_parser(
                        tool_parser_id)

            tool_parser = tool_parser_dict[output_index]
        else:
            tool_parser = ToolParserFactory.create_tool_parser(tool_parser_id)

    if tool_parser is not None and tools is not None:
        if not streaming:
            result = tool_parser.detect_and_parse(text, tools)
        else:
            result = tool_parser.parse_streaming_increment(text, tools)
        normal_text, calls = result.normal_text, result.calls
        if not streaming:
            warn_if_tool_call_unparsed(tool_parser_id, tool_parser, text, calls)
    else:
        normal_text, calls = text, []

    return normal_text, calls


def _streamed_items_cover(streamed_items: list[StreamedItem],
                          reasoning_text: Optional[str],
                          text: Optional[str]) -> bool:
    """Whether the streamed items hold the same characters as the re-parse.

    Item boundaries and edge whitespace may differ (the whole-text parse can
    strip what the stream already sent); anything else means the two views
    diverged and the re-parsed items are used instead.
    """
    streamed_reasoning = "".join(item.text for item in streamed_items
                                 if item.item_type == "reasoning")
    streamed_message = "".join(item.text for item in streamed_items
                               if item.item_type == "message")
    return (streamed_reasoning.strip() == (reasoning_text or "").strip()
            and streamed_message.strip() == (text or "").strip())


def _create_output_content(
    final_res: RequestOutput,
    reasoning_parser: Optional[str] = None,
    tool_parser: Optional[str] = None,
    tools: Optional[list[Tool]] = None,
    chat_template_kwargs: Optional[dict[str, Any]] = None,
    streamed_tool_calls: Optional[list[ResponseOutputItem]] = None,
    streamed_item_ids: Optional[list[StreamedItem]] = None,
) -> Tuple[list[ResponseOutputItem], list[ChatCompletionMessageParam],
           list[str]]:
    """Build the output items for a finished generation.

    For a streamed request (``streamed_item_ids``/``streamed_tool_calls`` not
    None) the snapshot repeats what the stream published: its reasoning and
    message items when they cover the re-parsed text, and its tool-call items
    as emitted. Otherwise items are derived from a whole-text parse, reusing
    streamed ids positionally where a stream ran.
    """
    output_items: list[ResponseOutputItem] = []
    output_messages: list[ChatCompletionMessageParam] = []
    # Raw reasoning per output, for counting reasoning tokens.
    reasoning_texts: list[str] = []
    available_tools = _get_chat_completion_function_tools(tools)

    # A stream only runs over a single output.
    single_output = len(final_res.outputs) == 1
    streamed_ids_by_type: dict[str, list[str]] = {
        "reasoning": [],
        "message": []
    }
    for record in streamed_item_ids or []:
        if record.item_type in streamed_ids_by_type:
            streamed_ids_by_type[record.item_type].append(record.item_id)
    used_ids_by_type = {"reasoning": 0, "message": 0}
    used_streamed_assembly = False

    def _streamed_or_fresh_id(item_type: str) -> str:
        pool = streamed_ids_by_type[item_type]
        index = used_ids_by_type[item_type]
        used_ids_by_type[item_type] = index + 1
        if index < len(pool):
            return pool[index]
        prefix = "rs" if item_type == "reasoning" else "msg"
        return f"{prefix}_{_random_uuid()}"

    for output in final_res.outputs:
        calls = []
        # chat_template_kwargs has to reach the parser: DeepSeekV4ReasoningParser
        # only splits reasoning out of the text when it is constructed with the
        # same thinking flag the prompt was rendered with. Without it the
        # factory hands back an identity parser and the reasoning, plus its
        # closing tag, stays in the visible answer.
        text, reasoning_text = _apply_reasoning_parser(
            reasoning_parser,
            output.index,
            output.text,
            False,
            chat_template_kwargs=chat_template_kwargs)
        reasoning_texts.append(reasoning_text or "")

        if text:
            text, calls = _apply_tool_parser(tool_parser, available_tools,
                                             output.index, text, False)

        stored_text: Optional[str] = None
        stored_reasoning: Optional[str] = None
        # Reasoning first, then the answer, then any tool calls: the stream's
        # order.
        if (streamed_item_ids is not None
                and single_output and _streamed_items_cover(
                    streamed_item_ids, reasoning_text, text)):
            used_streamed_assembly = True
            for record in streamed_item_ids:
                if record.item_type == "reasoning":
                    output_items.append(
                        ResponseReasoningItem(
                            id=record.item_id,
                            summary=[],
                            type="reasoning",
                            content=[
                                Content(text=record.text, type="reasoning_text")
                            ],
                            status=None,
                        ))
                else:
                    output_items.append(
                        ResponseOutputMessage(
                            id=record.item_id,
                            content=[
                                ResponseOutputText(
                                    text=record.text,
                                    annotations=[],
                                    type="output_text",
                                    logprobs=None,
                                )
                            ],
                            role="assistant",
                            status="completed",
                            type="message",
                        ))
            stored_reasoning = "".join(
                record.text for record in streamed_item_ids
                if record.item_type == "reasoning") or None
            stored_text = "".join(record.text for record in streamed_item_ids
                                  if record.item_type == "message") or None
        else:
            if reasoning_text:
                output_items.append(
                    ResponseReasoningItem(
                        id=_streamed_or_fresh_id("reasoning"),
                        summary=[],
                        type="reasoning",
                        content=[
                            Content(text=reasoning_text, type="reasoning_text")
                        ],
                        status=None,
                    ))
                stored_reasoning = reasoning_text

            # Check again after tool parsing to avoid empty text
            if text:
                output_items.append(
                    ResponseOutputMessage(
                        id=_streamed_or_fresh_id("message"),
                        content=[
                            ResponseOutputText(
                                text=text,
                                annotations=[],
                                type="output_text",
                                logprobs=None,
                            )
                        ],
                        role="assistant",
                        status="completed",
                        type="message",
                    ))
                stored_text = text

        if streamed_tool_calls is not None and single_output:
            tool_calls_item = list(streamed_tool_calls)
        else:
            tool_resolution = _tool_resolution(tools)
            tool_calls_item = [
                _tool_call_output_item(call, tool_resolution) for call in calls
            ]
        output_items.extend(tool_calls_item)

        output_messages.extend(
            _create_output_messages({
                "text_content": stored_text,
                "reasoning_content": stored_reasoning,
                "tool_calls": tool_calls_item,
            }))

    if streamed_item_ids is not None and not used_streamed_assembly:
        for item_type in ("reasoning", "message"):
            streamed = len(streamed_ids_by_type[item_type])
            rebuilt = used_ids_by_type[item_type]
            if streamed != rebuilt:
                logger.warning(
                    f"final response rebuilt {rebuilt} {item_type} item(s) but "
                    f"the stream published {streamed}; ids beyond the "
                    f"streamed ones are new")

    return output_items, output_messages, reasoning_texts


def _create_output_content_harmony(
        final_res: RequestOutput
) -> Tuple[list[ResponseOutputItem], list[Message]]:
    output_messages = _parse_output_tokens(final_res.outputs[0].token_ids)
    output_content = []

    if ENABLE_RESPONSES_DEBUG_MSG:
        _responses_debug_log(f"output messages: {len(output_messages)}")
        for msg in output_messages:
            _responses_debug_log(f" -> {msg.to_json()}")

    for msg in output_messages:
        output_content.extend(_parse_output_message_harmony(msg))

    return output_content, output_messages


def _tool_resolution(
        tools: Optional[list[Tool]]
) -> dict[str, Tuple[Optional[str], str, bool]]:
    """Every spelling a parsed call may carry -> (namespace, bare name, custom).

    A chat template can only describe a flat list of functions, so a namespaced
    tool is offered to the model as "namespace.tool" and has to be reported back
    with the two parts separated again - that is how the client identifies it.
    Models write the name back both ways, so the bare name resolves too, but
    only when exactly one declared tool answers to it.
    """
    resolved: dict[str, Tuple[Optional[str], str, bool]] = {}
    bare_claims: dict[str, list[str]] = {}

    def claim(exposed: str, namespace: Optional[str], bare: str,
              is_custom: bool) -> None:
        resolved[exposed] = (namespace, bare, is_custom)
        if exposed != bare:
            bare_claims.setdefault(bare, []).append(exposed)

    for tool in tools or []:
        tool_type = getattr(tool, "type", None)
        name = getattr(tool, "name", None)
        if not name:
            continue
        if tool_type == "namespace":
            for inner in getattr(tool, "tools", None) or []:
                inner_name = getattr(inner, "name", None)
                if inner_name:
                    claim(f"{name}.{inner_name}", name, inner_name,
                          getattr(inner, "type", None) == "custom")
        else:
            claim(name, None, name, tool_type == "custom")

    for bare, exposed_names in bare_claims.items():
        # A top-level tool owning the spelling wins; it is what the model was
        # shown under that exact name.
        if bare in resolved:
            continue
        if len(exposed_names) == 1:
            resolved[bare] = resolved[exposed_names[0]]
    return resolved


def _tool_call_output_item(
    call,
    tool_resolution: dict[str, Tuple[Optional[str], str, bool]],
    item_id: Optional[str] = None,
    status: Optional[str] = None,
) -> Union[ResponseFunctionToolCall, ResponseCustomToolCall]:
    """Build the output item for one parsed tool call.

    A custom tool is invoked with freeform text, so its call has to be
    reported as a custom tool call carrying that text. Reporting it as a
    function call hands the client JSON where it expects the raw payload,
    and the client rejects the call outright - for apply_patch, with
    "invoked with incompatible payload", which aborts the whole turn.
    """
    name = call.name or ""
    arguments = call.parameters or "{}"
    call_id = f"call_{_random_uuid()}"

    resolved = (tool_resolution or {}).get(name)
    if resolved is not None:
        namespace, name, is_custom = resolved
    else:
        namespace, is_custom = None, False
        logger.warning(
            f"tool call {name!r} matches no declared tool; reporting it as a "
            f"function call. If it is in fact a custom tool, the client will "
            f"reject it.")

    if is_custom:
        # Unwrap the single string argument the tool was described with. A
        # model that answered with something else still gets its payload
        # forwarded verbatim, which is closer to the intent than dropping it.
        text = arguments
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            if CUSTOM_TOOL_INPUT_ARG in parsed:
                text = parsed[CUSTOM_TOOL_INPUT_ARG]
            elif len(parsed) == 1:
                text = next(iter(parsed.values()))
        if not isinstance(text, str):
            text = json.dumps(text)

        return ResponseCustomToolCall(
            call_id=call_id,
            input=text,
            name=name,
            type="custom_tool_call",
            id=item_id or f"ctc_{_random_uuid()}",
            namespace=namespace,
        )

    item = ResponseFunctionToolCall(
        arguments=arguments,
        call_id=call_id,
        name=name,
        type="function_call",
        id=item_id or f"fc_{_random_uuid()}",
        namespace=namespace,
    )
    if status is not None:
        item.status = status
    return item


def _count_reasoning_tokens(
    tokenizer: Optional[TokenizerBase],
    reasoning_texts: list[str],
    output_tokens: int,
) -> int:
    """How many of the generated tokens went into reasoning.

    The engine does not track this, so the text the reasoning parser claimed is
    re-encoded, which keeps the count consistent with the parser's rules. The
    result is clamped to ``output_tokens``: re-encoding a substring need not
    reproduce its original tokenization. 0 without a tokenizer.
    """
    if tokenizer is None:
        return 0

    total = 0
    for text in reasoning_texts:
        if not text:
            continue
        try:
            total += len(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            # Not every tokenizer accepts the keyword.
            total += len(tokenizer.encode(text))
    return min(total, output_tokens)


def _create_usage(
        final_res: GenerationResult,
        num_prompt_tokens: Optional[int] = None,
        tokenizer: Optional[TokenizerBase] = None,
        reasoning_texts: Optional[list[str]] = None) -> Optional[ResponseUsage]:
    """Build the Responses-API usage block from a finished generation.

    Clients such as the Codex CLI rely on this to track how much of the
    context window a conversation has consumed and to decide when to
    compact it, so an absent usage block leaves long sessions running
    until they overflow the context. Token counts follow the same
    accounting as the chat completions path.

    The prompt length is taken from num_prompt_tokens, which the executor
    records on the postprocessing arguments when the request is submitted.
    A result handed to a postprocessing worker carries no reference to its
    originating request, so its prompt tokens are only reachable that way.
    """
    if num_prompt_tokens is None:
        prompt_token_ids = getattr(final_res, "prompt_token_ids", None)
        if prompt_token_ids is None:
            return None
        num_prompt_tokens = len(prompt_token_ids)

    input_tokens = num_prompt_tokens
    output_tokens = sum(len(output.token_ids) for output in final_res.outputs)
    cached_tokens = getattr(final_res, "cached_tokens", None) or 0

    # Under disaggregated serving the whole prompt reached this worker as
    # transferred KV; the context phase's usage, carried with the handoff,
    # says what was actually reused.
    from tensorrt_llm.serve.postprocess_handlers import _ctx_usage_from_outputs
    ctx_prompt_tokens, ctx_cached_tokens = get_usage_tokens_from_ctx(
        _ctx_usage_from_outputs(final_res.outputs))
    if ctx_prompt_tokens is not None:
        input_tokens = ctx_prompt_tokens
        cached_tokens = ctx_cached_tokens

    return ResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=cached_tokens),
        output_tokens=output_tokens,
        output_tokens_details=OutputTokensDetails(
            reasoning_tokens=_count_reasoning_tokens(tokenizer, reasoning_texts
                                                     or [], output_tokens)),
        total_tokens=input_tokens + output_tokens,
    )


def _create_response(
    final_res: GenerationResult,
    use_harmony: bool,
    request: ResponsesRequest,
    model_name: str,
    response_creation_time: int,
    sampling_params: SamplingParams,
    reasoning_parser: Optional[str] = None,
    tool_parser: Optional[str] = None,
    num_prompt_tokens: Optional[int] = None,
    streamed_tool_calls: Optional[list[ResponseOutputItem]] = None,
    streamed_item_ids: Optional[list[StreamedItem]] = None,
    tokenizer: Optional[TokenizerBase] = None,
) -> tuple[ResponsesResponse, list[Message | ChatCompletionMessageParam]]:
    _responses_debug_log("================================================")
    _responses_debug_log("RAW MODEL OUTPUT:")
    _responses_debug_log(final_res.outputs)
    _responses_debug_log("================================================")

    # prepare responses output
    output_content = []
    reasoning_texts: list[str] = []
    if use_harmony:
        # A context-only output is a single handoff token, not a complete
        # Harmony message, so there is nothing to parse.
        output_content, output_messages = (
            ([], []) if _is_context_only(request) else
            _create_output_content_harmony(final_res))
    else:
        output_content, output_messages, reasoning_texts = _create_output_content(
            final_res,
            reasoning_parser,
            _effective_tool_parser(tool_parser, request),
            request.tools,
            chat_template_kwargs=reasoning_chat_template_kwargs(request),
            streamed_tool_calls=streamed_tool_calls,
            streamed_item_ids=streamed_item_ids)

    finish_reason = final_res.outputs[0].finish_reason
    response = ResponsesResponse.from_request(
        request=request,
        sampling_params=sampling_params,
        model_name=model_name,
        created_time=response_creation_time,
        output=output_content,
        status=finish_reason_mapping(finish_reason),
        usage=_create_usage(final_res,
                            num_prompt_tokens,
                            tokenizer=tokenizer,
                            reasoning_texts=reasoning_texts),
    )
    # Only a token-budget cut is explained; "not_finished" (a disaggregated
    # context worker handing off) also maps to "incomplete".
    if finish_reason == "length":
        response.incomplete_details = IncompleteDetails(
            reason="max_output_tokens")
    # The disaggregated handoff fields are set only on a context-only response,
    # which the orchestrator reads and strips before anything reaches a client.
    if _is_context_only(request):
        response.finish_reason = finish_reason
        response.disaggregated_params = to_disaggregated_params(
            final_res.outputs[0].disaggregated_params)
        response.prompt_token_ids = getattr(final_res, "prompt_token_ids", None)

    _responses_debug_log("========== Response ===========")
    _responses_debug_log(response)
    _responses_debug_log("===============================")

    # return output_messages for store_response
    return response, output_messages


async def create_response(
    request: ResponsesRequest,
    sampling_params: SamplingParams,
    model_name: str,
    conversation_store: ConversationHistoryStore,
    generator: Optional[AsyncGenerator[RequestOutput, None]] = None,
    generation_result: Optional[RequestOutput] = None,
    enable_store: bool = False,
    use_harmony: bool = True,
    create_time: int = None,
    reasoning_parser: Optional[str] = None,
    tool_parser: Optional[str] = None,
    num_prompt_tokens: Optional[int] = None,
    tokenizer: Optional[TokenizerBase] = None,
) -> ResponsesResponse:

    final_res: Optional[RequestOutput] = None
    response_creation_time = create_time if create_time is not None else int(
        time.time())
    prev_response_id = request.previous_response_id

    if generation_result is not None:
        final_res = generation_result
    elif generator is not None:
        final_res = await generator

    if final_res is None:
        raise RuntimeError("No output generated or provided")

    # prepare responses output
    response, output_messages = _create_response(
        final_res=final_res,
        use_harmony=use_harmony,
        request=request,
        model_name=model_name,
        response_creation_time=response_creation_time,
        sampling_params=sampling_params,
        reasoning_parser=reasoning_parser,
        tool_parser=tool_parser,
        num_prompt_tokens=num_prompt_tokens,
        tokenizer=tokenizer,
    )

    if enable_store and request.store:
        await conversation_store.store_response(resp=response,
                                                resp_msgs=output_messages,
                                                prev_resp_id=prev_response_id)

    return response


def create_response_non_store(
    generation_result: RequestOutput,
    request: ResponsesRequest,
    sampling_params: SamplingParams,
    model_name: str,
    use_harmony: bool = True,
    create_time: Optional[int] = None,
    reasoning_parser: Optional[str] = None,
    tool_parser: Optional[str] = None,
    num_prompt_tokens: Optional[int] = None,
    streamed_tool_calls: Optional[list[ResponseOutputItem]] = None,
    streamed_item_ids: Optional[list[StreamedItem]] = None,
    tokenizer: Optional[TokenizerBase] = None,
) -> ResponsesResponse:
    response_creation_time = create_time if create_time is not None else int(
        time.time())

    # prepare responses output
    response, _ = _create_response(
        final_res=generation_result,
        use_harmony=use_harmony,
        request=request,
        model_name=model_name,
        response_creation_time=response_creation_time,
        sampling_params=sampling_params,
        reasoning_parser=reasoning_parser,
        tool_parser=tool_parser,
        num_prompt_tokens=num_prompt_tokens,
        streamed_tool_calls=streamed_tool_calls,
        streamed_item_ids=streamed_item_ids,
        tokenizer=tokenizer,
    )

    return response


class ResponsesStreamingStateTracker:
    current_content_index: int = 0
    current_output_index: int = 0
    current_item_id: str = ""
    sent_output_item_added: bool = False

    # Only for non-harmony streaming
    text_sent: bool = False
    reasoning_sent: bool = False
    # Deltas already streamed for the item currently open, so it can be closed
    # with its full text if generation ends before the parser says it is done.
    text_buffer: str = ""
    reasoning_buffer: str = ""

    def __init__(self) -> None:
        # Incremental tool-call fragments, by output index, then tool index.
        self.tool_call_fragments: dict[int, dict[int, dict[str, Any]]] = {}
        # Items already streamed, in emission order; the final snapshot
        # repeats them.
        self.emitted_tool_call_items: list[ResponseOutputItem] = []
        self.emitted_item_ids: list[StreamedItem] = []


class ResponsesStreamingEventsHelper:

    def __init__(self):
        self.state_tracker = ResponsesStreamingStateTracker()

    def tool_call_fragments(self,
                            output_index: int) -> dict[int, dict[str, Any]]:
        """The call fragments accumulated so far for one output."""
        return self.state_tracker.tool_call_fragments.setdefault(
            output_index, {})

    def content_index_increment(self):
        self.state_tracker.current_content_index += 1

    def output_index_increment(self):
        self.state_tracker.current_output_index += 1

    @property
    def emitted_tool_call_items(self) -> list[ResponseOutputItem]:
        """The tool-call items already streamed, in order."""
        return self.state_tracker.emitted_tool_call_items

    @property
    def emitted_item_ids(self) -> list[StreamedItem]:
        """The reasoning/message items streamed, in order; see StreamedItem."""
        return self.state_tracker.emitted_item_ids

    def _record_item_delta(self, item_type: str, delta: str) -> None:
        # The open item is the last one announced.
        records = self.state_tracker.emitted_item_ids
        if records and records[-1].item_type == item_type:
            records[-1].text += delta

    def append_text(self, delta: str) -> None:
        self.state_tracker.text_buffer += delta
        self._record_item_delta("message", delta)

    def append_reasoning(self, delta: str) -> None:
        self.state_tracker.reasoning_buffer += delta
        self._record_item_delta("reasoning", delta)

    def take_text(self) -> str:
        text = self.state_tracker.text_buffer
        self.state_tracker.text_buffer = ""
        return text

    def take_reasoning(self) -> str:
        text = self.state_tracker.reasoning_buffer
        self.state_tracker.reasoning_buffer = ""
        return text

    @property
    def item_id(self) -> str:
        return self.state_tracker.current_item_id

    @item_id.setter
    def item_id(self, item_id: str):
        self.state_tracker.current_item_id = item_id

    @property
    def is_output_item_added_sent(self) -> bool:
        return self.state_tracker.sent_output_item_added

    @is_output_item_added_sent.setter
    def is_output_item_added_sent(self, is_sent: bool):
        self.state_tracker.sent_output_item_added = is_sent

    @property
    def is_text_sent(self) -> bool:
        return self.state_tracker.text_sent

    @is_text_sent.setter
    def is_text_sent(self, is_sent: bool):
        self.state_tracker.text_sent = is_sent

    @property
    def is_reasoning_sent(self) -> bool:
        return self.state_tracker.reasoning_sent

    @is_reasoning_sent.setter
    def is_reasoning_sent(self, is_sent: bool):
        self.state_tracker.reasoning_sent = is_sent

    def get_response_created_event(
            self, response: ResponsesResponse) -> ResponseCreatedEvent:
        return ResponseCreatedEvent(
            type="response.created",
            sequence_number=-1,  # will set by _send_event function
            response=response,
        )

    def get_response_in_progress_event(
            self, response: ResponsesResponse) -> ResponseInProgressEvent:
        return ResponseInProgressEvent(
            type="response.in_progress",
            sequence_number=-1,
            response=response,
        )

    def get_reasoning_text_done_event(
            self, text: str) -> ResponseReasoningTextDoneEvent:
        return ResponseReasoningTextDoneEvent(
            type="response.reasoning_text.done",
            item_id=self.state_tracker.current_item_id,
            sequence_number=-1,
            output_index=self.state_tracker.current_output_index,
            content_index=self.state_tracker.current_content_index,
            text=text,
        )

    def get_text_done_event(self, text: str,
                            logprobs: list[float]) -> ResponseTextDoneEvent:
        return ResponseTextDoneEvent(
            type="response.output_text.done",
            sequence_number=-1,
            output_index=self.state_tracker.current_output_index,
            content_index=self.state_tracker.current_content_index,
            text=text,
            logprobs=logprobs,
            item_id=self.state_tracker.current_item_id,
        )

    def get_content_part_done_event(
            self, part: ResponseContentPart) -> ResponseContentPartDoneEvent:
        return ResponseContentPartDoneEvent(
            type="response.content_part.done",
            sequence_number=-1,
            item_id=self.state_tracker.current_item_id,
            output_index=self.state_tracker.current_output_index,
            content_index=self.state_tracker.current_content_index,
            part=part,
        )

    def get_output_item_done_event(
            self, item: ResponseOutputItem) -> ResponseOutputItemDoneEvent:
        return ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=-1,
            output_index=self.state_tracker.current_output_index,
            item=item,
        )

    def get_output_item_added_event(
            self, item: ResponseOutputItem) -> ResponseOutputItemAddedEvent:
        return ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            sequence_number=-1,
            output_index=self.state_tracker.current_output_index,
            item=item,
        )

    def get_content_part_added_event(
            self, part: ResponseContentPart) -> ResponseContentPartAddedEvent:
        return ResponseContentPartAddedEvent(
            type="response.content_part.added",
            sequence_number=-1,
            output_index=self.state_tracker.current_output_index,
            item_id=self.state_tracker.current_item_id,
            content_index=self.state_tracker.current_content_index,
            part=part,
        )

    def get_text_delta_event(self, delta: str,
                             logprobs: list[float]) -> ResponseTextDeltaEvent:
        return ResponseTextDeltaEvent(
            type="response.output_text.delta",
            sequence_number=-1,
            content_index=self.state_tracker.current_content_index,
            output_index=self.state_tracker.current_output_index,
            item_id=self.state_tracker.current_item_id,
            delta=delta,
            logprobs=logprobs,
        )

    def get_reasoning_text_delta_event(
            self, delta: str) -> ResponseReasoningTextDeltaEvent:
        return ResponseReasoningTextDeltaEvent(
            type="response.reasoning_text.delta",
            item_id=self.state_tracker.current_item_id,
            output_index=self.state_tracker.current_output_index,
            content_index=self.state_tracker.current_content_index,
            delta=delta,
            sequence_number=-1,
        )

    def _get_output_added_events(
        self, output_item: ResponseOutputMessage | ResponseReasoningItem
    ) -> list[StreamingResponsesResponse]:
        """Get the added events for a message item.

        Returns the item-added and content-part-added events when generation
        starts.

        Returns:
            list[StreamingResponsesResponse]: A list of streaming responses responses
        """
        if not self.is_output_item_added_sent:
            self.is_output_item_added_sent = True

            self.state_tracker.emitted_item_ids.append(
                StreamedItem(item_type=output_item.type,
                             item_id=output_item.id))

            if output_item.type == "message":
                content_part = ResponseOutputText(
                    type="output_text",
                    text="",
                    annotations=[],
                    logprobs=[],
                )
            elif output_item.type == "reasoning":
                content_part = PartReasoningText(
                    type="reasoning_text",
                    text="",
                )
            else:
                raise ValueError(
                    f"Unknown content part type: {output_item.type}")

            yield self.get_output_item_added_event(output_item)
            yield self.get_content_part_added_event(content_part)

    def _start_item(self, prefix: str) -> str:
        """Assign an id for an output item that is about to be opened.

        ``current_item_id`` had no writer, so every streaming event went out
        with ``item_id=""``. Clients key their active-item state on that id:
        Codex CLI rejects the whole turn with "OutputTextDelta without active
        item" and shows no reply at all, because a delta whose item_id is
        empty matches no item it has opened.
        """
        # A closed item resets sent_output_item_added but leaves the id in
        # place, so mint a new one whenever an item is being opened. Reusing
        # one id across two output items makes the stream ambiguous for a
        # client keying its state on item_id.
        if not self.is_output_item_added_sent or not self.item_id:
            self.item_id = f"{prefix}_{_random_uuid()}"
        return self.item_id

    def get_message_output_added_events(
            self) -> list[StreamingResponsesResponse]:
        return self._get_output_added_events(output_item=ResponseOutputMessage(
            id=self._start_item("msg"),
            type="message",
            role="assistant",
            content=[],
            status="in_progress",
        ))

    def get_reasoning_output_added_events(
            self) -> list[StreamingResponsesResponse]:
        return self._get_output_added_events(output_item=ResponseReasoningItem(
            id=self._start_item("rs"),
            type="reasoning",
            summary=[],
            status="in_progress",
        ))


def _accumulate_tool_call_fragments(fragments: dict[int, dict[str, Any]],
                                    calls: list[ToolCallItem]) -> None:
    """Fold the incremental parser's call fragments into whole calls.

    A call arrives as its name with empty parameters and then argument pieces;
    fragments are keyed by the parser's tool index, in the order calls start.
    """
    for call in calls:
        fragment = fragments.get(call.tool_index)
        if fragment is None:
            fragment = fragments[call.tool_index] = {
                "name": None,
                "parameters": [],
            }
        # Only the first fragment carries the name.
        if call.name:
            fragment["name"] = call.name
        if call.parameters:
            fragment["parameters"].append(call.parameters)


def _reject_json_constant(name: str) -> None:
    raise ValueError(f"{name} is not valid JSON")


def _assembled_tool_calls(
        fragments: dict[int, dict[str, Any]]) -> list[ToolCallItem]:
    """The accumulated fragments as whole calls, in the order they started.

    Skips a call with no name or whose arguments do not assemble into valid
    JSON: a client can run neither.
    """
    calls: list[ToolCallItem] = []
    for tool_index, fragment in fragments.items():
        if not fragment["name"]:
            continue
        arguments = "".join(fragment["parameters"])
        try:
            json.loads(arguments, parse_constant=_reject_json_constant)
        except ValueError as exc:
            logger.warning(
                f"Dropping the tool call to {fragment['name']!r}: its "
                f"arguments are not valid JSON ({exc}): {arguments[:200]!r}")
            continue
        calls.append(
            ToolCallItem(tool_index=tool_index,
                         name=fragment["name"],
                         parameters=arguments))
    return calls


def _close_open_item(helper):
    """Close whichever output item is currently open, if any.

    Reasoning and text live in different item types, so a generation that
    reasons and then answers has to close the reasoning item before opening
    the message item. Without this the message deltas are emitted while the
    reasoning item is still open - the client attributes the answer to the
    reasoning item and never receives a message item at all.
    """
    if not helper.is_output_item_added_sent:
        return
    if helper.is_reasoning_sent:
        text = helper.take_reasoning()
        item = ResponseReasoningItem(
            id=helper.item_id,
            summary=[],
            type="reasoning",
            content=[Content(text=text, type="reasoning_text")],
            status="completed",
        )
        yield helper.get_reasoning_text_done_event(text)
        yield helper.get_content_part_done_event(
            PartReasoningTextDone(type="reasoning_text", text=text))
        yield helper.get_output_item_done_event(item)
        helper.is_reasoning_sent = False
    else:
        text = helper.take_text()
        content = ResponseOutputText(text=text,
                                     annotations=[],
                                     type="output_text",
                                     logprobs=None)
        item = ResponseOutputMessage(id=helper.item_id,
                                     content=[content],
                                     role="assistant",
                                     status="completed",
                                     type="message")
        yield helper.get_text_done_event(text, [])
        yield helper.get_content_part_done_event(content)
        yield helper.get_output_item_done_event(item)
        helper.is_text_sent = False
    helper.output_index_increment()
    helper.is_output_item_added_sent = False


def _generate_streaming_event(
    output: RequestOutput,
    request: ResponsesRequest,
    finished_generation: bool,
    streaming_events_helper: ResponsesStreamingEventsHelper,
    reasoning_parser_id: Optional[str] = None,
    tool_parser_id: Optional[str] = None,
    reasoning_parser_dict: Optional[dict[int, BaseReasoningParser]] = None,
    tool_parser_dict: Optional[dict[int, BaseToolParser]] = None,
):
    available_tools = _get_chat_completion_function_tools(request.tools)
    tool_parser_id = _effective_tool_parser(tool_parser_id, request)
    output_idx = output.index
    delta_text = output.text_diff
    calls = []

    def check_parser(parser_id: Optional[str],
                     parser_dict: Optional[dict[int, BaseReasoningParser]]):
        if parser_id is not None:
            if parser_dict is None:
                raise RuntimeError(
                    f"Parser({parser_id}) dictionary is not provided for streaming"
                )

    check_parser(reasoning_parser_id, reasoning_parser_dict)
    check_parser(tool_parser_id, tool_parser_dict)

    delta_text, reasoning_delta_text = _apply_reasoning_parser(
        reasoning_parser_id=reasoning_parser_id,
        output_index=output_idx,
        text=delta_text,
        streaming=True,
        reasoning_parser_dict=reasoning_parser_dict,
        finished=finished_generation,
        chat_template_kwargs=reasoning_chat_template_kwargs(request),
    )

    if delta_text:
        delta_text, calls = _apply_tool_parser(
            tool_parser_id=tool_parser_id,
            tools=available_tools,
            output_index=output_idx,
            text=delta_text,
            streaming=True,
            tool_parser_dict=tool_parser_dict,
        )
    tool_parser = (tool_parser_dict or {}).get(output_idx)
    # Calls are assembled from the parser's increments only when those match its
    # whole-text parse; otherwise they come from a whole-text parse at the end.
    incremental_calls = (tool_parser is not None
                         and tool_parser.streaming_matches_whole_parse)

    _responses_debug_log(
        repr(
            f" ---------> delta text: {delta_text}, reasoning delta text: {reasoning_delta_text}, calls: {calls}"
        ))

    # Deltas go out before any done event, so each lands in the item it belongs
    # to, and an item is opened before its first delta (whitespace included).
    if reasoning_delta_text:
        if streaming_events_helper.is_text_sent:
            yield from _close_open_item(streaming_events_helper)
        streaming_events_helper.is_reasoning_sent = True
        yield from streaming_events_helper.get_reasoning_output_added_events()
        streaming_events_helper.append_reasoning(reasoning_delta_text)
        yield streaming_events_helper.get_reasoning_text_delta_event(
            reasoning_delta_text)
    if delta_text:
        if streaming_events_helper.is_reasoning_sent:
            yield from _close_open_item(streaming_events_helper)
        streaming_events_helper.is_text_sent = True
        yield from streaming_events_helper.get_message_output_added_events()
        streaming_events_helper.append_text(delta_text)
        yield streaming_events_helper.get_text_delta_event(delta_text, [])

    # A call has started, so the open item ends here; only the incremental
    # parser knows this while the call is still being generated.
    if calls:
        yield from _close_open_item(streaming_events_helper)

    if incremental_calls:
        call_fragments = streaming_events_helper.tool_call_fragments(output_idx)
        _accumulate_tool_call_fragments(call_fragments, calls)

    if not finished_generation:
        return

    final_calls: list[ToolCallItem] = []
    released_text = ""
    if incremental_calls:
        # finish() reports the calls the parser still holds and releases the
        # rest as text, as the whole-text parse reads it.
        flushed = tool_parser.finish(available_tools)
        released_text = flushed.normal_text
        _accumulate_tool_call_fragments(call_fragments, flushed.calls)
        final_calls = _assembled_tool_calls(call_fragments)
    elif tool_parser is not None:
        released_text, final_calls = _whole_text_tool_calls(
            output, request, streaming_events_helper, reasoning_parser_id,
            tool_parser_id, available_tools)

    # Text the parser still held follows everything the last chunk carried.
    if released_text:
        if streaming_events_helper.is_reasoning_sent:
            yield from _close_open_item(streaming_events_helper)
        streaming_events_helper.is_text_sent = True
        yield from streaming_events_helper.get_message_output_added_events()
        streaming_events_helper.append_text(released_text)
        yield streaming_events_helper.get_text_delta_event(released_text, [])

    # Nothing transitions after the last item, so it is closed here.
    yield from _close_open_item(streaming_events_helper)

    # Calls go out once generation finishes, after every other item. Already
    # emitted calls are skipped, so a finished output seen twice is harmless.
    emitted = streaming_events_helper.emitted_tool_call_items
    pending = final_calls[len(emitted):]
    if pending:
        tool_resolution = _tool_resolution(request.tools)
        for call in pending:
            tool_call_item = _tool_call_output_item(call,
                                                    tool_resolution,
                                                    status="completed")
            streaming_events_helper.item_id = tool_call_item.id
            yield streaming_events_helper.get_output_item_added_event(
                tool_call_item)
            yield streaming_events_helper.get_output_item_done_event(
                tool_call_item)
            emitted.append(tool_call_item)
            streaming_events_helper.output_index_increment()
        streaming_events_helper.is_output_item_added_sent = False


def _generate_streaming_event_harmony(
    harmony_adapter: HarmonyAdapter,
    stream_request_id: str,
    output: RequestOutput,
    request: ResponsesRequest,
    streaming_events_helper: ResponsesStreamingEventsHelper,
):
    tools = [tool.model_dump() for tool in request.tools]
    messages = harmony_adapter.stateful_stream_harmony_tokens_to_openai_messages(
        stream_request_id, output.token_ids_diff, tools, request.tool_choice)
    stream_state = harmony_adapter.get_stream_state(stream_request_id)
    assert stream_state is not None
    parser = stream_state.get_parser()
    if parser.state == StreamState.EXPECT_START:
        streaming_events_helper.output_index_increment()
        streaming_events_helper.is_output_item_added_sent = False

        if len(messages) > 0:
            previous_item = messages[-1]
            if previous_item.recipient is not None:
                # Deal with tool call here
                pass
            elif previous_item.channel == "analysis":
                reasoning_item = ResponseReasoningItem(
                    type="reasoning",
                    content=[
                        Content(
                            text=previous_item.content[0].text,
                            type="reasoning_text",
                        ),
                    ],
                    status="completed",
                    id=streaming_events_helper.item_id,
                    summary=[],
                )
                yield streaming_events_helper.get_reasoning_text_done_event(
                    previous_item.content[0].text)
                yield streaming_events_helper.get_output_item_done_event(
                    reasoning_item)

            elif previous_item.channel == "final":
                text_content = ResponseOutputText(
                    type="output_text",
                    text=previous_item.content[0].text,
                    annotations=[],
                )

                text_item = ResponseOutputMessage(
                    id=streaming_events_helper.item_id,
                    type="message",
                    role="assistant",
                    content=[text_content],
                    status="completed",
                )

                yield streaming_events_helper.get_text_done_event(
                    previous_item.content[0].text, [])
                yield streaming_events_helper.get_content_part_done_event(
                    text_content)
                yield streaming_events_helper.get_output_item_done_event(
                    text_item)

    if parser.last_content_delta:
        if (parser.current_channel == "final"
                and parser.current_recipient is None):
            if not streaming_events_helper.is_output_item_added_sent:
                streaming_events_helper.is_output_item_added_sent = True

                output_item = ResponseOutputMessage(
                    id=streaming_events_helper.item_id,
                    type="message",
                    role="assistant",
                    content=[],
                    status="in_progress",
                )

                content_part = ResponseOutputText(
                    type="output_text",
                    text="",
                    annotations=[],
                    logprobs=[],
                )
                yield streaming_events_helper.get_output_item_added_event(
                    output_item)
                yield streaming_events_helper.get_content_part_added_event(
                    content_part)

            yield streaming_events_helper.get_text_delta_event(
                parser.last_content_delta, [])

        elif (parser.current_channel == "analysis"
              and parser.current_recipient is None):
            if not streaming_events_helper.is_output_item_added_sent:
                streaming_events_helper.is_output_item_added_sent = True

                reasoning_item = ResponseReasoningItem(
                    id=streaming_events_helper.item_id,
                    type="reasoning",
                    summary=[],
                    status="in_progress",
                )

                reasoning_content = PartReasoningText(
                    type="reasoning_text",
                    text="",
                )

                yield streaming_events_helper.get_output_item_added_event(
                    reasoning_item)
                yield streaming_events_helper.get_content_part_added_event(
                    reasoning_content)

            yield streaming_events_helper.get_reasoning_text_delta_event(
                parser.last_content_delta)


def _stream_terminal_event(
    final_response: ResponsesResponse,
    sequence_number: int = -1,
) -> Union[ResponseCompletedEvent, ResponseIncompleteEvent,
           ResponseFailedEvent]:
    """The terminal event matching the response's status.

    "incomplete" ends in ``response.incomplete`` and "failed" in
    ``response.failed``; everything else, including the client-initiated
    "cancelled", ends in ``response.completed``.
    """
    payload = final_response.model_dump(by_alias=True)
    if final_response.status == "incomplete":
        return ResponseIncompleteEvent(
            type="response.incomplete",
            sequence_number=sequence_number,
            response=payload,
        )
    if final_response.status == "failed":
        return ResponseFailedEvent(
            type="response.failed",
            sequence_number=sequence_number,
            response=payload,
        )
    return ResponseCompletedEvent(
        type="response.completed",
        sequence_number=sequence_number,
        response=payload,
    )


class ResponsesStreamingProcessor:

    def __init__(
        self,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        model_name: str,
        create_time: Optional[int] = None,
        conversation_store: Optional[ConversationHistoryStore] = None,
        enable_store: bool = False,
        use_harmony: bool = True,
        reasoning_parser: Optional[str] = None,
        tool_parser: Optional[str] = None,
    ):
        self.model_name = model_name
        self.request = request
        self.sampling_params = sampling_params
        # get_initial_responses numbers the opening pair 0 and 1, so every
        # later event continues from 2 -- also in a postprocessing worker,
        # which receives a copy of this object before the pair is sent.
        self.sequence_number = 2
        self.streaming_events_helper = ResponsesStreamingEventsHelper()
        self.response_creation_time = create_time if create_time is not None else int(
            time.time())
        self.final_res: Optional[RequestOutput] = None
        self.reasoning_parser_dict: dict[int, BaseReasoningParser] = {}
        self.tool_parser_dict: dict[int, BaseToolParser] = {}
        self.stream_request_id = f"responses-api-{request.request_id}"
        self.conversation_store = conversation_store
        self.enable_store = enable_store
        self.use_harmony = use_harmony
        self.reasoning_parser = reasoning_parser
        self.tool_parser = tool_parser

    def _send_event(self, event: OpenAIBaseModel):
        if hasattr(event, 'sequence_number'):
            event.sequence_number = self.sequence_number
        self.sequence_number += 1
        return _format_sse_event(event)

    def get_initial_responses(self) -> List[str]:
        initial_response = ResponsesResponse.from_request(
            request=self.request,
            sampling_params=self.sampling_params,
            model_name=self.model_name,
            created_time=self.response_creation_time,
            output=[],
            status="in_progress",
            usage=None,
        ).model_dump(by_alias=True)
        created = self.streaming_events_helper.get_response_created_event(
            initial_response)
        in_progress = self.streaming_events_helper.get_response_in_progress_event(
            initial_response)
        created.sequence_number = 0
        in_progress.sequence_number = 1
        return [_format_sse_event(created), _format_sse_event(in_progress)]

    async def get_final_response(
        self,
        final_res: RequestOutput,
        num_prompt_tokens: Optional[int] = None,
    ) -> str:
        final_response = await create_response(
            generator=None,
            request=self.request,
            sampling_params=self.sampling_params,
            model_name=self.model_name,
            conversation_store=self.conversation_store,
            generation_result=final_res,
            enable_store=self.enable_store,
            use_harmony=self.use_harmony,
            create_time=self.response_creation_time,
            reasoning_parser=self.reasoning_parser,
            tool_parser=self.tool_parser,
            num_prompt_tokens=num_prompt_tokens,
        )

        return self._send_event(
            ResponseCompletedEvent(
                type="response.completed",
                sequence_number=-1,
                response=final_response.model_dump(),
            ))

    def get_final_response_non_store(
        self,
        final_res: RequestOutput,
        num_prompt_tokens: Optional[int] = None,
        tokenizer: Optional[TokenizerBase] = None,
    ) -> str:
        """The terminal event; its snapshot repeats the items already streamed.

        ``tokenizer`` (for counting reasoning tokens) is an argument because a
        postprocessing worker holds a pickled copy of this object.
        """
        final_response = create_response_non_store(
            generation_result=final_res,
            request=self.request,
            sampling_params=self.sampling_params,
            model_name=self.model_name,
            use_harmony=self.use_harmony,
            create_time=self.response_creation_time,
            reasoning_parser=self.reasoning_parser,
            tool_parser=self.tool_parser,
            num_prompt_tokens=num_prompt_tokens,
            streamed_tool_calls=self.streaming_events_helper.
            emitted_tool_call_items,
            streamed_item_ids=self.streaming_events_helper.emitted_item_ids,
            tokenizer=tokenizer,
        )
        return self._send_event(_stream_terminal_event(final_response))

    def process_single_output(self, res: GenerationResult) -> list[str]:
        event_generator = None
        output = res.outputs[0]
        if self.use_harmony:
            event_generator = _generate_streaming_event_harmony(
                harmony_adapter=get_harmony_adapter(),
                stream_request_id=self.stream_request_id,
                output=output,
                request=self.request,
                streaming_events_helper=self.streaming_events_helper,
            )

        else:
            event_generator = _generate_streaming_event(
                output=output,
                request=self.request,
                finished_generation=res._done,
                streaming_events_helper=self.streaming_events_helper,
                reasoning_parser_id=self.reasoning_parser,
                tool_parser_id=self.tool_parser,
                reasoning_parser_dict=self.reasoning_parser_dict,
                tool_parser_dict=self.tool_parser_dict,
            )

        if event_generator is None:
            raise RuntimeError("Failed to generate streaming events")

        return [self._send_event(event) for event in event_generator]

    def get_stream_failed_events(self,
                                 cause: str,
                                 detail: str,
                                 events_sent: int = 0) -> List[str]:
        """``error`` and ``response.failed`` for a stream that stopped early.

        ``events_sent`` is the number of frames that reached the wire; the
        terminal events are numbered after them.
        """
        self.sequence_number = events_sent
        error_event = self._send_event(
            ResponseErrorEvent(
                type="error",
                sequence_number=-1,
                code=cause,
                message=detail,
                param=None,
            ))
        snapshot = ResponsesResponse.from_request(
            request=self.request,
            sampling_params=self.sampling_params,
            model_name=self.model_name,
            created_time=self.response_creation_time,
            output=[],
            status="failed",
            usage=None,
        ).model_dump(by_alias=True)
        # Set on the dump: the SDK Response carries `error`, ResponsesResponse
        # does not.
        snapshot["error"] = {"code": "server_error", "message": detail}
        failed_event = self._send_event(
            ResponseFailedEvent(
                type="response.failed",
                sequence_number=-1,
                response=snapshot,
            ))
        return [error_event, failed_event]


# --------------------------------------------------------------------------
# Abnormal stream termination
# --------------------------------------------------------------------------

# Why a stream stopped before its terminal event, sent as the `error` code.
STREAM_TERMINATION_ENGINE_ERROR = "engine_error"
STREAM_TERMINATION_UPSTREAM_ERROR = "upstream_error"
STREAM_TERMINATION_INTERNAL_ERROR = "internal_error"

_SSE_EVENT_DELIMITER = "\n\n"
# A terminal event frame always follows another frame, and JSON payloads
# escape newlines, so these match only at a frame boundary.
_TERMINAL_FRAME_STARTS = tuple(f"{_SSE_EVENT_DELIMITER}event: response.{kind}"
                               for kind in ("completed", "incomplete",
                                            "failed"))
_TERMINAL_FRAME_STARTS_BYTES = tuple(marker.encode()
                                     for marker in _TERMINAL_FRAME_STARTS)
_SSE_TAIL_LENGTH = max(len(marker) for marker in _TERMINAL_FRAME_STARTS)


def _sse_delimiter_and_markers(sample: Any) -> Tuple[Any, tuple]:
    if isinstance(sample, bytes):
        return _SSE_EVENT_DELIMITER.encode(), _TERMINAL_FRAME_STARTS_BYTES
    return _SSE_EVENT_DELIMITER, _TERMINAL_FRAME_STARTS


def classify_stream_termination(exc: Exception) -> str:
    """Attribute a stream-ending exception to a cause."""
    if isinstance(exc, (RequestError, EngineDeadError)):
        return STREAM_TERMINATION_ENGINE_ERROR
    # By module, to keep the HTTP client library out of this layer.
    if type(exc).__module__.split(".")[0] in ("aiohttp", "httpx", "httpcore"):
        return STREAM_TERMINATION_UPSTREAM_ERROR
    return STREAM_TERMINATION_INTERNAL_ERROR


def describe_stream_termination(exc: Exception, cause: str) -> str:
    """The client-facing detail of a stream-ending exception.

    Only engine errors carry their message; transport and internal errors can
    name internal hosts.
    """
    if cause == STREAM_TERMINATION_ENGINE_ERROR and str(exc):
        return f"{type(exc).__name__}: {exc}"
    return type(exc).__name__


def stream_error_event(cause: str, detail: str,
                       events_sent: int) -> List[bytes]:
    """A bare ``error`` event, for a relay that holds no response to snapshot."""
    return [
        _sse_event(
            ResponseErrorEvent(
                type="error",
                sequence_number=events_sent,
                code=cause,
                message=detail,
                param=None,
            ))
    ]


async def guard_responses_stream(
    stream: AsyncGenerator[Any, None],
    terminal_events: Callable[[str, str, int], List[Any]],
) -> AsyncGenerator[Any, None]:
    """End a stream that fails before its terminal event with events saying so.

    The exception is re-raised and frames pass through untouched. A client hangup (CancelledError or
    GeneratorExit) is not caught: there is nobody left to tell.
    """
    events_sent = 0
    tail = None
    completed = False
    try:
        async for chunk in stream:
            if not completed:
                if tail is None:
                    tail = chunk[:0]
                text = tail + chunk
                delimiter, markers = _sse_delimiter_and_markers(text)
                events_sent += text.count(delimiter) - tail.count(delimiter)
                completed = any(marker in text for marker in markers)
                tail = text[-_SSE_TAIL_LENGTH:]
            yield chunk
    except Exception as exc:
        if completed:
            raise
        cause = classify_stream_termination(exc)
        logger.error("Responses stream terminated before completion "
                     f"({cause}): {type(exc).__name__}: {exc}")
        try:
            frames = terminal_events(cause,
                                     describe_stream_termination(exc, cause),
                                     events_sent)
        except Exception as report_error:  # noqa: BLE001 - keep the original error
            logger.error(f"Failed to build the terminal event: {report_error}")
            frames = []
        if frames and tail:
            delimiter, _ = _sse_delimiter_and_markers(tail)
            if not tail.endswith(delimiter):
                # A relay can fail mid-frame; end that frame before ours.
                yield delimiter
        for frame in frames:
            yield frame
        raise


async def process_streaming_events(
    generator,
    request: ResponsesRequest,
    sampling_params: SamplingParams,
    model_name: str,
    conversation_store: ConversationHistoryStore,
    enable_store: bool = False,
    use_harmony: bool = True,
    create_time: Optional[int] = None,
    reasoning_parser: Optional[str] = None,
    tool_parser: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    streaming_processor = ResponsesStreamingProcessor(
        request=request,
        sampling_params=sampling_params,
        model_name=model_name,
        create_time=create_time,
        conversation_store=conversation_store,
        enable_store=enable_store,
        use_harmony=use_harmony,
        reasoning_parser=reasoning_parser,
        tool_parser=tool_parser,
    )

    initial_responses = streaming_processor.get_initial_responses()
    for initial_response in initial_responses:
        yield initial_response

    async for res in generator:
        final_res = res
        events = streaming_processor.process_single_output(res)
        for event in events:
            yield event

    final_response = await streaming_processor.get_final_response(final_res)

    yield final_response


class ServerArrivalTimeMiddleware:
    """
    Custom ASGI middleware to track server arrival time.

    We implement this as a pure ASGI middleware instead of using FastAPI's
    @app.middleware("http") decorator because the decorator internally uses
    BaseHTTPMiddleware, which wraps the ASGI `receive` callable. This wrapping
    breaks Request.is_disconnected() functionality - the wrapped receive doesn't
    properly forward http.disconnect events while the middleware is waiting in
    call_next(), preventing detection of client disconnections during long-running
    non-streaming requests.

    By implementing pure ASGI middleware, we pass through the original receive/send
    callables unchanged, preserving the ability to detect client disconnections.

    See: https://github.com/encode/starlette/discussions/2094
    """

    def __init__(self,
                 app,
                 adjusted_clock: Optional[AdjustedSteadyClock] = None):
        self.app = app
        self._adjusted_clock = adjusted_clock or AdjustedSteadyClock()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Add arrival time to scope
            scope["state"] = {}
            scope["state"]["server_arrival_time"] = self._adjusted_clock.now()

        # Pass through the original receive/send - no wrapping!
        await self.app(scope, receive, send)


class PeriodicLatencyLogger:
    """Periodically log latency percentiles for a named coordinator API.

    This lock-free, self-resetting logger runs on one asyncio loop and profiles
    the in-process owner and HTTP client without per-call log spam.
    """

    def __init__(self, name: str, window: int = 500):
        self._name = name
        self._window = window
        self._samples: List[float] = []
        self._n = 0

    def record(self, dt_s: float) -> None:
        self._samples.append(dt_s * 1000.0)  # ms
        self._n += 1
        if self._n % self._window == 0:
            s = sorted(self._samples)
            m = len(s)

            def percentile(q):
                return s[min(int(q * m), m - 1)]

            logger.info(f"[coord_api] {self._name} n={self._n} ms: "
                        f"mean={sum(s)/m:.2f} p50={percentile(0.5):.2f} "
                        f"p90={percentile(0.9):.2f} "
                        f"p99={percentile(0.99):.2f} max={s[-1]:.2f}")
            self._samples = []


class ResponseHooks(ABC):
    """
    Hooks for response processing and (disagg) service perf observability.
    """

    @abstractmethod
    def on_req_begin(self, request: UCompletionRequest):
        pass

    def on_disagg_request_id(self, disagg_request_id: int):
        """Receive the request ID immediately after the service allocates it."""

    def on_ctx_dispatch(self, request: UCompletionRequest):
        """Record when the disaggregated service starts context placement.

        Arrival to this point measures the pre-context wait in the orchestrator
        or fleet. The default is a no-op for non-instrumented implementations.
        """

    def on_perf_metrics(self, server: str, role: str, metrics: dict):
        """Receive request-local metrics carried by an upstream response."""

    @abstractmethod
    def on_ctx_resp(self, ctx_server: str, response: UCompletionResponse):
        pass

    @abstractmethod
    def on_first_token(self,
                       gen_server: str,
                       request: UCompletionRequest,
                       response: UCompletionResponse = None):
        pass

    @abstractmethod
    def on_resp_done(self,
                     gen_server: str,
                     request: UCompletionRequest,
                     response: UCompletionResponse = None):
        pass


async def done_generator() -> AsyncGenerator[bytes, None]:
    yield "data: [DONE]\n\n".encode('utf-8')


def _format_sse_event(event: OpenAIBaseModel) -> str:
    # by_alias: fields such as text.format.schema go out under their wire name.
    return (f"event: {getattr(event, 'type', 'unknown')}\n"
            f"data: {event.model_dump_json(indent=None, by_alias=True)}\n\n")


def _sse_event(event: StreamingResponsesResponse) -> bytes:
    return _format_sse_event(event).encode("utf-8")


async def responses_done_generator(
        response: ResponsesResponse) -> AsyncGenerator[bytes, None]:
    """Stream an already-complete response as a well-formed SSE run.

    Used for a response the context worker finished; Responses has no
    ``[DONE]``.
    """
    payload = response.model_dump(by_alias=True)
    yield _sse_event(
        ResponseCreatedEvent(
            type="response.created",
            response=payload,
            sequence_number=0,
        ))
    yield _sse_event(
        ResponseInProgressEvent(
            type="response.in_progress",
            response=payload,
            sequence_number=1,
        ))
    yield _sse_event(_stream_terminal_event(response, sequence_number=2))


UCompletionResponseOrGenerator = Union[UCompletionResponse,
                                       AsyncGenerator[Any, None]]
