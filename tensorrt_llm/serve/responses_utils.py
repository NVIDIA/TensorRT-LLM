# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import base64
import json
import os
import time
import uuid
# yapf: disable
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Mapping
from copy import copy
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
from tensorrt_llm.inputs.utils import async_apply_chat_template
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
                                                UCompletionResponse, UsageInfo,
                                                to_disaggregated_params)
from tensorrt_llm.serve.responses_web_search import is_web_search_tool
from tensorrt_llm.serve.tool_parser.base_tool_parser import (
    BaseToolParser, warn_if_tool_call_unparsed)
from tensorrt_llm.serve.tool_parser.core_types import ToolCallItem
from tensorrt_llm.serve.tool_parser.tool_parser_factory import ToolParserFactory
from tensorrt_llm.serve.web_search import load_web_search_config

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


def _responses_debug_log(msg):
    if ENABLE_RESPONSES_DEBUG_MSG:
        logger.info(msg)


def _responses_debug_enabled() -> bool:
    """Whether the caller should bother building a debug message.

    ``_responses_debug_log`` drops the message, but its argument is evaluated
    first. Callers whose argument costs real work - decoding a whole prompt,
    for instance - must check this before constructing it.
    """
    return ENABLE_RESPONSES_DEBUG_MSG


def _is_context_only(request: ResponsesRequest) -> bool:
    """Whether this request is the context half of a disaggregated split.

    Such a request comes from the orchestrator rather than a client, and its
    response is consumed by the orchestrator alone.
    """
    params = getattr(request, "disaggregated_params", None)
    return params is not None and params.request_type == "context_only"


def _relayed_prompt_token_ids(request: ResponsesRequest) -> Optional[list[int]]:
    """The already-tokenized prompt the orchestrator relayed, if any.

    Returns None for an ordinary client request, which carries neither field
    and must be rendered and tokenized here.
    """
    if request.prompt_token_ids is not None:
        return request.prompt_token_ids
    if not request.prompt_token_ids_b64:
        return None
    # int32 little-endian buffer, matching what the context worker encodes.
    import numpy as np
    decoded = np.frombuffer(base64.b64decode(request.prompt_token_ids_b64),
                            dtype=np.int32).tolist()
    # Cache it so a later reader does not decode a second time.
    request.prompt_token_ids = decoded
    return decoded


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
            self.conversations[conversation_id] = msgs

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
            if tool.type in ("web_search_preview", "code_interpreter"):
                # These are built-in tools that are added to the system message.
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
            # Reached on the context worker in disaggregated serving: it is
            # capped at one token and hands the request off, so the engine
            # reports that generation has not finished. "incomplete" is the
            # closest public status, and the orchestrator does not read it -
            # it reads the unmapped value off ResponsesResponse.finish_reason.
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
                # The KF client serializes a sub-agent task payload inside an
                # agent_message as a part typed `encrypted_content` whose
                # same-named field holds the plain readable task text - the
                # field name is historical, not a description of the value.
                # Dropping the part delivered the task header ("Payload:")
                # with no payload behind it; one measured request lost a
                # 721-character task this way. Only a string is taken at face
                # value: a non-string here really is opaque, and guessing at
                # it would fabricate input, so those stay dropped as before.
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


# The Responses API spells a text part `input_text` on the way in and
# `output_text` on the way out; the chat-completions content parser knows only
# `text`. Both mean the same thing, so they are translated rather than
# rejected.
_RESPONSES_TEXT_PART_TYPES = frozenset(("input_text", "output_text"))


def _chat_content_parts(content: list) -> list:
    """Rewrite Responses-only text parts into the chat vocabulary.

    Everything else is passed through untouched: `image_url` and friends are
    already understood downstream, and flattening the list to a plain string
    would drop them.
    """
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

    `output` is a string on the simple path and a list of Responses content
    parts when the client structures it. Passing a list through untouched
    hands `input_text` to a parser that knows only `text`, which fails the
    whole request rather than the item -- and a tool result arrives on every
    turn after the first tool call, so the conversation stops there.
    """
    if isinstance(output, list):
        return _chat_content_parts(output)
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    return str(output)


# `developer` is OpenAI's rename of `system`. Chat templates written before
# that rename dispatch on role and simply have no branch for it -- GLM-5.3
# ends its dispatch after `system` with no fallback -- so a developer message
# renders to nothing and the client's instructions never reach the model.
# Every template understands `system`.
_ROLE_ALIASES = {"developer": "system"}


def _chat_role(role: Optional[str]) -> str:
    """The role to hand a chat template, as that template will understand it."""
    if not role:
        return "assistant"
    return _ROLE_ALIASES.get(role, role)


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
            item = {**item, "role": _chat_role(item.get("role"))}
            content = item.get("content")
            if isinstance(content, list):
                # An item with a role and no `type` is the API's
                # EasyInputMessage, where `type` defaults to "message". Its
                # parts are spelled in the Responses vocabulary, so passing
                # them through untouched hands `input_text` / `output_text` to
                # the chat-completions parser, which knows neither and fails
                # the request. The explicit "message" branch below already
                # handles those parts, so leaving this one verbatim made
                # success depend on a field the client may omit.
                return {**item, "content": _chat_content_parts(content)}
            return item
        case "message" | "reasoning":
            content = item.get("content") or []
            if not content and item_type == "reasoning":
                # Reasoning does not have to carry `content`. The API puts the
                # text in `summary` when a summary was requested and in
                # `encrypted_content` when it was not, and an item with neither
                # populated is a shape OpenAI itself emits. Falling back to the
                # summary keeps whatever text exists; when nothing readable is
                # left the item is skipped rather than rejected, because the
                # reasoning was already absent from the payload and failing
                # here costs the whole conversation instead.
                summary = item.get("summary") or []
                summary_text = "".join(
                    part.get("text") or "" if isinstance(part, dict) else (
                        getattr(part, "text", "") or "") for part in summary)
                if not summary_text:
                    return None
                return {"role": "assistant", "reasoning": summary_text}
            if not content:
                raise ValueError(
                    f"Input item of type {item_type!r} has empty or missing 'content'"
                )
            if isinstance(content, str):
                # The API accepts `content` as one plain string, and the
                # untyped branch above already preserves that shape - `type`
                # merely defaults to "message" there. Walking the string as a
                # list of parts iterates its characters, none of which carry a
                # `text` field, so an explicit {"type": "message", "content":
                # "hello"} converted to an empty message while the identical
                # item without `type` survived. The two spellings mean the
                # same thing and must convert the same way.
                text = content
            else:
                # Join every text part. Taking content[0] silently dropped the
                # rest of a multi-part message.
                parts = []
                for part in content:
                    text = part.get("text") if isinstance(
                        part, dict) else getattr(part, "text", None)
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
            return {"role": _chat_role(item.get("role")), "content": text}
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


async def _create_input_messages(
    request: ResponsesRequest,
    prev_msgs: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    messages: list[ChatCompletionMessageParam] = []
    if request.instructions:
        messages.append({
            "role": "system",
            "content": request.instructions,
        })

    # Prepend the conversation history.
    #
    # Reasoning is stripped on replay to save tokens, but a stored turn is
    # more than its reasoning: the store writer attaches the turn's tool
    # calls to the reasoning message, so discarding the whole message for its
    # "reasoning" key also discarded the calls - and the tool RESULT the
    # client sends next then replayed with no call before it, an orphan the
    # model cannot pair with anything. Strip the reasoning, keep the calls.
    # A turn that was only reasoning still vanishes, and a message without
    # the key passes through exactly as stored.
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
        for inp in request.input:
            message = _response_output_item_to_chat_completion_message(inp)
            if message is not None:
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

    Reasoning is stored here and stripped on replay instead
    (_create_input_messages), which keeps the token saving while letting the
    replay keep the tool calls the reasoning message carries. Tool calls are
    stored on whichever assistant message the turn produced - reasoning
    first, else the text message, else a bare assistant tool-call message -
    because a turn's calls must survive storage no matter what else the turn
    contained.

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
        # The calls used to be stored only on the reasoning message, so a
        # turn that called tools without reasoning lost them - and a turn
        # that was *nothing but* tool calls stored no assistant message at
        # all, leaving the client's tool RESULT to replay against a call
        # that was never in the history. A tool-call-bearing turn is an
        # assistant turn whether or not any text or reasoning came with it:
        # attach the calls to the text message when there is one, store a
        # bare assistant tool-call message when there is not.
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

    conversation, mm_coroutines, mm_placeholder_counts, _ = parse_chat_messages_coroutines(
        messages, model_config)
    tools_dict = [
        tool.model_dump()
        for tool in _get_chat_completion_function_tools(request.tools)
    ]
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

    # In disaggregated serving the context worker has already rendered the chat
    # template and tokenized; the orchestrator relays the result so the
    # generation worker does not repeat that work on a prompt it is being
    # handed verbatim. Rendering again would also be wrong, not just wasteful:
    # a template that samples anything per-render could produce a prompt the
    # context worker never prefilled, and the KV cache would not match.
    pretokenized = _relayed_prompt_token_ids(request)
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

    if _responses_debug_enabled():
        # Guarded rather than passed straight to _responses_debug_log: decoding
        # is a real tokenizer call on every request, and the argument would be
        # evaluated whether or not debug logging is on.
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


def _flush_tool_parser(
    tools: Optional[list[Tool]],
    output_index: int,
    tool_parser_dict: Optional[dict[int, BaseToolParser]] = None,
) -> Tuple[str, list[ToolCallItem], Optional[int]]:
    """Whatever the incremental tool parser still holds when the stream ends.

    Returns the text to release, the calls to add, and the parser's tool index
    for a call it was cut off in the middle of (None if it was not).

    The parser withholds any bytes that could still turn out to be tool-call
    markup, and reports at most one finished call per increment, so the end of
    a stream can leave both a complete call and a half-finished one behind with
    no further increment coming. `finish` is each parser's own decision about
    that remainder; most inherit the no-op, which leaves it sitting in
    `_buffer`, so the buffer is drained and then inspected as well.

    Unterminated markup is released as ordinary text rather than dropped.
    Dropping was the first design, on the grounds that half of a call is not
    a message the model meant to send - but the final response never dropped
    it: `_create_output_content` re-parses the whole text, its regex needs
    the closing tag, and the fragment falls back into the message item. The
    stream and the snapshot then told two different stories about the same
    generation - trace tr_8f312e9973954fa784ae7dc8f45e9d3e ended in a
    429-character unterminated call that the final response carried as
    message text while the stream showed nothing at all. Of the two ways to
    make the views agree, keeping the text wins: these turns are recorded as
    training data, and silently losing model output is worse than showing a
    call that never closed. It is rare (1 of 3437 recorded responses) and
    the warning keeps the malformed turn visible in the log.

    A remainder with no markup in it is ordinary output that the parser was
    holding only until it could rule out a call, and is released. This matches
    what DeepSeekR1Parser.finish already does for a trailing fragment that
    might still have grown into a `</think>`.
    """
    tool_parser = (tool_parser_dict or {}).get(output_index)
    if tool_parser is None or tools is None:
        # No parser was ever built for this output, so nothing is held back.
        return "", [], None

    released: list[str] = []
    calls: list[ToolCallItem] = []

    # Drain calls the parser has already received in full but not yet reported.
    #
    # Several parsers report at most one call per increment and leave the rest
    # buffered for the next one - Glm47ToolParser returns the moment it sees a
    # `</tool_call>`. Two calls arriving in a single stream chunk therefore
    # leave the second sitting complete and unreported, and at end of stream
    # there is no next increment to collect it; replaying one recorded
    # two-call response under randomised chunk boundaries lost a call in 67 of
    # 400 chunkings. Empty increments stand in for the chunks that will never
    # arrive.
    #
    # Bounded by the buffer having to shrink on every pass, so a parser that
    # cannot make progress on what it holds ends the loop rather than spinning.
    # Entered only while the buffer still holds markup, because plain text left
    # over is `finish`'s business and some parsers strip framing from it.
    held = getattr(tool_parser, "_buffer", "")
    while held and tool_parser.has_tool_call(held):
        drained = tool_parser.parse_streaming_increment("", tools)
        calls.extend(drained.calls)
        if drained.normal_text:
            released.append(drained.normal_text)
        remaining = getattr(tool_parser, "_buffer", "")
        if len(remaining) >= len(held):
            break
        held = remaining

    result = tool_parser.finish(tools)
    calls.extend(result.calls)
    held = getattr(tool_parser, "_buffer", "")

    unterminated = 0
    for remainder in (result.normal_text, held):
        if not remainder:
            continue
        if tool_parser.has_tool_call(remainder):
            unterminated += len(remainder)
        released.append(remainder)

    if held:
        # The buffer's bytes are being released here, so consume them: the
        # executor can present the same finished output more than once, and a
        # buffer left in place would release the same remainder into a second
        # message item on the repeat. Dropping never had this problem - a
        # drop repeated is still nothing.
        tool_parser._buffer = ""

    if unterminated:
        logger.warning(
            f"Stream ended inside a tool call; releasing {unterminated} "
            f"characters of unterminated {type(tool_parser).__name__} markup "
            "as the assistant's message, matching the final response's "
            "fallback for a call it cannot parse.")

    # `current_tool_name_sent` is the parser's own record of having announced a
    # call it has not yet closed, and `current_tool_id` numbers that call.
    # Read through getattr because a parser that overrides
    # parse_streaming_increment entirely need not maintain them; not
    # maintaining them means it never reports a half-started call, which is
    # the safe reading.
    unfinished = None
    if getattr(tool_parser, "current_tool_name_sent", False):
        unfinished = getattr(tool_parser, "current_tool_id", None)

    return "".join(released), calls, unfinished


def _effective_tool_parser(tool_parser_id: Optional[str],
                           request) -> Optional[str]:
    """The tool parser this request may run, honouring ``tool_choice="none"``.

    "none" is the one tool_choice value that can be enforced without
    falsifying anything the model generated: not parsing leaves any tool-call
    markup in the visible text verbatim, so nothing is truncated or invented -
    where "required" or a named function would need sampling-level
    constraints, which this path does not apply. Both the streaming event
    generator and the final rebuild resolve their parser through here, because
    the two views of one generation must agree on whether a call exists.

    Read with getattr for the same reason reasoning_chat_template_kwargs
    does: unit tests drive the event generator with stand-in request objects.
    """
    if getattr(request, "tool_choice", None) == "none":
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


def _create_output_content(
    final_res: RequestOutput,
    reasoning_parser: Optional[str] = None,
    tool_parser: Optional[str] = None,
    tools: Optional[list[Tool]] = None,
    chat_template_kwargs: Optional[dict[str, Any]] = None,
    streamed_tool_call_ids: Optional[list[Optional[Tuple[str, str]]]] = None,
    streamed_item_ids: Optional[list[Tuple[str, str]]] = None,
) -> Tuple[list[ResponseOutputItem], list[ChatCompletionMessageParam],
           list[str]]:
    output_items: list[ResponseOutputItem] = []
    output_messages: list[ChatCompletionMessageParam] = []
    # What the reasoning parser claimed as reasoning, per output and before
    # stripping, so usage accounting can count its tokens. Kept raw: the
    # whitespace around a reasoning block was generated too.
    reasoning_texts: list[str] = []
    available_tools = _get_chat_completion_function_tools(tools)

    # The (item type, item id) pairs the stream published in
    # `response.output_item.added/done`, in emission order. This rebuild is a
    # second, independent pass over the generated text and used to mint fresh
    # ids for the reasoning and message items it re-derives, so the snapshot
    # named the very items the stream had already announced under different
    # ids - 216 of 221 measured responses - and a client joining streamed
    # items with the snapshot by id saw phantom items. Reuse is positional
    # per item type, first-emitted first, because emission order is the only
    # correspondence the two passes share. None means no stream ran (a
    # non-streaming request) and every id is minted fresh, as before.
    streamed_ids_by_type: dict[str, list[str]] = {
        "reasoning": [],
        "message": []
    }
    for item_type, item_id in streamed_item_ids or []:
        if item_type in streamed_ids_by_type:
            streamed_ids_by_type[item_type].append(item_id)
    used_ids_by_type = {"reasoning": 0, "message": 0}

    def _streamed_or_fresh_id(item_type: str) -> str:
        # Running past the pool means this pass derived items the stream never
        # opened - the two views already diverged structurally (a stream cut
        # off inside a call whose text this pass keeps, say). A fresh id
        # cannot hide that and must not try to: the count mismatch is warned
        # about after the loop rather than silently papered over.
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
            parsed_source = text
            text, calls = _apply_tool_parser(tool_parser, available_tools,
                                             output.index, text, False)
            if calls and streamed_tool_call_ids is None:
                # No stream ran, so no per-entity record says which calls the
                # streamed assembly would have refused - the check the pairing
                # below performs for streamed requests. An empty-argument call
                # is the shape the whole-text parse gets wrong on its own (it
                # reports `{}` for argument markup it could not read), so it
                # is verified against a streamed replay of the same text; see
                # the reconciler for why the two views must agree here.
                text, calls = _verify_empty_calls_against_streaming(
                    tool_parser, available_tools, parsed_source, text, calls)

        text_item = None
        reasoning_item = None
        tool_calls_item = []

        # Reasoning first, then the answer, then any tool calls.
        #
        # `response.output` is ordered, and the model produced the reasoning
        # before the answer it leads to. The streaming path emits the items in
        # that order - reasoning done, then message done, then the function
        # calls (see _generate_streaming_event) - so a snapshot that appended
        # the message first contradicted the stream of the very same
        # generation: a client replaying `output` read the answer before the
        # reasoning that produced it, and one reconstructing a turn from the
        # snapshot fed the model its own thinking as a follow-up to its reply.
        if reasoning_text:
            reasoning_item = ResponseReasoningItem(
                id=_streamed_or_fresh_id("reasoning"),
                summary=[],
                type="reasoning",
                content=[
                    Content(text=reasoning_text.strip(), type="reasoning_text")
                ],
                status=None,
            )
            output_items.append(reasoning_item)

        # Check again after tool parsing to avoid empty text
        if text:
            output_text = ResponseOutputText(
                text=text.strip(),
                annotations=[],
                type="output_text",
                logprobs=None,
            )

            text_item = ResponseOutputMessage(
                id=_streamed_or_fresh_id("message"),
                content=[output_text],
                role="assistant",
                status="completed",
                type="message",
            )

            output_items.append(text_item)

        if calls:
            tool_resolution = _tool_resolution(tools)
            # Reuse the ids the streaming path already published. The stream's
            # record holds one entry per call the parser announced, in
            # announcement order - None where assembly dropped the call for
            # invalid-JSON arguments - and this pass enumerates the same
            # markup in the same order, so when the counts agree each call is
            # paired with its own entity's entry. Pairing over the *filtered*
            # list instead is how a dropped call A used to donate its position:
            # the whole-text parse still reports A (with empty arguments), A
            # consumed the id the stream had published for B, and B's identity
            # in the snapshot matched nothing the client had seen.
            reusable = list(streamed_tool_call_ids or [])
            entity_aligned = len(reusable) == len(calls)
            if reusable and not entity_aligned:
                # Counts disagreeing means the two passes did not see the same
                # calls (a cut-off stream, say), so per-entity pairing is
                # unknowable. Say so, then fall back to reusing the ids that
                # actually reached the wire, in order; placeholders name no
                # wire identity and are worth nothing to a client.
                logger.warning(
                    f"final response rebuilt {len(calls)} tool call(s) but the "
                    f"stream saw {len(reusable)}; ids beyond the streamed ones "
                    f"are new and will not match the client's tool outputs")
                reusable = [ids for ids in reusable if ids is not None]
            tool_calls_item = []
            for index, call in enumerate(calls):
                ids = reusable[index] if index < len(reusable) else (None, None)
                if entity_aligned and ids is None:
                    # This entity is the call the stream dropped: its
                    # arguments never assembled into valid JSON and no id was
                    # published for it. The snapshot must describe the stream,
                    # so it is not delivered here either - the placeholder is
                    # consumed by the drop instead of shifting onto the next
                    # call.
                    logger.warning(
                        f"final response drops the tool call to {call.name!r}: "
                        f"the stream assembled its arguments into invalid JSON "
                        f"and never delivered it, so the snapshot does not "
                        f"deliver it either")
                    continue
                tool_calls_item.append(
                    _tool_call_output_item(call,
                                           tool_resolution,
                                           item_id=ids[0],
                                           call_id=ids[1]))
            output_items.extend(tool_calls_item)

        output_messages.extend(
            _create_output_messages({
                "text_content":
                text_item.content[0].text if text_item else None,
                "reasoning_content":
                reasoning_item.content[0].text if reasoning_item else None,
                "tool_calls":
                tool_calls_item,
            }))

    # Checked against None, not truthiness: an empty list still means a
    # stream ran and published nothing of this type, and a rebuild that then
    # derives such an item is exactly the divergence worth saying out loud.
    if streamed_item_ids is not None:
        for item_type in ("reasoning", "message"):
            streamed = len(streamed_ids_by_type[item_type])
            rebuilt = used_ids_by_type[item_type]
            if streamed != rebuilt:
                logger.warning(
                    f"final response rebuilt {rebuilt} {item_type} item(s) but "
                    f"the stream published {streamed}; the two views of this "
                    f"generation differ structurally and ids beyond the "
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

    The model writes the name back BOTH ways. Measured over one agent session:
    247 calls arrived as `exec` and 34 as `functions.exec`. The two functions
    this replaces keyed only on the qualified spelling, so every bare-named call
    missed the lookup, was classified as a plain function call, and reached the
    client as JSON where a custom tool expects freeform text. The client rejects
    that outright: 154 of 154 such calls came back "aborted", while the 10 that
    happened to arrive qualified all ran. The agent then concluded its execution
    runtime was broken and burned its retries.

    A bare name is registered only when exactly one declared tool answers to it.
    Two namespaces offering the same tool name cannot be told apart from the
    name alone, and routing a call to the wrong namespace is worse than leaving
    it unresolved - the caller reports an unresolved call as it always did, and
    now warns about it.
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
    call_id: Optional[str] = None,
) -> Union[ResponseFunctionToolCall, ResponseCustomToolCall]:
    """Build the output item for one parsed tool call.

    A custom tool is invoked with freeform text, so its call has to be
    reported as a custom tool call carrying that text. Reporting it as a
    function call hands the client JSON where it expects the raw payload,
    and the client rejects the call outright - for apply_patch, with
    "invoked with incompatible payload", which aborts the whole turn.

    Which of the two it is turns on whether the call's name resolves to a
    declared tool, and the model spells that name inconsistently; see
    ``_tool_resolution`` for what that cost before it accepted both spellings.
    """
    name = call.name or ""
    arguments = call.parameters or "{}"
    # A fresh id only when this call has not been reported before. The final
    # response re-derives its output from the generated text, so it must be
    # given back the ids the stream already used; the client keys its tool
    # outputs on those and matches nothing otherwise.
    call_id = call_id or f"call_{_random_uuid()}"

    resolved = (tool_resolution or {}).get(name)
    if resolved is not None:
        namespace, name, is_custom = resolved
    else:
        namespace, is_custom = None, False
        # Previously silent, which is why a whole class of aborted calls went
        # unexplained for as long as it did: a custom tool reported as a
        # function call is rejected by the client, and nothing said so. SGLang
        # logs "Model attempted to call undefined function" at the equivalent
        # point; borrowing that is most of the value of this change.
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


def _ctx_usage_from_result(final_res: GenerationResult) -> Optional[Any]:
    """The context phase's usage, as the handoff left it on the outputs.

    Mirrors the chat path's `_ctx_usage_from_outputs`. An aggregated server has
    no handoff and returns None, which leaves the caller on the local counts.
    """
    for output in getattr(final_res, "outputs", None) or []:
        disaggregated_params = getattr(output, "disaggregated_params", None)
        if disaggregated_params is None:
            continue
        ctx_usage = getattr(disaggregated_params, "ctx_usage", None)
        if ctx_usage is None:
            continue
        if isinstance(ctx_usage, UsageInfo):
            return ctx_usage
        return UsageInfo.model_validate(ctx_usage)
    return None


def _count_reasoning_tokens(
    tokenizer: Optional[TokenizerBase],
    reasoning_texts: list[str],
    output_tokens: int,
) -> int:
    """How many of the generated tokens went into reasoning.

    The engine does not track this. `ThinkingBudgetLogitsProcessor` computes
    the same quantity in token space (thinking_budget.py:93), but it lives in
    the model engine process, only exists when `thinking_token_budget` is set,
    and a LogitsProcessor has no channel back to the result -- so the count
    has to be rebuilt here.

    This re-encodes the text the reasoning parser itself claimed, rather than
    searching `output.token_ids` for the `</think>` token. The marker search
    is cheaper and exact, and it was the first design; it was dropped because
    it has to restate the parser's rules and would silently disagree the
    moment they differ. GLM alone needs three of them: it closes the block
    more than once and only the first close counts, `<tool_call>` ends the
    block implicitly, and a turn that emits neither is reasoning to its last
    character (reasoning_parser.py:369-404, with a measured case of 26,055
    characters and no closing tag at all). Deriving the count from the
    parser's own output cannot drift from it, and it stays correct for the
    parsers that interleave reasoning with content, where no single prefix
    length describes the split.

    The cost is one encode of the reasoning text per response, off the
    engine's critical path, and a boundary that can differ by a token from
    the generated one -- re-encoding a substring need not reproduce the
    tokenization it came from. Returns 0 when it cannot be determined at all
    (no tokenizer, or no reasoning parser configured), which is what this
    reported unconditionally before.
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

    # Re-encoding can overshoot, and usage that claims more reasoning tokens
    # than were generated is visibly wrong to a client budgeting a context
    # window. Clamping is the conservative direction.
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

    # Under disaggregated serving the counts above describe the generation
    # worker, and from there the entire prompt arrived as KV transferred from
    # the context worker -- so its cached_tokens equals the prompt length and
    # every response claims a complete cache hit. The context phase is the only
    # place that knows what was really reused, and it sends its own usage along
    # with the handoff. Chat completions has taken it from there since #14177;
    # this route was reading the generation worker's numbers instead.
    ctx_prompt_tokens, ctx_cached_tokens = get_usage_tokens_from_ctx(
        _ctx_usage_from_result(final_res))
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
    streamed_tool_call_ids: Optional[list[Optional[Tuple[str, str]]]] = None,
    streamed_item_ids: Optional[list[Tuple[str, str]]] = None,
    tokenizer: Optional[TokenizerBase] = None,
) -> tuple[ResponsesResponse, list[Message | ChatCompletionMessageParam]]:
    _responses_debug_log("================================================")
    _responses_debug_log("RAW MODEL OUTPUT:")
    _responses_debug_log(final_res.outputs)
    _responses_debug_log("================================================")

    # prepare responses output
    output_content = []
    if use_harmony:
        output_content, output_messages = _create_output_content_harmony(
            final_res)
        # Harmony carries its reasoning on the `analysis` channel, which the
        # adapter already tracks per token (harmony_adapter.py:207). Counting
        # it is a separate change on that path; this one leaves it reported
        # as zero rather than half-counting it here.
        reasoning_texts = []
    else:
        output_content, output_messages, reasoning_texts = _create_output_content(
            final_res,
            reasoning_parser,
            # tool_choice="none" bypasses the parser here exactly as the
            # streaming events do, so both views keep the markup as text.
            _effective_tool_parser(tool_parser, request),
            request.tools,
            chat_template_kwargs=reasoning_chat_template_kwargs(request),
            streamed_tool_call_ids=streamed_tool_call_ids,
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
    # A generation the engine cut off at its token budget has status
    # "incomplete", and the spec explains that status in
    # incomplete_details.reason - "max_output_tokens". The streaming terminal
    # event has said so since the response.incomplete fix; setting it here
    # makes the non-streaming JSON body tell the same story. Only "length" is
    # explained: "not_finished" (a disaggregated context worker handing the
    # request off) also maps to "incomplete", and naming a public reason for
    # an internal handoff would be a guess presented as fact.
    if finish_reason == "length":
        response.incomplete_details = IncompleteDetails(
            reason="max_output_tokens")
    # Disaggregated serving. A context-only response is read by the
    # orchestrator, never by a client: it carries the KV-cache handle, the
    # first generated token and the tokenized prompt so a generation worker can
    # pick the request up. Everything else - an aggregated server, or the
    # generation response that does reach the client - leaves these unset.
    # ctx_info_endpoint in particular names an internal address, and
    # prompt_token_ids would add the whole prompt back to every reply.
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
    streamed_tool_call_ids: Optional[list[Optional[Tuple[str, str]]]] = None,
    streamed_item_ids: Optional[list[Tuple[str, str]]] = None,
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
        streamed_tool_call_ids=streamed_tool_call_ids,
        streamed_item_ids=streamed_item_ids,
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
    streamed_tool_call_ids: Optional[list[Optional[Tuple[str, str]]]] = None,
    streamed_item_ids: Optional[list[Tuple[str, str]]] = None,
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
        streamed_tool_call_ids=streamed_tool_call_ids,
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
    emitted_tool_calls: int = 0
    text_buffer: str = ""
    reasoning_buffer: str = ""

    def __init__(self) -> None:
        # Tool-call fragments from the incremental parser, keyed by output
        # index, then by the parser's tool index. One parser instance per
        # output lives in tool_parser_dict, so what those instances report has
        # to be kept apart in the same way.
        #
        # Assigned here rather than in the class body above: a dict written as
        # a class attribute is one object shared by every request, so two
        # concurrent streams would accumulate into each other's calls. The
        # fields above are immutable, so assigning to them rebinds per
        # instance and they are safe as class-level defaults.
        self.tool_call_fragments: dict[int, dict[int, dict[str, Any]]] = {}

        # (item id, call id) of every tool call already streamed, in the order
        # they went out. `response.completed` rebuilds the output from scratch,
        # and _tool_call_output_item mints a fresh random id each time it runs,
        # so without this the final response names every call differently from
        # the events that carried it - and the client's tool outputs, which
        # echo the streamed id, match nothing. Measured before the fix: 671 of
        # 671 calls across three campaigns.
        #
        # One entry per call the parser announced, in announcement order: None
        # holds the place of a call that was assembled but dropped for
        # invalid-JSON arguments. The rebuild enumerates the same markup in
        # the same order, so the placeholder is what lets a drop consume its
        # own position instead of shifting the streamed identities onto the
        # calls behind it.
        #
        # A list here rather than a class attribute, for the reason above.
        self.emitted_tool_call_ids: list[Optional[Tuple[str, str]]] = []

        # (item type, item id) of every reasoning and message item this
        # stream opened, in the order their `output_item.added` went out.
        # The final response re-derives those items from the accumulated text
        # and used to mint fresh ids for them, so the snapshot named the very
        # items the stream had already announced under different ids - 216 of
        # 221 measured responses - and a client joining streamed items with
        # the snapshot by id saw phantom items. Same shape and cure as
        # emitted_tool_call_ids above.
        #
        # A list here rather than a class attribute, for the reason above.
        self.emitted_item_ids: list[Tuple[str, str]] = []


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
    def emitted_tool_calls(self) -> int:
        return self.state_tracker.emitted_tool_calls

    @emitted_tool_calls.setter
    def emitted_tool_calls(self, count: int) -> None:
        self.state_tracker.emitted_tool_calls = count

    @property
    def emitted_tool_call_ids(self) -> list[Optional[Tuple[str, str]]]:
        """(item id, call id) of the tool calls already streamed, in order.

        None entries hold the place of calls the stream assembled and then
        dropped for invalid-JSON arguments; see the state tracker.
        """
        return self.state_tracker.emitted_tool_call_ids

    @property
    def emitted_item_ids(self) -> list[Tuple[str, str]]:
        """(item type, item id) of the reasoning/message items streamed, in order."""
        return self.state_tracker.emitted_item_ids

    def record_emitted_tool_call(self, item) -> None:
        """Remember the ids a streamed call went out under.

        The final response is built by a second, independent pass over the
        generated text; handing it these lets it name the same call the same
        way instead of minting new ids the client has never seen.
        """
        self.state_tracker.emitted_tool_call_ids.append((item.id, item.call_id))

    def record_dropped_tool_call(self) -> None:
        """Hold a dropped call's place in the emission record.

        A call whose assembled arguments were invalid JSON is never streamed,
        but the final rebuild's whole-text parse still reports the entity (the
        markup is well-formed; only the arguments are unusable). Recording the
        drop keeps the record aligned with the parser's announcement order, so
        in the rebuild the drop consumes its own position instead of handing
        the next call an identity the stream published for a different one.
        """
        self.state_tracker.emitted_tool_call_ids.append(None)

    def append_text(self, delta: str) -> None:
        self.state_tracker.text_buffer += delta

    def append_reasoning(self, delta: str) -> None:
        self.state_tracker.reasoning_buffer += delta

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
            sequence_number=-1,  # set by _send_event, restamped at egress
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

            # Remember the id this item is announced under, at the one point
            # where announcing actually happens. The final response is built
            # by a second, independent pass over the generated text; handing
            # it these lets it name the same items the same way instead of
            # minting ids the client has never seen.
            self.state_tracker.emitted_item_ids.append(
                (output_item.type, output_item.id))

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

    `parse_streaming_increment` reports one call in several pieces: the name
    first with empty parameters, then the arguments JSON a fragment at a time,
    then the closing brace. Only the concatenation is something a client can
    run, so the pieces are gathered here and the finished calls are emitted
    when generation ends.

    Keyed by the parser's own `tool_index`, which numbers the calls in the
    order it saw them start; a dict preserves that order.

    The call id is minted when the parser first announces the call - before
    anyone can know whether its arguments will assemble into something
    deliverable - so the identity belongs to the call entity itself. A call
    that is later dropped dies holding its own id instead of donating it to
    the next call in line, which is the positional swap this prevents.
    """
    for call in calls:
        fragment = fragments.setdefault(call.tool_index, {
            "name": None,
            "parameters": "",
            "call_id": f"call_{_random_uuid()}",
        })
        # Only a call's first fragment carries the name; every later one
        # carries None, which must not erase the name already recorded.
        if call.name:
            fragment["name"] = call.name
        fragment["parameters"] += call.parameters or ""


def _assembled_tool_calls(
        fragments: dict[int, dict[str, Any]],
        unfinished_tool_index: Optional[int] = None) -> list[ToolCallItem]:
    r"""The accumulated fragments as whole calls, in the order they started.

    A call whose name never arrived is dropped, which is what
    `parse_base_json` does on the non-streaming path: a call the client cannot
    name is not a call it can run.

    `unfinished_tool_index` names a call the parser announced but never closed,
    because the stream ended inside it. Its accumulated arguments are a JSON
    prefix - `{"input": "\n// Read the problem` and nothing more - which no
    client can parse or run, so it is dropped rather than reported. The
    non-streaming path reports no call at all for the same text, since its
    regex needs the closing tag, and the two endpoints have to agree about
    what the model asked for. Only the *call* is dropped: the raw markup it
    was read from reaches the client as message text on both views
    (`_flush_tool_parser` releases it, the whole-text re-parse falls back to
    it), so nothing the model generated is lost.

    A call whose assembled arguments are not valid JSON is dropped for the same
    reason, one test later. That test keys on the markup being unterminated;
    this one keys on the result being unusable, which is the property that
    actually matters and which covers calls the markup test passes. A GLM-4.7
    block that opens `<arg_value>` and never closes it, ending
    `</think></tool_call>` instead, has a balanced `<tool_call>` pair and so
    finalises normally - with arguments that assemble to `{"cmd": }`. That
    reached a client, which stored it, replayed it in the next request's
    history and was answered `tool_calls[0].function.arguments must be valid
    JSON`; 2 of 13,014 delivered calls, each costing the agent run that hit it.

    Here rather than in the parser because a parser cannot unsay what it has
    already said. It streams `{`, `"cmd": `, the value and `}` over several
    increments, each folded into `fragments` as it arrives, and only discovers
    at `</tool_call>` that the whole does not parse. This is the last point at
    which the fragments are known as a whole *and* declining to report them is
    still possible.

    Dropping beats repairing. Closing the quote and the brace would invent an
    argument the model never wrote, and the client would then run a truncated
    shell command with nothing to indicate anything was lost; executing half a
    command is worse than executing none. The warning is what keeps the loss
    from being silent.

    The non-streaming path cannot reuse this validity test - `parse_base_json`
    builds its arguments with `json.dumps`, so they always parse - and its
    pair regex reads nothing out of such a block, reporting the call with
    empty arguments: a call the model never made, delivered as runnable.
    That is settled now, one layer up: when no stream ran,
    `_verify_empty_calls_against_streaming` replays the text through this
    streaming machinery, and a `{}`-argument call this assembly would have
    refused is not reported there either - its markup falls back into the
    message text instead. What still differs is a call whose markup yielded
    *some* readable pairs: the whole-text parse delivers what it read, the
    stream drops the call whose tail it could not assemble. That is a real
    divergence with no obviously right answer and it stays open, documented
    at the reconciler.
    """
    calls: list[ToolCallItem] = []
    for tool_index, fragment in fragments.items():
        if not fragment["name"] or tool_index == unfinished_tool_index:
            # No warning for the unfinished call: `_flush_tool_parser` has
            # already warned and released its raw markup as message text, and
            # a second warning would read as a second loss.
            continue
        arguments = fragment["parameters"]
        try:
            json.loads(arguments)
        except ValueError as exc:
            # The drop consumes the call's own id (minted when the parser
            # announced it) and the log says which, so the retired identity
            # is auditable and can never be mistaken for the next call's.
            consumed = fragment.get("call_id")
            consumed_note = (f" The drop consumes its call id {consumed!r}, "
                             "which is retired with it." if consumed else "")
            logger.warning(
                f"Dropping the tool call to {fragment['name']!r}: its "
                f"arguments did not assemble into valid JSON ({exc}). The "
                "model emitted a malformed tool call, and reporting it would "
                "hand the client arguments it can only reject. Assembled: "
                f"{arguments[:200]!r}.{consumed_note}")
            continue
        calls.append(
            ToolCallItem(tool_index=tool_index,
                         name=fragment["name"],
                         parameters=arguments))
    return calls


def _verify_empty_calls_against_streaming(
    tool_parser_id: Optional[str],
    tools: Optional[list[ChatCompletionToolsParam]],
    source_text: str,
    normal_text: str,
    calls: list[ToolCallItem],
) -> Tuple[str, list[ToolCallItem]]:
    r"""Drop whole-text `{}`-argument calls whose markup could not be read.

    Only for a request no stream ran for. A streamed request already settles
    this per call entity in `_create_output_content` - the stream's id record
    carries a None where assembly dropped a call, and the rebuild consumes it.

    The whole-text parse reports `{}` for two markups it cannot tell apart: a
    genuine zero-argument call (`<tool_call>get_time</tool_call>`) and a call
    whose argument markup its pair regex read nothing out of - GLM-4.7
    opening an `<arg_value>` it never closes, ending `</think></tool_call>`
    instead. The streaming machinery *can* tell them apart, because it
    assembles exactly the JSON it read and `_assembled_tool_calls` refuses
    what does not parse: the zero-argument call streams as a valid `{}`, the
    unreadable one as `{"cmd": }` and is dropped. So when the whole-text
    parse reports an empty-argument call, the same text is replayed through a
    fresh streaming parser and the two views are paired positionally - the
    same premise the id reuse in `_create_output_content` rests on, both
    passes enumerating the same markup in the same order. An empty call the
    streamed assembly would have refused is a call the model never made, and
    it is not delivered here either.

    The markup is not discarded with it: when the refused call was everything
    the parse found, the original text goes back verbatim - these turns are
    recorded as training data, and the raw markup as message text is the same
    fallback both views already use for an unterminated call. Alongside calls
    that really were made, the surrounding text keeps its parsed form and only
    the refused call is dropped, which is what the streamed view of the same
    text delivers.

    Any disagreement about what the call entities *are* - a replay error, or
    the two passes finding different numbers of calls - leaves the whole-text
    result untouched: this is a cross-check, not the parse of record, and
    guessing at a pairing would trade a known defect for an unknowable one.
    """
    empty_positions = {
        index
        for index, call in enumerate(calls)
        if _parses_to_empty_object(call.parameters)
    }
    if not empty_positions or tool_parser_id is None or tools is None:
        return normal_text, calls

    fragments: dict[int, dict[str, Any]] = {}
    try:
        replay_parser = ToolParserFactory.create_tool_parser(tool_parser_id)
        increment = replay_parser.parse_streaming_increment(source_text, tools)
        _accumulate_tool_call_fragments(fragments, increment.calls)
        _, flushed, unfinished = _flush_tool_parser(
            tools=tools,
            output_index=0,
            tool_parser_dict={0: replay_parser},
        )
        _accumulate_tool_call_fragments(fragments, flushed)
        assembled = _assembled_tool_calls(fragments, unfinished)
    except Exception as exc:  # noqa: BLE001 - cross-check must not fail the parse
        logger.warning(
            f"Could not replay the generation through the streaming tool "
            f"parser to verify its empty-argument calls ({exc}); delivering "
            "the whole-text parse as is.")
        return normal_text, calls

    if len(fragments) != len(calls):
        return normal_text, calls

    delivered = {call.tool_index for call in assembled}
    entity_order = list(fragments)
    phantoms = {
        index
        for index in empty_positions if entity_order[index] not in delivered
    }
    if not phantoms:
        return normal_text, calls

    for index in sorted(phantoms):
        logger.warning(
            f"Dropping the tool call to {calls[index].name!r} from the "
            "whole-text parse: it reports no arguments only because its "
            "argument markup could not be read, and the streamed assembly of "
            "the same text refuses the call. The model emitted a malformed "
            "tool call, and an invented '{}' would run the tool with "
            "arguments the model never wrote.")
    if len(phantoms) == len(calls):
        return source_text, []
    return normal_text, [
        call for index, call in enumerate(calls) if index not in phantoms
    ]


def _parses_to_empty_object(parameters: Optional[str]) -> bool:
    try:
        return json.loads(parameters or "") == {}
    except ValueError:
        return False


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
    # tool_choice="none" is honoured by not parsing at all, so the model's
    # markup stays in the visible text verbatim. The final rebuild resolves
    # its parser the same way (see _create_response); the two views of one
    # generation must agree on whether a call exists.
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

    # End of stream: the tool parser gets one last word before its state is
    # discarded, so bytes it was still withholding are released by a decision
    # rather than lost because nobody ever asked. Only computed here; what it
    # releases is emitted further down, after the call in front of it has
    # closed the item it belongs to.
    flushed_text, flushed_calls, unfinished_tool_index = "", [], None
    if finished_generation:
        flushed_text, flushed_calls, unfinished_tool_index = _flush_tool_parser(
            tools=available_tools,
            output_index=output_idx,
            tool_parser_dict=tool_parser_dict,
        )

    _responses_debug_log(
        repr(
            f" ---------> delta text: {delta_text}, reasoning delta text: {reasoning_delta_text}, calls: {calls}"
        ))

    # Send delta events for ongoing content BEFORE any done events.
    #
    # The close-on-calls block below ends the item that is currently open. If
    # it ran first, this chunk's delta would arrive after that close and open a
    # brand new output item for the tail of the same message - splitting one
    # assistant turn across two items, sometimes mid-word, and clients that
    # render the last item alone show only that fragment. The chat completions
    # path has the same shape: it appends the content delta to the chunk and
    # only then stamps finish_reason.
    #
    # It is also what makes the done payload right in the common case: a chunk
    # that carries the text before a call *and* the start of that call arrives
    # as one delta plus one non-empty `calls`, so the text has to be appended
    # to the item before the item is closed with it.
    #
    # The item must be opened before *any* delta, including a whitespace-only
    # one. Gating the added-events on delta_text.strip() while emitting the
    # delta unconditionally sends output_text.delta with no item open, and a
    # client that keys on the active item drops the whole turn: Codex CLI
    # reports "OutputTextDelta without active item" and prints nothing. Short
    # replies are the ones that hit it, because a leading whitespace token is
    # more likely to be the first delta of the message.
    #
    # get_*_output_added_events is idempotent - it is guarded internally by
    # sent_output_item_added - so calling it for every delta is safe.
    if delta_text:
        # One chunk can carry both halves. The chunk whose raw text spans the
        # closing think tag is parsed into a reasoning part (everything before
        # the tag) and a content part (everything after), and this branch is
        # chosen on the content part alone -- so the reasoning part has to be
        # flushed here or it is never emitted at all. The `elif` below cannot
        # run for this chunk, and nothing else reads reasoning_delta_text.
        #
        # It is not a rare boundary case. The reasoning between the last chunk
        # boundary and the tag is lost every time the two land in one chunk,
        # which measured 56% of reasoning items across four fleets, and 100%
        # of reasoning short enough to fit inside a single chunk. Nothing
        # downstream can recover it either: every done payload is now the sum
        # of the deltas already streamed, so reasoning that was never streamed
        # as a delta is simply not in the reasoning item at all.
        if reasoning_delta_text:
            if streaming_events_helper.is_text_sent:
                yield from _close_open_item(streaming_events_helper)
            if not streaming_events_helper.is_reasoning_sent:
                streaming_events_helper.is_reasoning_sent = True
            yield from streaming_events_helper.get_reasoning_output_added_events(
            )
            streaming_events_helper.append_reasoning(reasoning_delta_text)
            yield streaming_events_helper.get_reasoning_text_delta_event(
                reasoning_delta_text)

        # Reasoning has ended and the answer is starting: close the reasoning
        # item so the message deltas are not attributed to it.
        if streaming_events_helper.is_reasoning_sent:
            yield from _close_open_item(streaming_events_helper)
        if not streaming_events_helper.is_text_sent:
            streaming_events_helper.is_text_sent = True
        yield from streaming_events_helper.get_message_output_added_events()
        streaming_events_helper.append_text(delta_text)
        yield streaming_events_helper.get_text_delta_event(delta_text, [])
    elif reasoning_delta_text:
        if streaming_events_helper.is_text_sent:
            yield from _close_open_item(streaming_events_helper)
        if not streaming_events_helper.is_reasoning_sent:
            streaming_events_helper.is_reasoning_sent = True
        yield from streaming_events_helper.get_reasoning_output_added_events()
        streaming_events_helper.append_reasoning(reasoning_delta_text)
        yield streaming_events_helper.get_reasoning_text_delta_event(
            reasoning_delta_text)

    # A tool call has started, so whichever item is open ends here.
    #
    # Only the incremental parser can report this. It is the one holding back
    # the markup's bytes, so it knows it is mid-call; a re-parse of the
    # accumulated text has no such memory and has to classify an unterminated
    # call as ordinary text. That is exactly how the previous implementation
    # published `<tool_call>functions.exec<arg_key>input</arg_key>...` as the
    # assistant's message: the non-streaming tool regex needs a closing tag,
    # so a call that had closed kept `tool_calls` non-empty while a second,
    # still-open call stayed in the text, and both halves of its condition
    # were true at once. It needed one closed and one open call to coexist,
    # which is why it hit 41.7% of two-call responses and every three-call one
    # while single-call responses were almost untouched.
    #
    # _close_open_item closes with take_text()/take_reasoning() - the deltas
    # already streamed - so a done payload cannot contain anything that was
    # never streamed as a delta, whatever the accumulated text looks like.
    # Which item that is depends on what was open: prose before the call
    # closes a message item, `</think><tool_call>` closes the reasoning item,
    # and a call with nothing in front of it closes nothing and emits no
    # events, rather than inventing an empty message item.
    if calls:
        yield from _close_open_item(streaming_events_helper)

    # Whole calls have to be reassembled from the fragments the parser reports
    # (name, then arguments piece by piece); see _accumulate_tool_call_fragments.
    call_fragments = streaming_events_helper.tool_call_fragments(output_idx)
    _accumulate_tool_call_fragments(call_fragments, calls)
    _accumulate_tool_call_fragments(call_fragments, flushed_calls)

    # Text the parser was still holding when the stream ended, emitted here
    # rather than folded into the delta block above because it sits *after*
    # whatever the last chunk contained. The common shape is a call and the
    # sentence following it arriving in one chunk: the parser returns the call
    # and keeps the sentence, so folding it in would put that sentence in the
    # message item the call just closed - one item reading "Before. After."
    # with the call between them lost from the ordering.
    if flushed_text:
        if streaming_events_helper.is_reasoning_sent:
            yield from _close_open_item(streaming_events_helper)
        streaming_events_helper.is_text_sent = True
        yield from streaming_events_helper.get_message_output_added_events()
        streaming_events_helper.append_text(flushed_text)
        yield streaming_events_helper.get_text_delta_event(flushed_text, [])

    # Close whatever item is still open once generation has finished.
    #
    # Every other close above is triggered by a transition - reasoning into
    # text, text into a tool call - and the last item of a response has no
    # transition after it, so nothing else ever closes it. It would reach the
    # client with no output_text.done, content_part.done or output_item.done,
    # i.e. with no terminal state: Codex CLI renders it but echoes it back on
    # the next turn without a `status`, and ResponseOutputMessageParam
    # requires one, so the following request is rejected outright.
    #
    # The chat completions path has no equivalent problem because it finalises
    # on `output.finish_reason is not None` rather than on parser state. This
    # mirrors that: when generation is finished, any open item is closed.
    if finished_generation and streaming_events_helper.is_output_item_added_sent:
        if streaming_events_helper.is_reasoning_sent:
            reasoning_text = streaming_events_helper.take_reasoning()
            reasoning_item = ResponseReasoningItem(
                id=streaming_events_helper.item_id,
                summary=[],
                type="reasoning",
                content=[Content(text=reasoning_text, type="reasoning_text")],
                status="completed",
            )
            yield streaming_events_helper.get_reasoning_text_done_event(
                reasoning_text)
            yield streaming_events_helper.get_output_item_done_event(
                reasoning_item)
            streaming_events_helper.is_reasoning_sent = False
        else:
            text = streaming_events_helper.take_text()
            text_content = ResponseOutputText(
                text=text,
                annotations=[],
                type="output_text",
                logprobs=None,
            )
            text_item = ResponseOutputMessage(
                id=streaming_events_helper.item_id,
                content=[text_content],
                role="assistant",
                status="completed",
                type="message",
            )
            yield streaming_events_helper.get_text_done_event(text, [])
            yield streaming_events_helper.get_content_part_done_event(
                text_content)
            yield streaming_events_helper.get_output_item_done_event(text_item)
            streaming_events_helper.is_text_sent = False
        streaming_events_helper.output_index_increment()
        streaming_events_helper.is_output_item_added_sent = False

    # Emit the tool calls the parser found, as function_call output items.
    #
    # Without this the call is stripped out of the text by the tool parser and
    # then dropped, so the client receives prose - or, when the whole
    # generation was a tool call, an empty message - and no indication that a
    # tool should run. Codex CLI shows the model announcing an action and then
    # nothing happening at all.
    #
    # TODO(JunyiXu-nv): stream the call items as the parser produces them.
    # They are held back until generation finishes for now. Streaming them is
    # the direction both SGLang and vLLM have gone, but it is an observable
    # protocol change - a client that assumes every call item arrives together
    # at the end would see them spread out instead - and landing it here would
    # mean a reviewer could not tell the leak fix from the protocol change,
    # nor roll back one without the other. The close above is what the leak
    # fix needs, and it already happens incrementally.
    #
    # Emitted after any open item has been closed, so a call item is never
    # nested inside a message item. The counter keeps emission idempotent: it
    # guarded against the old per-chunk re-parse re-reporting the same calls,
    # and now guards against a finished output being presented twice. It
    # counts the parser's announced call entities, kept or dropped, because
    # a drop has to be accounted exactly once too.
    if finished_generation:
        entities = list(call_fragments.items())
        pending = entities[streaming_events_helper.emitted_tool_calls:]
        if pending:
            keep = {
                call.tool_index: call
                for call in _assembled_tool_calls(call_fragments,
                                                  unfinished_tool_index)
            }
            tool_resolution = _tool_resolution(request.tools)
            # Walked in entity order, not kept-list order: a call dropped for
            # invalid-JSON arguments consumes its own slot in the emission
            # record, so the final rebuild pairs each surviving call with the
            # identity the stream actually gave it rather than the one that
            # belonged to the call in front of it. Nameless and unfinished
            # entities take no slot - the rebuild's whole-text re-parse does
            # not report those as calls either, so a placeholder for them
            # would misalign the record.
            for tool_index, fragment in pending:
                call = keep.get(tool_index)
                if call is None:
                    if (fragment.get("name")
                            and tool_index != unfinished_tool_index):
                        streaming_events_helper.record_dropped_tool_call()
                    continue
                tool_call_item = _tool_call_output_item(
                    call,
                    tool_resolution,
                    status="completed",
                    # The id minted when the parser announced this call; see
                    # _accumulate_tool_call_fragments.
                    call_id=fragment.get("call_id"))
                streaming_events_helper.item_id = tool_call_item.id
                yield streaming_events_helper.get_output_item_added_event(
                    tool_call_item)
                yield streaming_events_helper.get_output_item_done_event(
                    tool_call_item)
                streaming_events_helper.record_emitted_tool_call(tool_call_item)
                streaming_events_helper.output_index_increment()
            streaming_events_helper.emitted_tool_calls = len(entities)
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
    finish_reason: Optional[str],
    sequence_number: int = -1,
) -> Union[ResponseCompletedEvent, ResponseIncompleteEvent,
           ResponseFailedEvent]:
    """The terminal event for a stream that ran to completion, by status.

    ``finish_reason_mapping`` marks a generation the engine cut off at its
    token budget as status "incomplete", and the terminal event used to say
    ``response.completed`` around it - an event whose name asserts the one
    thing its payload denies, so a client believed the truncated answer was
    the whole one. The Responses spec pairs the status with its own terminal
    event, ``response.incomplete``, and explains it in
    ``incomplete_details.reason``; "length" is the token budget, spelled
    ``max_output_tokens``, and any other cause is left unstated rather than
    guessed at.

    ``incomplete_details`` now lives on ResponsesResponse and the response
    builder (see ``_create_response``) sets it for a token-budget cut, so the
    dump usually carries it already; setting it here as well keeps this event
    right for any snapshot that did not pass through that builder. The
    dump-level route is the same one ``get_stream_failed_events`` takes for
    ``error``, whose field is still commented out.

    A status of "failed" is a generation the engine gave up on (a timeout),
    and it ships under ``response.failed`` - the spec's terminal event for
    that status - not under a ``response.completed`` whose name promises the
    opposite. "cancelled" stays under ``response.completed`` on purpose: the
    cancellation was client-initiated, the Responses spec pairs no terminal
    event with it, and inventing one would hand SDK clients an event type
    they cannot parse.
    """
    payload = final_response.model_dump()
    if final_response.status == "incomplete":
        if finish_reason == "length":
            payload["incomplete_details"] = {"reason": "max_output_tokens"}
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
        self.sequence_number = 0
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
        # Set sequence_number if the event has this attribute. The number is
        # provisional: with postprocessing workers a pickled copy of this
        # processor builds the per-token frames in another process, counting
        # from zero again, so the frontend restamps every frame at the egress
        # (stamp_sse_sequence_number) and what is written here only has to be
        # ordered within this process.
        if hasattr(event, 'sequence_number'):
            event.sequence_number = self.sequence_number
        self.sequence_number += 1
        # Get event type from the event's type field if it exists
        event_type = getattr(event, 'type', 'unknown')
        return (f"event: {event_type}\n"
                f"data: {event.model_dump_json(indent=None)}\n\n")

    def get_initial_responses(self) -> List[str]:
        initial_response = ResponsesResponse.from_request(
            request=self.request,
            sampling_params=self.sampling_params,
            model_name=self.model_name,
            created_time=self.response_creation_time,
            output=[],
            status="in_progress",
            usage=None,
        ).model_dump()

        resp_created = self._send_event(
            self.streaming_events_helper.get_response_created_event(
                initial_response))
        resp_in_progress = self._send_event(
            self.streaming_events_helper.get_response_in_progress_event(
                initial_response))
        return [resp_created, resp_in_progress]

    async def get_final_response(
        self,
        final_res: RequestOutput,
        num_prompt_tokens: Optional[int] = None,
        tokenizer: Optional[TokenizerBase] = None,
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
            # Name the calls the way the stream already named them.
            streamed_tool_call_ids=self.streaming_events_helper.
            emitted_tool_call_ids,
            # And the reasoning/message items likewise: the snapshot has to
            # answer to the ids the stream already announced.
            streamed_item_ids=self.streaming_events_helper.emitted_item_ids,
            # Taken as an argument rather than held on this object: the
            # postproc-worker path pickles the processor across a process
            # boundary, and that worker already has its own tokenizer
            # (postproc_worker.py:195). Shipping one per request would be
            # pure cost.
            tokenizer=tokenizer,
        )

        return self._send_event(
            _stream_terminal_event(final_response,
                                   final_res.outputs[0].finish_reason))

    def get_final_response_non_store(
        self,
        final_res: RequestOutput,
        num_prompt_tokens: Optional[int] = None,
        tokenizer: Optional[TokenizerBase] = None,
    ) -> str:
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
            # Name the calls the way the stream already named them.
            streamed_tool_call_ids=self.streaming_events_helper.
            emitted_tool_call_ids,
            # And the reasoning/message items likewise: the snapshot has to
            # answer to the ids the stream already announced.
            streamed_item_ids=self.streaming_events_helper.emitted_item_ids,
            # Taken as an argument rather than held on this object: the
            # postproc-worker path pickles the processor across a process
            # boundary, and that worker already has its own tokenizer
            # (postproc_worker.py:195). Shipping one per request would be
            # pure cost.
            tokenizer=tokenizer,
        )

        return self._send_event(
            _stream_terminal_event(final_response,
                                   final_res.outputs[0].finish_reason))

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
        """The terminal events for a stream that stopped before completing.

        Carries no generated content. The accumulated text lives only in the
        deltas already on the wire and re-emitting it here would guess at
        whether those deltas reached anyone; this says what happened, nothing
        more.

        ``events_sent`` is the number of frames that actually reached the
        wire. Normally it equals this processor's own counter, because this
        processor numbered every one of them. With postprocessing workers it
        does not: the per-token events are built by a pickled copy of this
        object in another process and only the opening two are numbered here,
        so the local counter is short by the whole body of the turn. Taking
        the larger of the two keeps the terminal events after everything the
        client has already seen either way.
        """
        self.sequence_number = max(self.sequence_number, events_sent)
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
        ).model_dump()
        # Set on the dump rather than on ResponsesResponse, whose `error` field
        # is commented out. The event re-validates this dict against the SDK's
        # Response, which does carry `error`, so the field reaches the wire
        # without widening the local model - and without adding a key to the
        # happy path's response.completed payload.
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

# Why a stream stopped before `response.completed`. Recorded in the terminal
# `error` event's `code` and passed to the guard's `on_termination` callback.
# Telling a server-side fault apart from a client hangup is what decides
# whether the accumulated text still existed in this process when the stream
# died.
STREAM_TERMINATION_CLIENT_DISCONNECT = "client_disconnect"
STREAM_TERMINATION_ENGINE_ERROR = "engine_error"
STREAM_TERMINATION_UPSTREAM_ERROR = "upstream_error"
STREAM_TERMINATION_INTERNAL_ERROR = "internal_error"

# The terminal events of a stream that ran to completion - a truncated
# generation ends in `response.incomplete`, a failed one in `response.failed`,
# everything else in `response.completed` (see _stream_terminal_event; the
# guard's own abnormal-termination path emits `response.failed` too). Matched
# as substrings because a relayed stream arrives as transport-sized chunks
# that may carry several events, not as one frame per event. Kept in both
# encodings so the scan never has to decode a relayed chunk just to look at
# it.
_COMPLETED_EVENT_MARKER = "event: response.completed"
_INCOMPLETE_EVENT_MARKER = "event: response.incomplete"
_FAILED_EVENT_MARKER = "event: response.failed"
_TERMINAL_EVENT_MARKERS = (_COMPLETED_EVENT_MARKER, _INCOMPLETE_EVENT_MARKER,
                           _FAILED_EVENT_MARKER)
_TERMINAL_EVENT_MARKERS_BYTES = tuple(
    marker.encode("utf-8") for marker in _TERMINAL_EVENT_MARKERS)

_SSE_EVENT_DELIMITER = "\n\n"
_SSE_EVENT_DELIMITER_BYTES = _SSE_EVENT_DELIMITER.encode("utf-8")


def classify_stream_termination(exc: BaseException) -> str:
    """Attribute a stream-ending exception to a cause.

    The distinction that matters is server-side fault vs client hangup. A
    cancelled or closed generator means the consumer went away: on the
    aggregated server that is the HTTP client, on the orchestrator it is the
    client of the orchestrator. Everything else happened on this side of the
    socket, and is split further only as far as the exception type can be
    trusted to say - an engine error means generation itself failed, an
    upstream transport error means the response this process was relaying was
    cut, and anything else is a fault in this process's own code, where the
    accumulated text was still in memory when the stream died.
    """
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        return STREAM_TERMINATION_CLIENT_DISCONNECT
    if isinstance(exc, (RequestError, EngineDeadError)):
        return STREAM_TERMINATION_ENGINE_ERROR
    # By module rather than by class: the HTTP client library is an
    # implementation detail of the relay and importing it here would pull a
    # transport dependency into the protocol layer.
    if type(exc).__module__.split(".")[0] in ("aiohttp", "httpx", "httpcore"):
        return STREAM_TERMINATION_UPSTREAM_ERROR
    return STREAM_TERMINATION_INTERNAL_ERROR


def describe_stream_termination(exc: BaseException) -> str:
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def stream_error_event(cause: str, detail: str,
                       events_sent: int) -> List[bytes]:
    """A bare ``error`` event for a relay that cannot build a snapshot.

    ``response.failed`` carries a whole ``Response``, which the disaggregated
    orchestrator cannot assemble - it forwards bytes and never holds the
    sampling parameters the snapshot is built from. ``error`` is flat, so it
    needs only the number of events already forwarded to keep the sequence
    contiguous. Bytes, because the relay it terminates is a byte stream.
    """
    event = ResponseErrorEvent(
        type="error",
        sequence_number=events_sent,
        code=cause,
        message=detail,
        param=None,
    )
    return [(f"event: error\n"
             f"data: {event.model_dump_json(indent=None)}\n\n").encode("utf-8")]


def stamp_sse_sequence_number(frame: str, sequence_number: int) -> str:
    """Overwrite the ``sequence_number`` inside one serialized SSE frame.

    Numbering cannot be left to the producers, because one response can have
    two of them. With postprocessing workers enabled the opening
    ``response.created``/``response.in_progress`` pair is built by the
    frontend's streaming processor while every later frame comes from a
    pickled copy of that processor in a worker process, each counting from
    zero - so a single response's stream carried 0,1,0,1,2,... (221 of 221
    measured responses). Only the egress point, where the frames converge
    just before leaving the server, can hand out one monotonic sequence; it
    restamps every frame with this and the producers' numbers are treated as
    provisional.

    The frame is parsed rather than patched with a substring replace: the
    ``response.completed`` payload embeds the model's own output, which can
    contain anything - including the literal ``"sequence_number":`` when the
    model writes code against this very API. A frame with no data line, an
    unparsable payload, or no sequence_number field is returned untouched;
    no producer builds such a frame today, and inventing a field on one would
    be worse than leaving its numbering alone.
    """
    head, sep, rest = frame.partition("data: ")
    if not sep:
        return frame
    # The payload is one line: _send_event serializes with indent=None, and
    # JSON strings carry newlines escaped.
    payload_text, newline, tail = rest.partition("\n")
    try:
        payload = json.loads(payload_text)
    except ValueError:
        return frame
    if not isinstance(payload, dict) or "sequence_number" not in payload:
        return frame
    payload["sequence_number"] = sequence_number
    # Compact separators and raw unicode, matching model_dump_json, so the
    # restamped frame differs from the produced one only in the number.
    restamped = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return f"{head}data: {restamped}{newline}{tail}"


def _count_frames(chunk: Any, carry: Any) -> Tuple[int, Any]:
    """Count completed SSE events in ``chunk``, tolerating split delimiters.

    The carry is the chunk's last byte/character: a transport read can land
    between the two newlines that end an event, and without it that event is
    never counted and every sequence number after it is one short.
    """
    if isinstance(chunk, bytes):
        text = (carry or b"") + chunk
        return text.count(_SSE_EVENT_DELIMITER_BYTES), text[-1:]
    text = (carry or "") + chunk
    return text.count(_SSE_EVENT_DELIMITER), text[-1:]


async def guard_responses_stream(
    stream: AsyncGenerator[Any, None],
    terminal_events: Callable[[str, str, int], List[Any]],
    on_termination: Optional[Callable[[str, str], None]] = None,
) -> AsyncGenerator[Any, None]:
    """Make a Responses stream say so when it stops before completing.

    Without this such a stream simply ends: the generator raises, the ASGI
    server abandons a half-written chunked body, and nothing in the bytes
    distinguishes that from a stream still in flight. Everything the turn
    produced survives only as deltas, because only the terminal event repeats
    the full text.

    Frames are forwarded untouched and nothing is added once a terminal event
    (``response.completed``, ``response.incomplete`` for a truncated
    generation, or ``response.failed`` for one the engine gave up on) has
    gone out, so a stream that completes normally is byte-for-byte what it
    was.

    The two abnormal endings are handled differently on purpose:

    * A **fault on this side** yields the terminal events and then re-raises,
      so the client gets bytes that explain the truncation and the server
      still records an error rather than a clean finish.
    * A **client hangup** yields nothing. There is nobody left to send to, and
      a generator that yields while ``GeneratorExit`` is propagating raises
      ``RuntimeError: async generator ignored GeneratorExit`` - trading a
      diagnosable truncation for an undiagnosable one. The cause is still
      reported through ``on_termination`` when a caller supplies one.
    """
    events_sent = 0
    carry = None
    completed = False
    try:
        async for chunk in stream:
            if not completed:
                frames, carry = _count_frames(chunk, carry)
                events_sent += frames
                markers = (_TERMINAL_EVENT_MARKERS_BYTES if isinstance(
                    chunk, bytes) else _TERMINAL_EVENT_MARKERS)
                completed = any(marker in chunk for marker in markers)
            yield chunk
    except (asyncio.CancelledError, GeneratorExit) as exc:
        if on_termination is not None:
            on_termination(STREAM_TERMINATION_CLIENT_DISCONNECT,
                           describe_stream_termination(exc))
        raise
    except Exception as exc:
        cause = classify_stream_termination(exc)
        detail = describe_stream_termination(exc)
        if on_termination is not None:
            on_termination(cause, detail)
        if not completed:
            logger.error("Responses stream terminated before completion "
                         f"({cause}): {detail}")
            # Built before yielding, and inside its own guard: this runs only
            # when something has already gone wrong, and a failure here would
            # replace the exception the caller needs to see with one about
            # reporting it.
            try:
                frames = terminal_events(cause, detail, events_sent)
            except Exception as report_error:  # noqa: BLE001
                logger.error(
                    f"Failed to build the terminal event: {report_error}")
                frames = []
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
    tokenizer: Optional[TokenizerBase] = None,
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

    final_response = await streaming_processor.get_final_response(
        final_res, tokenizer=tokenizer)

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


def _sse_event(event: StreamingResponsesResponse) -> bytes:
    return (f"event: {event.type}\n"
            f"data: {event.model_dump_json(indent=None)}\n\n").encode("utf-8")


async def responses_done_generator(
        response: ResponsesResponse) -> AsyncGenerator[bytes, None]:
    """Stream an already-complete ResponsesResponse as a well-formed SSE run.

    Used when a request finishes without ever reaching a generation worker.
    The Responses protocol has no ``[DONE]`` sentinel - a client watches for
    a terminal event - so emitting the completions-style terminator here
    would leave a streaming client waiting for an event that never arrives.

    The terminal event is chosen by the response's status, like every other
    completed stream (_stream_terminal_event): a replayed response that was
    cut off at its token budget must end in ``response.incomplete``, not in a
    ``response.completed`` that contradicts the status it carries.
    """
    payload = response.model_dump()
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
    # finish_reason is carried on context-only responses, which is the only
    # kind this generator replays; None simply leaves the reason unstated.
    yield _sse_event(
        _stream_terminal_event(response,
                               response.finish_reason,
                               sequence_number=2))


UCompletionResponseOrGenerator = Union[UCompletionResponse,
                                       AsyncGenerator[Any, None]]
