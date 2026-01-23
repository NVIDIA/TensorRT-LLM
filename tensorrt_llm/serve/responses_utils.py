# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import os
import time
import uuid
# yapf: disable
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from copy import copy
from typing import Any, List, Literal, Optional, OrderedDict, Tuple, Union

from openai.types.responses import (ResponseCompletedEvent,
                                    ResponseContentPartAddedEvent,
                                    ResponseContentPartDoneEvent,
                                    ResponseCreatedEvent,
                                    ResponseFunctionToolCall,
                                    ResponseInProgressEvent, ResponseOutputItem,
                                    ResponseOutputItemAddedEvent,
                                    ResponseOutputItemDoneEvent,
                                    ResponseOutputMessage, ResponseOutputText,
                                    ResponseReasoningItem,
                                    ResponseReasoningTextDeltaEvent,
                                    ResponseReasoningTextDoneEvent,
                                    ResponseTextDeltaEvent,
                                    ResponseTextDoneEvent)
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

from tensorrt_llm.bindings import steady_clock_now
from tensorrt_llm.executor import GenerationResult
from tensorrt_llm.inputs.utils import apply_chat_template
from tensorrt_llm.llmapi import SamplingParams
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.llmapi.reasoning_parser import (BaseReasoningParser,
                                                  ReasoningParserFactory)
from tensorrt_llm.llmapi.tokenizer import TokenizerBase, TransformersTokenizer
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines
from tensorrt_llm.serve.openai_protocol import (ChatCompletionMessageParam,
                                                ChatCompletionToolsParam,
                                                FunctionDefinition,
                                                OpenAIBaseModel,
                                                ReasoningAssistantMessage,
                                                ResponseInputOutputItem,
                                                ResponsesRequest,
                                                ResponsesResponse,
                                                StreamingResponsesResponse,
                                                UCompletionRequest,
                                                UCompletionResponse)
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.core_types import ToolCallItem
from tensorrt_llm.serve.tool_parser.tool_parser_factory import ToolParserFactory

from .harmony_adapter import HarmonyAdapter, get_harmony_adapter

# yapf: enable

# yapf: enable

REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}

ENABLE_RESPONSES_DEBUG_MSG = False


def _responses_debug_log(msg):
    if ENABLE_RESPONSES_DEBUG_MSG:
        logger.info(msg)


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


def get_steady_clock_now_in_seconds() -> float:
    return steady_clock_now().total_seconds()


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
        """
        Store the response and its messages(model output messages) in the conversation store. If the previous response id is provided,
        the messages will be appended to the conversation. Otherwise, a new conversation will be created.

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
                while len(self.conversations[conversation_id]
                          ) > self.conversation_capacity:
                    self._pop_conversation(resp_id)
            else:
                conversation_id = _random_uuid()
                self.conversations[conversation_id] = resp_msgs

            _responses_debug_log(
                f" * storing at conversation id: {conversation_id}")

            self.response_to_conversation[resp_id] = conversation_id
            self.conversation_to_response[conversation_id] = resp_id
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
            if len(self.conversations[conversation_id]
                   ) > self.conversation_capacity:
                self._pop_conversation(resp_id)

            self.response_to_conversation[resp_id] = conversation_id
            self.conversation_to_response[conversation_id] = resp_id
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
        """
        Update the visited conversation to the front of the conversation store.
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
        """
        Pop the oldest conversation messages from a conversation.
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

        conversation = self.conversations[conversation_id]
        if len(conversation) == 0:
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

    raise RuntimeError("Should never reach here!")


def _response_output_item_to_chat_completion_message(
        item: Union[dict,
                    ResponseInputOutputItem]) -> ChatCompletionMessageParam:
    if not isinstance(item, dict):
        item = item.model_dump()

    item_type = item.get("type", "")

    match item_type:
        case "":
            if "role" in item:
                return item
            else:
                raise ValueError(f"Invalid input message item: {item}")
        case "message":
            return {
                "role": "assistant",
                "content": item["content"][0]["text"],
            }
        case "reasoning":
            return {
                "role": "assistant",
                "reasoning": item["content"][0]["text"],
            }
        case "function_call":
            return {
                "role": "function",
                "content": item["arguments"],
            }
        case "function_call_output":
            return {
                "role": "tool",
                "content": item["output"],
                "tool_call_id": item["call_id"],
            }
        case _:
            raise ValueError(
                f"Unsupported input item type: {item_type}, item: {item}")


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
    # Skip the reasoning output.
    for msg in prev_msgs:
        if "reasoning" not in msg:
            messages.append(msg)

    # Append the new input.
    # Responses API supports simple text inputs without chat format.
    if isinstance(request.input, str):
        messages.append({"role": "user", "content": request.input})
    else:
        for inp in request.input:
            messages.append(
                _response_output_item_to_chat_completion_message(inp))

    return messages


def _create_output_messages(
        output_contents: dict[str, Any]) -> list[ChatCompletionMessageParam]:
    """
    Convert output contents to chat completion messages for conversation store.

    Reasoning content is not included in the output messages to reduce the token usage.

    Input:
        output_contents: dict[str, str]
        - text_content: Optional[str]
        - reasoning_content: Optional[str]
        - tool_calls: Optional[list[ToolCall]]

    Returns:
        list[ChatCompletionMessageParam]: Chat completion messages for conversation store.
    """
    messages: list[ChatCompletionMessageParam] = []

    text_content = output_contents.get("text_content", None)
    if text_content:
        messages.append({
            "role": "assistant",
            "content": text_content,
        })

    reasoning_content = output_contents.get("reasoning_content", None)
    if reasoning_content:
        reasoning_msg = ReasoningAssistantMessage(
            role="assistant",
            reasoning=reasoning_content,
        )

        tool_calls = output_contents.get("tool_calls", [])
        tool_call_msgs = [{
            "id": call.call_id,
            "function": {
                "arguments": call.arguments,
                "name": call.name,
            },
            "type": "function",
        } for call in tool_calls]

        _responses_debug_log(f"tool_call_msgs: {tool_call_msgs}")
        reasoning_msg["tool_calls"] = tool_call_msgs

        messages.append(reasoning_msg)

    return messages


def _get_chat_completion_function_tools(
        tools: Optional[list[Tool]]) -> list[ChatCompletionToolsParam]:
    function_tools: list[ChatCompletionToolsParam] = []
    if tools is None:
        return function_tools

    for tool in tools:
        if isinstance(tool, FunctionTool):
            function_tools.append(
                ChatCompletionToolsParam(
                    type="function",
                    function=FunctionDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters,
                    ),
                ))
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

    conversation, mm_coroutines, mm_placeholder_counts = parse_chat_messages_coroutines(
        messages, model_config)
    mm_data = await mm_coroutines

    tools_dict = [
        tool.model_dump()
        for tool in _get_chat_completion_function_tools(request.tools)
    ]
    token_ids = apply_chat_template(
        model_type=model_config.model_type,
        tokenizer=tokenizer,
        processor=processor,
        conversation=conversation,
        add_generation_prompt=True,
        tools=tools_dict,
        mm_placeholder_counts=mm_placeholder_counts,
        enable_tokenize=True,
    )

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

    if use_harmony:
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

    _responses_debug_log("======= Complete Inputs to model =======")
    _responses_debug_log(_decode_tokens(input_tokens, tokenizer))
    _responses_debug_log("========================================")
    return input_tokens, sampling_params


# TODO(JunyiXu-nv): move to use the same function in postprocess_handlers after multiple post processors are supported
def _apply_reasoning_parser(
    reasoning_parser_id: Optional[str],
    output_index: int,
    text: str,
    streaming: bool,
    reasoning_parser_dict: Optional[dict[int, BaseReasoningParser]] = None,
) -> Tuple[str, str]:
    reasoning_parser: Optional[BaseReasoningParser] = None
    if reasoning_parser_id is not None:
        if reasoning_parser_dict is not None:
            if output_index not in reasoning_parser_dict:
                reasoning_parser_dict[
                    output_index] = ReasoningParserFactory.create_reasoning_parser(
                        reasoning_parser_id)

            reasoning_parser = reasoning_parser_dict[output_index]
        else:
            reasoning_parser = ReasoningParserFactory.create_reasoning_parser(
                reasoning_parser_id)

    if reasoning_parser is not None:
        if not streaming:
            result = reasoning_parser.parse(text)
        else:
            result = reasoning_parser.parse_delta(text)
        content, reasoning_content = result.content, result.reasoning_content
    else:
        content, reasoning_content = text, ""

    return content, reasoning_content


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
    else:
        normal_text, calls = text, []

    return normal_text, calls


def _create_output_content(
    final_res: RequestOutput,
    reasoning_parser: Optional[str] = None,
    tool_parser: Optional[str] = None,
    tools: Optional[list[Tool]] = None,
) -> Tuple[list[ResponseOutputItem], list[ChatCompletionMessageParam]]:
    output_items: list[ResponseOutputItem] = []
    output_messages: list[ChatCompletionMessageParam] = []
    available_tools = _get_chat_completion_function_tools(tools)

    for output in final_res.outputs:
        calls = []
        text, reasoning_text = _apply_reasoning_parser(reasoning_parser,
                                                       output.index,
                                                       output.text, False)

        if text:
            text, calls = _apply_tool_parser(tool_parser, available_tools,
                                             output.index, text, False)

        text_item = None
        reasoning_item = None
        tool_calls_item = []
        # Check again after tool parsing to avoid empty text
        if text:
            output_text = ResponseOutputText(
                text=text.strip(),
                annotations=[],
                type="output_text",
                logprobs=None,
            )

            text_item = ResponseOutputMessage(
                id=f"msg_{_random_uuid()}",
                content=[output_text],
                role="assistant",
                status="completed",
                type="message",
            )

            output_items.append(text_item)

        if reasoning_text:
            reasoning_item = ResponseReasoningItem(
                id=f"rs_{_random_uuid()}",
                summary=[],
                type="reasoning",
                content=[
                    Content(text=reasoning_text.strip(), type="reasoning_text")
                ],
                status=None,
            )
            output_items.append(reasoning_item)

        if calls:
            tool_calls_item = [
                ResponseFunctionToolCall(
                    arguments=call.parameters,
                    call_id=f"call_{_random_uuid()}",
                    name=call.name,
                    type="function_call",
                    id=f"fc_{_random_uuid()}",
                ) for call in calls
            ]
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

    return output_items, output_messages


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


def _create_response(
    final_res: GenerationResult,
    use_harmony: bool,
    request: ResponsesRequest,
    model_name: str,
    response_creation_time: int,
    sampling_params: SamplingParams,
    reasoning_parser: Optional[str] = None,
    tool_parser: Optional[str] = None,
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
    else:
        output_content, output_messages = _create_output_content(
            final_res, reasoning_parser, tool_parser, request.tools)

    response = ResponsesResponse.from_request(
        request=request,
        sampling_params=sampling_params,
        model_name=model_name,
        created_time=response_creation_time,
        output=output_content,
        status=finish_reason_mapping(final_res.outputs[0].finish_reason),
    )

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


class ResponsesStreamingEventsHelper:

    def __init__(self):
        self.state_tracker = ResponsesStreamingStateTracker()

    def content_index_increment(self):
        self.state_tracker.current_content_index += 1

    def output_index_increment(self):
        self.state_tracker.current_output_index += 1

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
        """
        Get item added event and content part added event for a message item which is starting
        to be generated.

        Returns:
            list[StreamingResponsesResponse]: A list of streaming responses responses
        """
        if not self.is_output_item_added_sent:
            self.is_output_item_added_sent = True

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

    def get_message_output_added_events(
            self) -> list[StreamingResponsesResponse]:
        return self._get_output_added_events(output_item=ResponseOutputMessage(
            id=self.item_id,
            type="message",
            role="assistant",
            content=[],
            status="in_progress",
        ))

    def get_reasoning_output_added_events(
            self) -> list[StreamingResponsesResponse]:
        return self._get_output_added_events(output_item=ResponseReasoningItem(
            id=self.item_id,
            type="reasoning",
            summary=[],
            status="in_progress",
        ))


def _should_send_done_events(
    output: RequestOutput,
    output_index: int,
    reasoning_parser_id: Optional[str] = None,
    tool_parser_id: Optional[str] = None,
    tools: Optional[list[Tool]] = None,
    reasoning_parser_dict: Optional[dict[int, BaseReasoningParser]] = None,
    tool_parser_dict: Optional[dict[int, BaseToolParser]] = None,
    streaming_events_helper: Optional[ResponsesStreamingEventsHelper] = None,
    finished_generation: bool = False,
) -> Tuple[bool, bool, Optional[str], Optional[str]]:
    """
    Determine if done events should be sent for text or reasoning items.

    Analyzes the complete output text to detect when reasoning or text sections
    have been completed and should receive done events.

    Args:
        output: RequestOutput containing full generated text in output.text
        output_index: Index of the output being processed
        reasoning_parser_id: Parser ID for extracting reasoning content
        tool_parser_id: Parser ID for extracting tool calls
        tools: Available tools for tool parsing
        reasoning_parser_dict: Dictionary of reasoning parsers
        tool_parser_dict: Dictionary of tool parsers
        streaming_events_helper: Helper tracking current streaming state

    Returns:
        Tuple of (should_send_reasoning_done, should_send_text_done,
                  reasoning_content, text_content)
    """
    should_send_reasoning_done = False
    should_send_text_done = False
    reasoning_content = ""
    text_content = ""

    # TODO(JunyiXu-nv): find a more efficient way to decide if we need to send done events
    # Parse complete output using non-streaming mode to get full content
    full_text, full_reasoning = _apply_reasoning_parser(
        reasoning_parser_id=reasoning_parser_id,
        output_index=output_index,
        text=output.text,
        streaming=False,
        reasoning_parser_dict=reasoning_parser_dict,
    )

    # Apply tool parsing to get tool calls
    tool_calls = []
    if full_text:
        full_text, tool_calls = _apply_tool_parser(
            tool_parser_id=tool_parser_id,
            tools=tools,
            output_index=output_index,
            text=full_text,
            streaming=False,
            tool_parser_dict=tool_parser_dict,
        )

    # Detect reasoning -> text transition
    # Reasoning is done when we have sent reasoning content and now have text content
    if full_reasoning and full_text:
        if streaming_events_helper and streaming_events_helper.is_reasoning_sent and not streaming_events_helper.is_text_sent:
            should_send_reasoning_done = True
            reasoning_content = full_reasoning

    # Detect text -> tool call transition
    # Text is done when we have sent text content and now have tool calls
    if full_text and tool_calls:
        if streaming_events_helper and streaming_events_helper.is_text_sent:
            should_send_text_done = True
            text_content = full_text

    # Also check if text is done because generation finished (no tool calls case)
    # Text is done when generation completes and we've sent text
    if full_text and not tool_calls and finished_generation:
        if streaming_events_helper and streaming_events_helper.is_text_sent:
            should_send_text_done = True
            text_content = full_text

    # Similarly, reasoning is done if generation finished with only reasoning (no text case)
    if full_reasoning and not full_text and finished_generation:
        if streaming_events_helper and streaming_events_helper.is_reasoning_sent:
            should_send_reasoning_done = True
            reasoning_content = full_reasoning

    return should_send_reasoning_done, should_send_text_done, reasoning_content, text_content


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
    )

    if delta_text:
        # TODO(JunyiXu-nv): handle tool calls in streaming mode
        delta_text, calls = _apply_tool_parser(
            tool_parser_id=tool_parser_id,
            tools=available_tools,
            output_index=output_idx,
            text=delta_text,
            streaming=True,
            tool_parser_dict=tool_parser_dict,
        )

    _responses_debug_log(
        repr(
            f" ---------> delta text: {delta_text}, reasoning delta text: {reasoning_delta_text}, calls: {calls}"
        ))

    # Check if we need to send done events for completed sections
    should_send_reasoning_done, should_send_text_done, reasoning_full_content, text_full_content = _should_send_done_events(
        output=output,
        output_index=output_idx,
        reasoning_parser_id=reasoning_parser_id,
        tool_parser_id=tool_parser_id,
        tools=available_tools,
        reasoning_parser_dict=reasoning_parser_dict,
        tool_parser_dict=tool_parser_dict,
        streaming_events_helper=streaming_events_helper,
        finished_generation=finished_generation,
    )

    # Send done events if needed
    if should_send_reasoning_done and reasoning_full_content:
        reasoning_item = ResponseReasoningItem(
            id=streaming_events_helper.item_id,
            summary=[],
            type="reasoning",
            content=[
                Content(text=reasoning_full_content, type="reasoning_text")
            ],
            status="completed",
        )
        yield streaming_events_helper.get_reasoning_text_done_event(
            reasoning_full_content)
        yield streaming_events_helper.get_output_item_done_event(reasoning_item)
        streaming_events_helper.output_index_increment()
        streaming_events_helper.is_output_item_added_sent = False
        streaming_events_helper.is_reasoning_sent = False

    if should_send_text_done and text_full_content:
        text_content = ResponseOutputText(
            text=text_full_content,
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
        yield streaming_events_helper.get_text_done_event(text_full_content, [])
        yield streaming_events_helper.get_content_part_done_event(text_content)
        yield streaming_events_helper.get_output_item_done_event(text_item)
        streaming_events_helper.output_index_increment()
        streaming_events_helper.is_output_item_added_sent = False
        streaming_events_helper.is_text_sent = False

    # Send delta events for ongoing content
    if delta_text:
        if delta_text.strip():
            if not streaming_events_helper.is_text_sent:
                streaming_events_helper.is_text_sent = True
            yield from streaming_events_helper.get_message_output_added_events()
        yield streaming_events_helper.get_text_delta_event(delta_text, [])
    elif reasoning_delta_text:
        if reasoning_delta_text.strip():
            if not streaming_events_helper.is_reasoning_sent:
                streaming_events_helper.is_reasoning_sent = True
            yield from streaming_events_helper.get_reasoning_output_added_events(
            )
        yield streaming_events_helper.get_reasoning_text_delta_event(
            reasoning_delta_text)


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
        # Set sequence_number if the event has this attribute
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
        )

        return self._send_event(
            ResponseCompletedEvent(
                type="response.completed",
                sequence_number=-1,
                response=final_response.model_dump(),
            ))

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

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Add arrival time to scope
            scope["state"] = {}
            scope["state"][
                "server_arrival_time"] = get_steady_clock_now_in_seconds()

        # Pass through the original receive/send - no wrapping!
        await self.app(scope, receive, send)


class ResponseHooks(ABC):
    """
    Hooks for response processing and (disagg) service perf observability.
    """

    @abstractmethod
    def on_req_begin(self, request: UCompletionRequest):
        pass

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


UCompletionResponseOrGenerator = Union[UCompletionResponse,
                                       AsyncGenerator[Any, None]]
