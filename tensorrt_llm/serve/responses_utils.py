# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncGenerator
from copy import copy
from typing import Literal, Optional, OrderedDict, Union

# yapf: disable
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
# yapf: enable
from openai.types.responses.response_function_web_search import (
    ActionFind, ActionOpenPage, ActionSearch, ResponseFunctionWebSearch)
from openai.types.responses.response_reasoning_item import Content
from openai.types.responses.tool import Tool
from openai_harmony import (Author, Conversation, DeveloperContent,
                            HarmonyEncodingName, Message, ReasoningEffort, Role,
                            StreamState, SystemContent, TextContent,
                            ToolDescription, load_harmony_encoding)

from tensorrt_llm.bindings import steady_clock_now
from tensorrt_llm.llmapi import SamplingParams
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import (OpenAIBaseModel,
                                                ResponseInputOutputItem,
                                                ResponsesRequest,
                                                ResponsesResponse)

from .harmony_adapter import HarmonyAdapter

REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}

ENABLE_RESPONSES_DEBUG_MSG = False


def responses_debug_log(msg):
    if ENABLE_RESPONSES_DEBUG_MSG:
        logger.debug(msg)


_harmony_encoding = None


def random_uuid():
    return str(uuid.uuid4().hex)


def get_encoding():
    global _harmony_encoding
    if _harmony_encoding is None:
        _harmony_encoding = load_harmony_encoding(
            HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_encoding


def decode_tokens(tokens):
    return get_encoding().decode(tokens)


def get_steady_clock_now_in_seconds() -> float:
    return steady_clock_now().total_seconds()


def parse_response_input(
    input_msg: ResponseInputOutputItem,
    prev_responses: list[Union[ResponseOutputItem, ResponseReasoningItem]]
) -> Message:
    if not isinstance(input_msg, dict):
        input_msg = input_msg.model_dump()

    responses_debug_log(f"------- Parsing input -----------")
    responses_debug_log(input_msg)
    responses_debug_log("")

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
        self.response_capacity = resp_capacity
        self.conversation_capacity = resp_capacity * 4
        self.max_conversations = max_conversations

        self.responses_lock = asyncio.Lock()
        self.responses: OrderedDict[str, ResponsesResponse] = OrderedDict()

        self.conversations_lock = asyncio.Lock()
        self.conversations: OrderedDict[str, list[Message]] = OrderedDict()
        self.response_to_conversation: dict[str, str] = {}
        self.conversation_to_response: dict[str, str] = {}

    async def load_response(self, resp_id: str) -> ResponsesResponse:
        responses_debug_log(f"ConversationHistoryStore loading resp: {resp_id}")
        async with self.responses_lock:
            return self.responses.get(resp_id)

    async def store_response(self,
                             resp: ResponsesResponse,
                             resp_msgs: Optional[list[Message]] = [],
                             prev_resp_id: Optional[str] = None) -> None:
        resp_id = resp.id
        responses_debug_log(f"ConversationHistoryStore storing resp: {resp_id}")
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
                conversation_id = self.response_to_conversation[prev_resp_id]
                self.conversations[conversation_id].extend(resp_msgs)
                while len(self.conversations[conversation_id]
                          ) > self.conversation_capacity:
                    self._pop_conversation(resp_id)
            else:
                conversation_id = random_uuid()
                self.conversations[conversation_id] = resp_msgs

            responses_debug_log(
                f" * storing at conversation id: {conversation_id}")

            self.response_to_conversation[resp_id] = conversation_id
            self.conversation_to_response[conversation_id] = resp_id
            self._update_visited_conversation(conversation_id)

    async def store_messages(self, resp_id: str, msgs: list[Message],
                             prev_resp_id: Optional[str]):
        responses_debug_log(f"ConversationHistoryStore storing msg:")
        for msg in msgs:
            responses_debug_log(f" -> {msg.to_json()}")

        async with self.conversations_lock:
            conversation_id: str
            if prev_resp_id is not None:
                conversation_id = self.response_to_conversation[prev_resp_id]
            else:
                conversation_id = random_uuid()

            responses_debug_log(
                f" * storing at conversation: {conversation_id}")
            self.conversations[conversation_id] = msgs
            if len(self.conversations[conversation_id]
                   ) > self.conversation_capacity:
                self._pop_conversation(resp_id)

            self.response_to_conversation[resp_id] = conversation_id
            self.conversation_to_response[conversation_id] = resp_id
            self._update_visited_conversation(conversation_id)

    async def append_messages(self, resp_id: str, msgs: list[Message]):
        responses_debug_log(f"ConversationHistoryStore appending msgs:")
        for msg in msgs:
            responses_debug_log(f" -> {msg.to_json()}")

        async with self.conversations_lock:
            assert resp_id in self.response_to_conversation
            conversation_id = self.response_to_conversation[resp_id]

            responses_debug_log(
                f" * appending at conversation: {conversation_id}")
            self.conversations[conversation_id].extend(msgs)
            if len(self.conversations[conversation_id]
                   ) > self.conversation_capacity:
                self._pop_conversation(resp_id)
            self._update_visited_conversation(conversation_id)

    async def get_conversation_history(self, resp_id: str) -> list[Message]:
        responses_debug_log(f"ConversationHistoryStore getting prev_msgs:")
        responses_debug_log(f" -> prev_resp_id: {resp_id}")
        async with self.conversations_lock:
            if resp_id in self.response_to_conversation:
                conversation_id = self.response_to_conversation[resp_id]
                self._update_visited_conversation(conversation_id)
                return self.conversations.get(conversation_id, [])

            return []

    def _update_visited_conversation(self, conversation_id) -> None:
        if conversation_id not in self.conversations:
            return

        self.conversations.move_to_end(conversation_id)
        if len(self.conversations) > self.max_conversations:
            removed_id, _ = self.conversations.popitem(last=False)
            responses_debug_log(
                f"ConversationHistoryStore Removing conversation {removed_id}")
            removed_resp_id = self.conversation_to_response[removed_id]
            # The responses may have been removed due to response capacity
            if removed_resp_id in self.response_to_conversation:
                self.response_to_conversation.pop(removed_resp_id)
            self.conversation_to_response.pop(removed_id)

    def _pop_conversation(self, resp_id) -> None:
        conversation_id = self.response_to_conversation.get(resp_id, None)
        if conversation_id is None:
            return

        conversation = self.conversations[conversation_id]
        first_conversation_range = []
        for i, msg in enumerate(conversation):
            if msg.author.role == Role.USER:
                first_conversation_range.append(i)
            elif msg.channel == "final":
                first_conversation_range.append(i)
                break
        del conversation[
            first_conversation_range[0]:first_conversation_range[1] + 1]

    def _pop_response(self) -> None:
        responses_debug_log(f"responses type: {type(self.responses)}")
        resp_id, _ = self.responses.popitem(last=False)
        if resp_id in self.response_to_conversation:
            self.response_to_conversation.pop(resp_id)


def get_system_message(
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


def get_developer_message(instructions: Optional[str] = None,
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


def get_user_message(content: str) -> Message:
    return Message.from_role_and_content(Role.USER, content)


def construct_harmony_messages(
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
        sys_msg = get_system_message(reasoning_effort=reasoning_effort, )
        messages.append(sys_msg)
        dev_msg = get_developer_message(request.instructions, request.tools)
        messages.append(dev_msg)
    else:
        messages.extend(prev_msgs)
    # Append the new input.
    # Responses API supports simple text inputs without chat format.
    if isinstance(request.input, str):
        messages.append(get_user_message(request.input))
    else:
        if prev_response is not None:
            prev_outputs = copy(prev_response.output)
        else:
            prev_outputs = []
        for input_msg in request.input:
            msg = parse_response_input(input_msg, prev_outputs)
            if msg is not None:
                messages.append(msg)
            # User passes in a a tool call request and its output. We need
            # to add the tool call request to prev_outputs so that the
            # parse_response_input can find the tool call request when
            # parsing the tool call output.
            if isinstance(input_msg, ResponseFunctionToolCall):
                prev_outputs.append(input_msg)
    return messages


def render_for_completion(messages: list[Message]) -> list[int]:
    conversation = Conversation.from_messages(messages)
    responses_debug_log("Rendering conversation:")
    responses_debug_log(conversation.to_json())
    token_ids = get_encoding().render_conversation_for_completion(
        conversation, Role.ASSISTANT)
    return token_ids


def parse_output_tokens(tokens: list[int]) -> list[Message]:
    return get_encoding().parse_messages_from_completion_tokens(
        tokens, role=Role.ASSISTANT)


def parse_output_message(message: Message) -> list[ResponseOutputItem]:
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
            id=f"ws_{random_uuid()}",
            action=action,
            status="completed",
            type="web_search_call",
        )
        output_items.append(web_search_item)
    elif message.channel == "analysis":
        for content in message.content:
            reasoning_item = ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
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
                random_id = random_uuid()
                response_item = ResponseFunctionToolCall(
                    arguments=content.text,
                    call_id=f"call_{random_id}",
                    type="function_call",
                    name=function_name,
                    id=f"fc_{random_id}",
                )
                output_items.append(response_item)
        elif message.recipient.startswith(
                "python") or message.recipient.startswith("browser"):
            for content in message.content:
                reasoning_item = ResponseReasoningItem(
                    id=f"rs_{random_uuid()}",
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
            id=f"msg_{random_uuid()}",
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


async def request_preprocess(request: ResponsesRequest,
                             prev_response: Optional[ResponsesResponse],
                             harmony_adapter: HarmonyAdapter,
                             conversation_store: ConversationHistoryStore,
                             enable_store=False):
    # TODO: fix default_max_tokens
    sampling_params = request.to_sampling_params(
        default_max_tokens=int(16384),
        default_sampling_params={
            "stop_token_ids": harmony_adapter.get_stop_tokens()
        })

    prev_response_id = request.previous_response_id

    # TODO: better way to enable metrics
    if len(os.getenv("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", "")) > 0:
        sampling_params.return_perf_metrics = True

    prev_msgs = []
    if enable_store:
        prev_msgs = await conversation_store.get_conversation_history(
            prev_response_id)

        responses_debug_log(f"Prev msgs:")
        for msg in prev_msgs:
            responses_debug_log(f" -> {msg.to_json()}")

    messages = construct_harmony_messages(request,
                                          prev_response,
                                          prev_msgs=prev_msgs)

    if enable_store and request.store:
        # Remove reasoning messages to save token usage during multi-turn conversation
        msgs_to_store = [msg for msg in messages if msg.channel != "analysis"]
        await conversation_store.store_messages(request.request_id,
                                                msgs_to_store, prev_response_id)

    input_tokens = render_for_completion(messages)

    responses_debug_log("======= Complete Inputs to model =======")
    responses_debug_log(decode_tokens(input_tokens))
    responses_debug_log("========================================")
    return input_tokens, sampling_params


async def create_response(
    generator,
    request: ResponsesRequest,
    sampling_params,
    model_name: str,
    conversation_store: ConversationHistoryStore,
    generation_result: RequestOutput = None,
    enable_store=False,
    create_time: int = None,
) -> ResponsesResponse:

    final_res: Optional[RequestOutput] = None
    response_creation_time = create_time if create_time is not None else int(
        time.time())
    prev_response_id = request.previous_response_id

    if generation_result is not None:
        final_res = generation_result
    else:
        final_res = await generator

    if final_res is None:
        raise RuntimeError("No output generated or provided")

    responses_debug_log("================================================")
    responses_debug_log("RAW MODEL OUTPUT:")
    responses_debug_log(final_res.outputs)
    responses_debug_log("================================================")

    output_messages = parse_output_tokens(final_res.outputs[0].token_ids)

    responses_debug_log(f"output messages: {len(output_messages)}")
    for msg in output_messages:
        responses_debug_log(f" -> {msg.to_json()}")

    # prepare responses output
    output_content = []
    for msg in output_messages:
        output_content.extend(parse_output_message(msg))

    response = ResponsesResponse.from_request(
        request=request,
        sampling_params=sampling_params,
        model_name=model_name,
        created_time=response_creation_time,
        output=output_content,
        status=finish_reason_mapping(final_res.outputs[0].finish_reason),
    )

    if enable_store and request.store:
        await conversation_store.store_response(resp=response,
                                                resp_msgs=output_messages,
                                                prev_resp_id=prev_response_id)

    responses_debug_log("========== Response ===========")
    responses_debug_log(response)
    responses_debug_log("===============================")
    return response


async def process_streaming_events(
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        generator,
        harmony_adapter: HarmonyAdapter,
        model_name: str,
        conversation_store: ConversationHistoryStore,
        create_time: int = None,
        enable_store=False) -> AsyncGenerator[str, None]:
    sequence_number = 0
    response_creation_time = create_time if create_time is not None else int(
        time.time())
    final_res: Optional[RequestOutput] = None

    def _send_event(event: OpenAIBaseModel):
        nonlocal sequence_number
        # Set sequence_number if the event has this attribute
        if hasattr(event, 'sequence_number'):
            event.sequence_number = sequence_number
        sequence_number += 1
        # Get event type from the event's type field if it exists
        event_type = getattr(event, 'type', 'unknown')
        return (f"event: {event_type}\n"
                f"data: {event.model_dump_json(indent=None)}\n\n")

    current_content_index = 0  # FIXME: this number is never changed
    current_output_index = 0
    current_item_id = ""  # FIXME: this number is never changed
    sent_output_item_added = False

    initial_response = ResponsesResponse.from_request(
        request,
        sampling_params,
        model_name=model_name,
        created_time=response_creation_time,
        output=[],
        status="in_progress",
        usage=None,
    ).model_dump()
    yield _send_event(
        ResponseCreatedEvent(
            type="response.created",
            sequence_number=-1,
            response=initial_response,
        ))
    yield _send_event(
        ResponseInProgressEvent(
            type="response.in_progress",
            sequence_number=-1,
            response=initial_response,
        ))

    tools = [tool.model_dump() for tool in request.tools]
    stream_request_id = f"responses-api-{request.request_id}"
    async for res in generator:
        final_res = res
        output = res.outputs[0]

        messages = harmony_adapter.stateful_stream_harmony_tokens_to_openai_messages(
            stream_request_id, output.token_ids_diff, tools,
            request.tool_choice)
        stream_state = harmony_adapter.get_stream_state(stream_request_id)
        assert stream_state is not None
        parser = stream_state.get_parser()

        if parser.state == StreamState.EXPECT_START:
            current_output_index += 1
            sent_output_item_added = False

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
                        id=current_item_id,
                        summary=[],
                    )
                    yield _send_event(
                        ResponseReasoningTextDoneEvent(
                            type="response.reasoning_text.done",
                            item_id=current_item_id,
                            sequence_number=-1,
                            output_index=current_output_index,
                            content_index=current_content_index,
                            text=previous_item.content[0].text,
                        ))
                    yield _send_event(
                        ResponseOutputItemDoneEvent(
                            type="response.output_item.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=reasoning_item,
                        ))
                elif previous_item.channel == "final":
                    text_content = ResponseOutputText(
                        type="output_text",
                        text=previous_item.content[0].text,
                        annotations=[],
                    )
                    yield _send_event(
                        ResponseTextDoneEvent(
                            type="response.output_text.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            content_index=current_content_index,
                            text=previous_item.content[0].text,
                            logprobs=[],
                            item_id=current_item_id,
                        ))
                    yield _send_event(
                        ResponseContentPartDoneEvent(
                            type="response.content_part.done",
                            sequence_number=-1,
                            item_id=current_item_id,
                            output_index=current_output_index,
                            content_index=current_content_index,
                            part=text_content,
                        ))
                    yield _send_event(
                        ResponseOutputItemDoneEvent(
                            type="response.output_item.done",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=ResponseOutputMessage(
                                id=current_item_id,
                                type="message",
                                role="assistant",
                                content=[text_content],
                                status="completed",
                            ),
                        ))

        if parser.last_content_delta:
            if (parser.current_channel == "final"
                    and parser.current_recipient is None):
                if not sent_output_item_added:
                    sent_output_item_added = True
                    yield _send_event(
                        ResponseOutputItemAddedEvent(
                            type="response.output_item.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=ResponseOutputMessage(
                                id=current_item_id,
                                type="message",
                                role="assistant",
                                content=[],
                                status="in_progress",
                            ),
                        ))
                    yield _send_event(
                        ResponseContentPartAddedEvent(
                            type="response.content_part.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            content_index=current_content_index,
                            part=ResponseOutputText(
                                type="output_text",
                                text="",
                                annotations=[],
                                logprobs=[],
                            ),
                        ))
                yield _send_event(
                    ResponseTextDeltaEvent(
                        type="response.output_text.delta",
                        sequence_number=-1,
                        content_index=current_content_index,
                        output_index=current_output_index,
                        item_id=current_item_id,
                        delta=parser.last_content_delta,
                        # TODO, use logprobs from ctx.last_request_output
                        logprobs=[],
                    ))
            elif (parser.current_channel == "analysis"
                  and parser.current_recipient is None):
                if not sent_output_item_added:
                    sent_output_item_added = True
                    yield _send_event(
                        ResponseOutputItemAddedEvent(
                            type="response.output_item.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item=ResponseReasoningItem(
                                type="reasoning",
                                id=current_item_id,
                                summary=[],
                                status="in_progress",
                            ),
                        ))
                    yield _send_event(
                        ResponseContentPartAddedEvent(
                            type="response.content_part.added",
                            sequence_number=-1,
                            output_index=current_output_index,
                            item_id=current_item_id,
                            content_index=current_content_index,
                            part=ResponseOutputText(
                                type="output_text",
                                text="",
                                annotations=[],
                                logprobs=[],
                            ),
                        ))
                yield _send_event(
                    ResponseReasoningTextDeltaEvent(
                        type="response.reasoning_text.delta",
                        item_id=current_item_id,
                        output_index=current_output_index,
                        content_index=current_content_index,
                        delta=parser.last_content_delta,
                        sequence_number=-1,
                    ))

        # TODO(JunyiXu-nv): support built-in tools(python/browser/code interpreter)

    final_response = await create_response(generator, request, sampling_params,
                                           model_name, conversation_store,
                                           final_res, enable_store,
                                           response_creation_time)

    yield _send_event(
        ResponseCompletedEvent(
            type="response.completed",
            sequence_number=-1,
            response=final_response.model_dump(),
        ))


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
