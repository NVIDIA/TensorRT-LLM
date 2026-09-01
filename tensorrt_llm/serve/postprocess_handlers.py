# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import time
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple, Union

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.responses_utils import ResponsesStreamingProcessor
from tensorrt_llm.serve.responses_utils import \
    create_response_non_store as responses_api_create_response_non_store

from .._utils import nvtx_range_debug
from ..executor import (DetokenizedGenerationResultBase, GenerationResult,
                        GenerationResultBase)
from ..executor.postproc_worker import PostprocArgs
from ..executor.result import Logprob, TokenLogprobs
from ..llmapi import SamplingParams
from ..llmapi.disagg_utils import (get_usage_tokens_from_ctx,
                                   rewrite_usage_info_from_ctx,
                                   rewrite_usage_response_from_ctx)
from ..llmapi.reasoning_parser import (BaseReasoningParser,
                                       ReasoningParserFactory,
                                       ReasoningParserResult)
from ..llmapi.tokenizer import TransformersTokenizer
# yapf: disable
from .chat_utils import make_tool_call_id
from .harmony_adapter import (handle_non_streaming_response,
                              handle_streaming_response)
from .openai_protocol import (ChatCompletionLogProbs,
                              ChatCompletionLogProbsContent,
                              ChatCompletionNamedToolChoiceParam,
                              ChatCompletionRequest, ChatCompletionResponse,
                              ChatCompletionResponseChoice,
                              ChatCompletionResponseStreamChoice,
                              ChatCompletionStreamResponse,
                              ChatCompletionToolsParam, ChatMessage,
                              CompletionLogProbs, CompletionRequest,
                              CompletionResponse, CompletionResponseChoice,
                              CompletionResponseStreamChoice,
                              CompletionStreamResponse, DeltaFunctionCall,
                              DeltaMessage, DeltaToolCall, FunctionCall,
                              PromptTokensDetails, ResponsesRequest,
                              ResponsesResponse, StreamOptions, ToolCall,
                              UsageInfo, to_disaggregated_params)
from .tool_parser.base_tool_parser import BaseToolParser
from .tool_parser.core_types import StreamingParseResult, ToolCallItem
from .tool_parser.tool_parser_factory import ToolParserFactory

# yapf: enable


def _ctx_usage_from_outputs(outputs: List[Any]) -> Optional[UsageInfo]:
    for output in outputs:
        disaggregated_params = getattr(output, "disaggregated_params", None)
        if disaggregated_params is None:
            continue
        ctx_usage = disaggregated_params.ctx_usage
        if ctx_usage is None:
            continue
        if isinstance(ctx_usage, UsageInfo):
            return ctx_usage
        return UsageInfo.model_validate(ctx_usage)
    return None


def _ctx_usage_for_postproc(args: PostprocArgs,
                            outputs: List[Any]) -> Optional[UsageInfo]:
    ctx_usage = args.ctx_usage
    if ctx_usage is not None:
        return ctx_usage
    return _ctx_usage_from_outputs(outputs)


@dataclass(kw_only=True)
class ChatPostprocArgs(PostprocArgs):
    echo: bool = False
    role: str
    model: str
    num_choices: int = 1
    tools: Optional[List[ChatCompletionToolsParam]] = None
    # None means "not specified": only an explicit client "none" (always set
    # by from_request) suppresses parsed tool calls in apply_tool_parser.
    tool_choice: Optional[Union[Literal["none", "auto", "required"],
                                ChatCompletionNamedToolChoiceParam]] = None
    return_logprobs: bool = False
    top_logprobs: bool = False
    stream_options: Optional[StreamOptions] = None
    last_message_content: Optional[str] = None
    reasoning_parser: Optional[str] = None
    tool_parser: Optional[str] = None
    reasoning_parser_dict: dict[int, BaseReasoningParser] = field(
        default_factory=dict)
    tool_parser_dict: dict[int, BaseToolParser] = field(default_factory=dict)
    has_tool_call: dict[int, bool] = field(default_factory=dict)
    tool_call_id_type: str = "random"
    # Per-output flag tracking whether the streaming forced-call path has
    # already emitted the opening delta (with id and function name). The id is
    # generated once on the first non-empty delta; subsequent deltas omit it
    # and only stream argument fragments.
    forced_tool_name_sent: dict[int, bool] = field(default_factory=dict)
    # Streaming forced-call bookkeeping, per output index: the raw text seen so
    # far, how much of it has already been streamed as ``arguments``, and
    # whether the arguments value has finished. Together these let the stream
    # stop at the end of the JSON value instead of forwarding whatever the
    # model generates afterwards.
    forced_tool_args_buffer: dict[int, str] = field(default_factory=dict)
    forced_tool_args_sent_len: dict[int, int] = field(default_factory=dict)
    forced_tool_args_done: dict[int, bool] = field(default_factory=dict)
    chat_template_kwargs: Optional[dict[str, Any]] = None
    ctx_usage: Optional[UsageInfo] = None
    # Cache per-request stream metadata so every chunk reuses the same response
    # id and created timestamp instead of regenerating them for each chunk.
    stream_response_id: Optional[str] = None
    stream_created: Optional[int] = None

    @classmethod
    def from_request(cls, request: ChatCompletionRequest):
        return cls(
            echo=request.echo,
            role="assistant"
            if request.add_generation_prompt else request.messages[-1]["role"],
            model=request.model,
            num_choices=request.n if request.n else 1,
            tools=request.tools,
            tool_choice=request.tool_choice,
            stream_options=request.stream_options,
            return_logprobs=bool(request.logprobs),
            top_logprobs=bool(request.top_logprobs),
            chat_template_kwargs=request.chat_template_kwargs,
            ctx_usage=None if request.disaggregated_params is None else
            request.disaggregated_params.ctx_usage,
        )


def _ensure_stream_metadata(args: Any, rsp: GenerationResultBase,
                            prefix: str) -> Tuple[str, int]:
    if args.stream_response_id is None:
        args.stream_response_id = f"{prefix}-{rsp.id}"
    if args.stream_created is None:
        args.stream_created = int(time.time())
    return args.stream_response_id, args.stream_created


def create_logprobs(token_ids: List[int], tokenizer: TransformersTokenizer,
                    logprobs: List[float] | TokenLogprobs,
                    top_logprobs: bool) -> ChatCompletionLogProbs:
    assert len(token_ids) == len(logprobs), \
            "token_ids and logprobs have different lengths"
    content: List[ChatCompletionLogProbsContent] = []
    for token_id, logprob in zip(token_ids, logprobs):
        logprob: float | dict[int, Logprob]
        token = tokenizer.decode(token_id)
        chat_logprob = ChatCompletionLogProbsContent(
            token=token,
            bytes=list(token.encode("utf-8", errors="replace")),
        )
        if isinstance(logprob, dict):
            if token_id in logprob:
                chat_logprob.logprob = max(logprob[token_id].logprob, -9999.0)
                if top_logprobs:
                    chat_logprob.top_logprobs = [
                        ChatCompletionLogProbsContent(
                            token=(tk := tokenizer.decode(tid)),
                            logprob=max(logprob.logprob, -9999.0),
                            bytes=list(tk.encode("utf-8", errors="replace")))
                        for tid, logprob in logprob.items()
                    ]
        else:
            chat_logprob.logprob = max(logprob, -9999.0)
        content.append(chat_logprob)
    chat_logprobs = ChatCompletionLogProbs(content=content)
    return chat_logprobs


def apply_reasoning_parser(args: ChatPostprocArgs,
                           output_index: int,
                           text: str,
                           streaming: bool,
                           finished: bool = False) -> Tuple[str, str]:
    reasoning_parser = None
    if args.reasoning_parser is not None:
        if output_index not in args.reasoning_parser_dict:
            chat_template_kwargs = getattr(args, "chat_template_kwargs", None)
            args.reasoning_parser_dict[
                output_index] = ReasoningParserFactory.create_reasoning_parser(
                    args.reasoning_parser, chat_template_kwargs)
        reasoning_parser = args.reasoning_parser_dict[output_index]

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


def apply_tool_parser(args: ChatPostprocArgs,
                      output_index: int,
                      text: str,
                      streaming: bool,
                      finished: bool = False) -> Tuple[str, List[ToolCallItem]]:
    tool_parser = None
    tools = args.tools
    if args.tool_parser is not None and tools is not None:
        if output_index not in args.tool_parser_dict:
            args.tool_parser_dict[
                output_index] = ToolParserFactory.create_tool_parser(
                    args.tool_parser)
        tool_parser = args.tool_parser_dict[output_index]

    if tool_parser is not None and tools is not None:
        if not streaming:
            result = tool_parser.detect_and_parse(text, tools)
        else:
            result = tool_parser.parse_streaming_increment(text, tools)
            if finished:
                finish_result = tool_parser.finish(tools)
                result = StreamingParseResult(
                    normal_text=result.normal_text + finish_result.normal_text,
                    calls=result.calls + finish_result.calls)
        if args.tool_choice == "none":
            # tool_choice="none": still run the parser (including the finish
            # flush above) so tool-call markup is stripped from content, but
            # never surface tool calls.
            return result.normal_text, []
        normal_text, calls = result.normal_text, result.calls
        if result.calls:
            args.has_tool_call[output_index] = True
    else:
        normal_text, calls = text, []

    return normal_text, calls


def _forced_tool_choice(
        args: ChatPostprocArgs) -> Optional[ChatCompletionNamedToolChoiceParam]:
    """Return the named tool_choice param if the request forces a tool."""
    if isinstance(args.tool_choice, ChatCompletionNamedToolChoiceParam):
        return args.tool_choice
    return None


def _forced_choice_uses_tool_parser(args: ChatPostprocArgs) -> bool:
    """Whether the configured tool parser extracts forced/named tool calls.

    Parsers whose forced output is grammar-constrained bare JSON keep the
    raw-passthrough behavior; parsers that opt in (see
    ``BaseToolParser.extracts_forced_tool_calls``) still emit their native
    markup on forced calls and need the extraction path.
    """
    if args.tool_parser is None or args.tools is None:
        return False
    parser_cls = ToolParserFactory.parsers.get(args.tool_parser.lower())
    return bool(parser_cls
                and getattr(parser_cls, "extracts_forced_tool_calls", False))


def _forced_call_name(calls: List[ToolCallItem], forced_name: str) -> str:
    """Validate parser-extracted calls against the request's forced tool.

    The name always comes from the request (the caller chose it); the parser
    output is only checked so a disagreeing model is visible in the logs.
    """
    if len(calls) > 1:
        logger.warning(
            f"Forced tool_choice '{forced_name}' produced {len(calls)} tool "
            "calls; keeping the first.")
    parsed_name = calls[0].name
    if parsed_name and parsed_name != forced_name:
        logger.warning(
            f"Forced tool_choice '{forced_name}' but the model emitted a call "
            f"to '{parsed_name}'; using the forced name.")
    return forced_name


_JSON_DECODER = json.JSONDecoder()


def forced_tool_arguments_end(text: str) -> Optional[int]:
    """Return the index just past the first complete JSON value in ``text``.

    The forced-tool-call path prefix-injects ``{"name": X, "arguments":`` into
    the prompt, so generation starts inside a tool-call object. A model that
    keeps going after the arguments closes the *outer* object too and then
    emits the parser's end tag and ordinary assistant text -- all of which
    would otherwise be reported as ``function.arguments``.

    Returns ``None`` while ``text`` is not yet a complete JSON value, so a
    streaming caller can keep buffering.
    """
    stripped = text.lstrip()
    if not stripped:
        return None
    leading_ws = len(text) - len(stripped)
    try:
        _, end = _JSON_DECODER.raw_decode(text, leading_ws)
    except ValueError:
        return None
    return end


@nvtx_range_debug("chat_stream_post_processor")
def chat_stream_post_processor(rsp: GenerationResultBase,
                               args: ChatPostprocArgs) -> List[str]:

    def yield_first_chat(num_tokens: int,
                         idx: int,
                         role: str | None = None,
                         content: str | None = None):
        choice_data = ChatCompletionResponseStreamChoice(index=idx,
                                                         delta=DeltaMessage(
                                                             role=role,
                                                             content=content),
                                                         finish_reason=None)
        chunk = ChatCompletionStreamResponse(choices=[choice_data],
                                             model=args.model,
                                             id=stream_response_id,
                                             created=stream_created)
        if include_continuous_usage:
            chunk.usage = UsageInfo(
                prompt_tokens=num_tokens,
                total_tokens=num_tokens,
                completion_tokens=0,
                prompt_tokens_details=PromptTokensDetails(
                    cached_tokens=rsp.cached_tokens),
            )
            rewrite_usage_info_from_ctx(chunk.usage, ctx_usage)
        data = chunk.model_dump_json(exclude_none=True)
        return data

    res: List[str] = []
    finish_reason_sent = [False] * args.num_choices
    # num_prompt_tokens stays None until a prompt length is recorded, and only
    # the usage branches below consume it, so offset it only once it exists.
    prompt_tokens = args.num_prompt_tokens
    if prompt_tokens is not None:
        prompt_tokens -= args.num_prompt_tokens_offset
    ctx_usage = _ctx_usage_for_postproc(args, rsp.outputs)
    stream_response_id, stream_created = _ensure_stream_metadata(
        args, rsp, "chatcmpl")
    if stream_option := args.stream_options:
        include_usage = stream_option.include_usage
        include_continuous_usage = include_usage and stream_option.continuous_usage_stats
    else:
        include_usage = False
        include_continuous_usage = False
    if args.first_iteration:
        for i in range(args.num_choices):
            res.append(
                f"data: {yield_first_chat(prompt_tokens, i, role=args.role)} \n\n"
            )
            if args.echo and args.last_message_content:
                res.append(
                    f"data: {yield_first_chat(prompt_tokens, i, content=args.last_message_content)} \n\n"
                )
        args.first_iteration = False

    for output in rsp.outputs:
        i = output.index

        if finish_reason_sent[i]:
            continue

        has_token_delta = bool(output.token_ids_diff)
        delta_text = output.text_diff
        delta_text, reasoning_delta_text = apply_reasoning_parser(
            args,
            i,
            delta_text,
            True,
            finished=(output.finish_reason is not None))

        forced_tool = _forced_tool_choice(args)
        if forced_tool and not _forced_choice_uses_tool_parser(args):
            # Forced call constrained by JSON-schema guided decoding: the
            # deltas stream the arguments value. Buffer them so the stream can
            # stop at the end of that value -- generation starts inside the
            # tool call, so a model that overruns goes on to close the
            # enclosing object, emit the parser's end tag and then plain
            # prose, none of which belongs in ``arguments``.
            forced_name = forced_tool.function.name
            delta_arguments = ""
            if delta_text and not args.forced_tool_args_done.get(i, False):
                buffered = args.forced_tool_args_buffer.get(i, "") + delta_text
                args.forced_tool_args_buffer[i] = buffered
                arguments_end = forced_tool_arguments_end(buffered)
                if arguments_end is None:
                    limit = len(buffered)
                else:
                    limit = arguments_end
                    args.forced_tool_args_done[i] = True
                already_sent = args.forced_tool_args_sent_len.get(i, 0)
                delta_arguments = buffered[already_sent:limit]
                args.forced_tool_args_sent_len[i] = max(already_sent, limit)

            tool_calls = []
            if delta_arguments:
                if not args.forced_tool_name_sent.get(i, False):
                    args.forced_tool_name_sent[i] = True
                    tool_calls.append(
                        DeltaToolCall(
                            id=make_tool_call_id(id_type=args.tool_call_id_type,
                                                 func_name=forced_name,
                                                 idx=0),
                            index=0,
                            type="function",
                            function=DeltaFunctionCall(
                                name=forced_name,
                                arguments=delta_arguments,
                            ),
                        ))
                else:
                    tool_calls.append(
                        DeltaToolCall(
                            index=0,
                            function=DeltaFunctionCall(
                                arguments=delta_arguments, ),
                        ))
            # finish_reason may only flip once a call has actually been
            # streamed. A run that ends before any arguments arrived (token
            # budget, abort) would otherwise report "tool_calls" with no call.
            if args.forced_tool_name_sent.get(i, False):
                args.has_tool_call[i] = True
            if not tool_calls and not output.finish_reason and not has_token_delta:
                continue
            delta_message = DeltaMessage(
                tool_calls=tool_calls if tool_calls else None)
        else:
            delta_text, calls = apply_tool_parser(args,
                                                  i,
                                                  delta_text,
                                                  True,
                                                  finished=(output.finish_reason
                                                            is not None))
            if forced_tool and calls:
                forced_name = _forced_call_name(calls,
                                                forced_tool.function.name)
                calls = calls[:1]
                if calls[0].name:
                    calls[0].name = forced_name
            tool_calls = []
            for call_item in calls:
                # Tool call ID should be generated only once per tool call
                if call_item.name:
                    # First chunk: include ID and function name
                    tool_call_id = make_tool_call_id(
                        id_type=args.tool_call_id_type,
                        func_name=call_item.name,
                        idx=call_item.tool_index)
                    function_name = call_item.name
                else:
                    # Subsequent chunks: null ID and name for argument deltas
                    tool_call_id = None
                    function_name = None

                tool_calls.append(
                    DeltaToolCall(
                        id=tool_call_id,
                        index=call_item.tool_index,
                        function=DeltaFunctionCall(
                            name=function_name,
                            arguments=call_item.parameters,
                        ),
                    ))
            # Keep token-bearing chunks visible even when detokenization has
            # no text to flush yet. Only this branch builds ``delta_message``
            # here; the forced branch above builds its own and must not have
            # it rebuilt with ``content``, since a forced call produces no
            # assistant text.
            if (tool_calls or delta_text or reasoning_delta_text
                    or output.finish_reason or has_token_delta):
                delta_message = DeltaMessage(
                    content=delta_text,
                    reasoning_content=reasoning_delta_text,
                    tool_calls=tool_calls if tool_calls else None)
            else:
                continue

        choice = ChatCompletionResponseStreamChoice(
            index=i,
            delta=delta_message,
            avg_decoded_tokens_per_iter=getattr(rsp,
                                                'avg_decoded_tokens_per_iter',
                                                None),
            stop_reason=output.stop_reason,
        )
        if args.return_logprobs:
            logprobs = output.logprobs_diff
            token_ids = output.token_ids_diff
            choice.logprobs = create_logprobs(token_ids, args.tokenizer,
                                              logprobs, args.top_logprobs)
        if output.finish_reason is not None:
            if output.finish_reason == "stop" and args.has_tool_call.get(
                    i, False):
                choice.finish_reason = "tool_calls"
            else:
                choice.finish_reason = output.finish_reason
            choice.stop_reason = output.stop_reason
            finish_reason_sent[i] = True
        chunk = ChatCompletionStreamResponse(choices=[choice],
                                             model=args.model,
                                             id=stream_response_id,
                                             created=stream_created)
        if include_continuous_usage:
            chunk.usage = UsageInfo(prompt_tokens=prompt_tokens,
                                    completion_tokens=output.length,
                                    total_tokens=output.length + prompt_tokens,
                                    prompt_tokens_details=PromptTokensDetails(
                                        cached_tokens=rsp.cached_tokens))
            rewrite_usage_info_from_ctx(chunk.usage, ctx_usage)
        data = chunk.model_dump_json(exclude_none=True)
        res.append(f"data: {data}\n\n")

    if include_usage and rsp._done:
        completion_tokens = sum(output.length for output in rsp.outputs)
        final_usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetails(
                cached_tokens=rsp.cached_tokens),
        )
        rewrite_usage_info_from_ctx(final_usage, ctx_usage)

        final_usage_chunk = ChatCompletionStreamResponse(choices=[],
                                                         model=args.model,
                                                         usage=final_usage,
                                                         id=stream_response_id,
                                                         created=stream_created)
        final_usage_data = final_usage_chunk.model_dump_json()
        res.append(f"data: {final_usage_data}\n\n")
    return res


@nvtx_range_debug("chat_response_post_processor")
def chat_response_post_processor(
        rsp: GenerationResultBase,
        args: ChatPostprocArgs) -> ChatCompletionResponse:
    choices: List[ChatCompletionResponseChoice] = []
    role = args.role
    for output in rsp.outputs:
        text, reasoning_text = apply_reasoning_parser(args, output.index,
                                                      output.text, False)

        forced_tool = _forced_tool_choice(args)
        if forced_tool and not _forced_choice_uses_tool_parser(args):
            # Forced call constrained by JSON-schema guided decoding: the text
            # is the arguments value. Keep only that value -- guided decoding
            # should already stop generation there, but a model that overruns
            # would otherwise have the enclosing brace, the parser's end tag
            # and its trailing prose reported as ``arguments``, which then
            # fails to parse as JSON for the caller.
            if text is None:
                text = ""
            arguments_end = forced_tool_arguments_end(text)
            if arguments_end is None:
                # No complete JSON value: the generation was truncated (token
                # budget, abort) or empty. Reporting the partial text as
                # ``arguments`` would hand the caller something json.loads()
                # rejects, and claiming finish_reason="tool_calls" would assert
                # a call that was never completed. Return the text as content
                # instead, mirroring the no-markup fallback on the extraction
                # path below and the streaming path, which only flips
                # finish_reason once a call has actually been streamed.
                logger.warning(
                    f"Forced tool_choice '{forced_tool.function.name}' but the "
                    "model did not produce a complete JSON arguments value; "
                    "returning the text as content.")
                message = ChatMessage(role=role,
                                      content=text,
                                      reasoning_content=reasoning_text)
            else:
                args.has_tool_call[output.index] = True
                message = ChatMessage(
                    role=role,
                    content="",
                    tool_calls=[
                        ToolCall(id=make_tool_call_id(
                            id_type=args.tool_call_id_type,
                            func_name=forced_tool.function.name,
                            idx=0),
                                 function=FunctionCall(
                                     name=forced_tool.function.name,
                                     arguments=text[:arguments_end]))
                    ])
        elif forced_tool:
            # The parser extracts the forced call from the model's native
            # markup; any free-text preamble becomes content per OpenAI
            # semantics.
            text, calls = apply_tool_parser(args, output.index, text or "",
                                            False)
            if calls:
                forced_name = _forced_call_name(calls,
                                                forced_tool.function.name)
                message = ChatMessage(
                    role=role,
                    content=text,
                    reasoning_content=reasoning_text,
                    tool_calls=[
                        ToolCall(function=FunctionCall(
                            name=forced_name, arguments=calls[0].parameters))
                    ])
            else:
                # No tool markup despite the forced choice (nothing
                # constrains the model for these parsers). Returning the text
                # as arguments would hand the caller garbage JSON, so return
                # it as content and keep finish_reason honest.
                logger.warning(
                    f"Forced tool_choice '{forced_tool.function.name}' but the "
                    "model emitted no tool-call markup; returning the text as "
                    "content.")
                message = ChatMessage(role=role,
                                      content=text,
                                      reasoning_content=reasoning_text)
        else:
            text, calls = apply_tool_parser(args, output.index, text, False)
            tool_calls = [
                ToolCall(function=FunctionCall(name=call.name or "",
                                               arguments=call.parameters))
                for call in calls
            ]
            # Only the non-forced path builds ``message`` here; each forced
            # branch above builds its own. Keeping this outside the ``else``
            # would clobber those and read ``tool_calls`` unbound.
            message = ChatMessage(role=role,
                                  content=text,
                                  reasoning_content=reasoning_text,
                                  tool_calls=tool_calls)
        disaggregated_params = to_disaggregated_params(
            output.disaggregated_params)
        if (disaggregated_params is not None and args.chat_template_kwargs
                and args.reasoning_parser
                and ReasoningParserFactory.resolves_thinking_from_prompt(
                    args.reasoning_parser)):
            # Relay the mode we resolved from the rendered prompt; the
            # generation worker never renders and so cannot resolve it. Gated
            # on the parser opting in, so we never overwrite a mode another
            # parser derives from the caller's own kwargs.
            resolved = args.chat_template_kwargs.get("enable_thinking")
            if resolved is not None:
                disaggregated_params.resolved_thinking = resolved
        choice = ChatCompletionResponseChoice(
            index=output.index,
            message=message,
            stop_reason=output.stop_reason,
            disaggregated_params=disaggregated_params,
            avg_decoded_tokens_per_iter=getattr(rsp,
                                                'avg_decoded_tokens_per_iter',
                                                None),
        )
        if output.finish_reason == "stop" and args.has_tool_call.get(
                output.index, False):
            choice.finish_reason = "tool_calls"
        else:
            choice.finish_reason = output.finish_reason

        if args.return_logprobs:
            choice.logprobs = create_logprobs(output.token_ids, args.tokenizer,
                                              output.logprobs,
                                              args.top_logprobs)
        choices.append(choice)

    if args.echo and args.last_message_content:
        for choice in choices:
            full_message = args.last_message_content + choice.message.content
            choice.message.content = full_message

    num_prompt_tokens = args.num_prompt_tokens - args.num_prompt_tokens_offset
    num_generated_tokens = sum(len(output.token_ids) for output in rsp.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
        prompt_tokens_details=PromptTokensDetails(
            cached_tokens=rsp.cached_tokens),
    )
    ctx_usage = _ctx_usage_for_postproc(args, rsp.outputs)
    response = ChatCompletionResponse(
        model=args.model,
        choices=choices,
        usage=usage,
    )
    rewrite_usage_response_from_ctx(response, ctx_usage)
    return response


@dataclass(kw_only=True)
class CompletionPostprocArgs(PostprocArgs):
    echo: bool = False
    model: str = None
    num_choices: int = 1
    prompt_idx: int = 0
    detokenize: bool = True
    prompt: Optional[str] = None
    return_logprobs: bool = False
    stream_options: Optional[StreamOptions] = None
    ctx_usage: Optional[UsageInfo] = None
    # Cache per-request stream metadata so every chunk reuses the same response
    # id and created timestamp instead of regenerating them for each chunk.
    stream_response_id: Optional[str] = None
    stream_created: Optional[int] = None

    @classmethod
    def from_request(cls, request: CompletionRequest):
        return cls(
            echo=request.echo,
            model=request.model,
            num_choices=request.n if request.n else 1,
            stream_options=request.stream_options,
            detokenize=request.detokenize,
            return_logprobs=bool(request.logprobs),
            ctx_usage=None if request.disaggregated_params is None else
            request.disaggregated_params.ctx_usage,
        )


def create_completion_logprobs(token_ids: List[int],
                               tokenizer: TransformersTokenizer,
                               logprobs: List[float] | TokenLogprobs,
                               initial_offset: int = 0) -> CompletionLogProbs:
    assert len(token_ids) == len(logprobs), \
            "token_ids and logprobs have different lengths"
    text_offset = []
    token_logprobs = []
    top_logprobs_list = []
    tokens = []
    for token_id, logprob in zip(token_ids, logprobs):
        if isinstance(logprob, dict):
            token_logprobs.append(max(logprob[token_id].logprob, -9999.0))
            top_logprobs_list.append({
                tokenizer.decode(tid):
                max(lp.logprob, -9999.0)
                for tid, lp in logprob.items()
            })
        else:
            token_logprobs.append(max(logprob, -9999.0))

        token = tokenizer.decode(token_id)
        if len(text_offset) == 0:
            text_offset.append(initial_offset)
        else:
            text_offset.append(text_offset[-1] + len(token))
        tokens.append(token)
    return CompletionLogProbs(text_offset=text_offset,
                              token_logprobs=token_logprobs,
                              tokens=tokens,
                              top_logprobs=top_logprobs_list)


@nvtx_range_debug("completion_stream_post_processor")
def completion_stream_post_processor(rsp: DetokenizedGenerationResultBase,
                                     args: CompletionPostprocArgs) -> List[str]:
    res: List[str] = []
    prompt_tokens = args.num_prompt_tokens
    ctx_usage = _ctx_usage_for_postproc(args, rsp.outputs)
    stream_response_id, stream_created = _ensure_stream_metadata(
        args, rsp, "cmpl")
    if stream_option := args.stream_options:
        include_usage = stream_option.include_usage
        include_continuous_usage = include_usage and stream_option.continuous_usage_stats
    else:
        include_usage = False
        include_continuous_usage = False

    for output in rsp.outputs:
        delta_text = output.text_diff
        if args.echo and args.first_iteration:
            delta_text = args.prompt + delta_text
        choice = CompletionResponseStreamChoice(
            index=args.prompt_idx * args.num_choices + output.index,
            text=delta_text if args.detokenize else "",
            token_ids=None if args.detokenize else output.token_ids_diff,
            finish_reason=output.finish_reason,
            stop_reason=output.stop_reason,
            avg_decoded_tokens_per_iter=getattr(rsp,
                                                'avg_decoded_tokens_per_iter',
                                                None),
        )
        if args.return_logprobs:
            logprobs = output.logprobs_diff
            token_ids = output.token_ids_diff
            choice.logprobs = create_completion_logprobs(
                token_ids, args.tokenizer, logprobs, output._last_text_len)

        chunk = CompletionStreamResponse(model=args.model,
                                         choices=[choice],
                                         id=stream_response_id,
                                         created=stream_created)
        if include_continuous_usage:
            chunk.usage = UsageInfo(prompt_tokens=prompt_tokens,
                                    completion_tokens=output.length,
                                    total_tokens=output.length + prompt_tokens,
                                    prompt_tokens_details=PromptTokensDetails(
                                        cached_tokens=rsp.cached_tokens))
            rewrite_usage_info_from_ctx(chunk.usage, ctx_usage)
        data = chunk.model_dump_json(exclude_unset=False)
        res.append(f"data: {data}\n\n")

    if include_usage and rsp._done:
        completion_tokens = sum(output.length for output in rsp.outputs)
        final_usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetails(
                cached_tokens=rsp.cached_tokens),
        )
        rewrite_usage_info_from_ctx(final_usage, ctx_usage)

        final_usage_chunk = CompletionStreamResponse(choices=[],
                                                     model=args.model,
                                                     usage=final_usage,
                                                     id=stream_response_id,
                                                     created=stream_created)
        final_usage_data = final_usage_chunk.model_dump_json()
        res.append(f"data: {final_usage_data}\n\n")
    args.first_iteration = False
    return res


@nvtx_range_debug("completion_response_post_processor")
def completion_response_post_processor(
        rsp: GenerationResult,
        args: CompletionPostprocArgs) -> CompletionResponse:
    prompt_tokens = args.num_prompt_tokens
    completion_tokens = 0
    choices = []
    for output in rsp.outputs:
        text = output.text
        if args.echo:
            text = args.prompt + text
        disaggregated_params = to_disaggregated_params(
            output.disaggregated_params)
        choice = CompletionResponseChoice(
            text=text if args.detokenize else "",
            token_ids=None if args.detokenize else output.token_ids,
            index=args.prompt_idx * args.num_choices + output.index,
            disaggregated_params=disaggregated_params,
            context_logits=None
            if rsp.context_logits is None else rsp.context_logits.tolist(),
            stop_reason=output.stop_reason,
            finish_reason=output.finish_reason,
            avg_decoded_tokens_per_iter=getattr(rsp,
                                                'avg_decoded_tokens_per_iter',
                                                None),
        )
        if args.return_logprobs:
            logprobs = output.logprobs
            token_ids = output.token_ids
            choice.logprobs = create_completion_logprobs(
                token_ids, args.tokenizer, logprobs)

        completion_tokens += output.length
        choices.append(choice)

    usage = UsageInfo(prompt_tokens=prompt_tokens,
                      completion_tokens=completion_tokens,
                      total_tokens=completion_tokens + prompt_tokens,
                      prompt_tokens_details=PromptTokensDetails(
                          cached_tokens=rsp.cached_tokens))
    response = CompletionResponse(choices=choices,
                                  model=args.model,
                                  usage=usage)
    ctx_usage = _ctx_usage_for_postproc(args, rsp.outputs)
    rewrite_usage_response_from_ctx(response, ctx_usage)
    return response


@dataclass(kw_only=True)
class ChatCompletionPostprocArgs(PostprocArgs):
    model: str
    tools: Optional[List[ChatCompletionToolsParam]]
    tool_choice: Optional[Union[Literal["none", "auto", "required"],
                                ChatCompletionNamedToolChoiceParam]]
    request_id: Optional[int] = None
    stream_options: Optional[StreamOptions] = None
    chat_template_kwargs: Optional[dict[str, Any]] = None
    ctx_usage: Optional[UsageInfo] = None
    stream_response_id: Optional[str] = None
    stream_created: Optional[int] = None

    @classmethod
    def from_request(cls, request: ChatCompletionRequest):
        return cls(
            model=request.model,
            tools=request.tools,
            tool_choice=request.tool_choice,
            stream_options=request.stream_options if request.stream else None,
            chat_template_kwargs=request.chat_template_kwargs,
            ctx_usage=None if request.disaggregated_params is None else
            request.disaggregated_params.ctx_usage,
        )


@nvtx_range_debug("chat_harmony_post_processor")
def chat_harmony_post_processor(
        rsp: GenerationResult,
        args: ChatCompletionPostprocArgs) -> ChatCompletionResponse:
    response = handle_non_streaming_response(
        tools=args.tools,
        tool_choice=args.tool_choice,
        outputs=rsp.outputs,
        model=args.model,
        num_prompt_tokens=args.num_prompt_tokens,
        cached_tokens=rsp.cached_tokens,
    )
    ctx_usage = _ctx_usage_for_postproc(args, rsp.outputs)
    rewrite_usage_response_from_ctx(response, ctx_usage)
    return response


@nvtx_range_debug("chat_harmony_streaming_post_processor")
def chat_harmony_streaming_post_processor(
        rsp: GenerationResult, args: ChatCompletionPostprocArgs) -> List[str]:
    # Read the request ID directly from rsp.id instead of args.request_id.
    # Both are the same executor-assigned ID, but args.request_id is set too
    # late (after generate_async returns) for the postprocess worker path:
    # the worker receives a copy of args before the ID is assigned, so
    # args.request_id is always None with num_postprocess_workers > 0.
    prompt_tokens = args.num_prompt_tokens
    cached_tokens = rsp.cached_tokens
    ctx_usage = _ctx_usage_for_postproc(args, rsp.outputs)
    ctx_prompt_tokens, ctx_cached_tokens = get_usage_tokens_from_ctx(ctx_usage)
    if ctx_prompt_tokens is not None:
        prompt_tokens = ctx_prompt_tokens
        cached_tokens = ctx_cached_tokens
    stream_response_id, stream_created = _ensure_stream_metadata(
        args, rsp, "chatcmpl")
    response = handle_streaming_response(
        tools=args.tools,
        tool_choice=args.tool_choice,
        result=rsp,
        model=args.model,
        request_id=str(rsp.id),
        done=rsp._done,
        num_prompt_tokens=prompt_tokens,
        first_iteration=args.first_iteration,
        stream_options=args.stream_options,
        cached_tokens=cached_tokens,
        stream_response_id=stream_response_id,
        stream_created=stream_created,
    )
    args.first_iteration = False
    return response


@dataclass(kw_only=True)
class ResponsesAPIPostprocArgs(PostprocArgs):
    model: str
    request: ResponsesRequest
    sampling_params: SamplingParams
    use_harmony: bool
    reasoning_parser: Optional[str] = None
    tool_parser: Optional[str] = None
    streaming_processor: Optional[ResponsesStreamingProcessor] = None


@nvtx_range_debug("responses_api_post_processor")
def responses_api_post_processor(
        rsp: GenerationResult,
        args: ResponsesAPIPostprocArgs) -> ResponsesResponse:
    return responses_api_create_response_non_store(
        generation_result=rsp,
        request=args.request,
        sampling_params=args.sampling_params,
        model_name=args.model,
        use_harmony=args.use_harmony,
        reasoning_parser=args.reasoning_parser,
        tool_parser=args.tool_parser,
        num_prompt_tokens=args.num_prompt_tokens,
    )


@nvtx_range_debug("responses_api_streaming_post_processor")
def responses_api_streaming_post_processor(
        rsp: GenerationResult, args: ResponsesAPIPostprocArgs) -> List[str]:
    if args.streaming_processor is None:
        raise ValueError(
            "streaming_processor is required for streaming post-processing")
    outputs = args.streaming_processor.process_single_output(rsp)
    if rsp._done:
        outputs.append(
            args.streaming_processor.get_final_response_non_store(
                rsp, args.num_prompt_tokens))
    return outputs
