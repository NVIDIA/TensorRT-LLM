from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union

from .._utils import nvtx_range_debug
from ..executor import (DetokenizedGenerationResultBase, GenerationResult,
                        GenerationResultBase)
from ..executor.postproc_worker import PostprocArgs
from ..executor.result import Logprob, TokenLogprobs
from ..llmapi.reasoning_parser import (BaseReasoningParser,
                                       ReasoningParserFactory)
from ..llmapi.tokenizer import TransformersTokenizer
# yapf: disable
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
                              CompletionRequest, CompletionResponse,
                              CompletionResponseChoice,
                              CompletionResponseStreamChoice,
                              CompletionStreamResponse, DeltaMessage,
                              FunctionCall, PromptTokensDetails, StreamOptions,
                              ToolCall, UsageInfo, to_disaggregated_params)

# yapf: enable


@dataclass(kw_only=True)
class ChatPostprocArgs(PostprocArgs):
    echo: bool = False
    role: str = None
    model: str = None
    num_choices: int = 1
    tools: Optional[List[ChatCompletionToolsParam]] = None
    tool_choice: Optional[Union[Literal["none"],
                                ChatCompletionNamedToolChoiceParam]] = "none"
    return_logprobs: bool = False
    top_logprobs: bool = False
    stream_options: Optional[StreamOptions] = None
    last_message_content: Optional[str] = None
    reasoning_parser: Optional[str] = None
    reasoning_parser_dict: dict[int, BaseReasoningParser] = field(
        default_factory=dict)

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
        )


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


def apply_reasoning_parser(args: ChatPostprocArgs, output_index: int, text: str,
                           streaming: bool) -> Tuple[str, str]:
    reasoning_parser = None
    if args.reasoning_parser is not None:
        if output_index not in args.reasoning_parser_dict:
            args.reasoning_parser_dict[
                output_index] = ReasoningParserFactory.create_reasoning_parser(
                    args.reasoning_parser)
        reasoning_parser = args.reasoning_parser_dict[output_index]

    if reasoning_parser is not None:
        if not streaming:
            result = reasoning_parser.parse(text)
        else:
            result = reasoning_parser.parse_delta(text)
        content, reasoning_content = result.content, result.reasoning_content
    else:
        content, reasoning_content = text, ""

    return content, reasoning_content


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
                                             model=args.model)
        if include_continuous_usage:
            chunk.usage = UsageInfo(
                prompt_tokens=num_tokens,
                total_tokens=num_tokens,
                completion_tokens=0,
                prompt_tokens_details=PromptTokensDetails(
                    cached_tokens=rsp.cached_tokens),
            )
        data = chunk.model_dump_json(exclude_none=True)
        return data

    res: List[str] = []
    finish_reason_sent = [False] * args.num_choices
    prompt_tokens = args.num_prompt_tokens
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

        delta_text = output.text_diff

        delta_text, reasoning_delta_text = apply_reasoning_parser(
            args, i, delta_text, True)

        if args.tool_choice and type(
                args.tool_choice) is ChatCompletionNamedToolChoiceParam:
            delta_message = DeltaMessage(tool_calls=[
                ToolCall(function=FunctionCall(
                    name=args.tool_choice.function.name, arguments=delta_text))
            ])
        else:
            delta_message = DeltaMessage(content=delta_text,
                                         reasoning_content=reasoning_delta_text)

        choice = ChatCompletionResponseStreamChoice(
            index=i,
            delta=delta_message,
            finish_reason=None,
            avg_decoded_tokens_per_iter=getattr(rsp,
                                                'avg_decoded_tokens_per_iter',
                                                None))
        if args.return_logprobs:
            logprobs = output.logprobs_diff
            token_ids = output.token_ids_diff
            choice.logprobs = create_logprobs(token_ids, args.tokenizer,
                                              logprobs, args.top_logprobs)
        if output.finish_reason is not None:
            choice.finish_reason = output.finish_reason
            choice.stop_reason = output.stop_reason
            finish_reason_sent[i] = True
        chunk = ChatCompletionStreamResponse(choices=[choice], model=args.model)
        if include_continuous_usage:
            chunk.usage = UsageInfo(prompt_tokens=prompt_tokens,
                                    completion_tokens=output.length,
                                    total_tokens=output.length + prompt_tokens,
                                    prompt_tokens_details=PromptTokensDetails(
                                        cached_tokens=rsp.cached_tokens))
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

        final_usage_chunk = ChatCompletionStreamResponse(choices=[],
                                                         model=args.model,
                                                         usage=final_usage)
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

        if args.tool_choice and isinstance(args.tool_choice,
                                           ChatCompletionNamedToolChoiceParam):
            message = ChatMessage(
                role=role,
                content="",
                tool_calls=[
                    ToolCall(function=FunctionCall(
                        name=args.tool_choice.function.name, arguments=text))
                ])
        else:
            message = ChatMessage(role=role,
                                  content=text,
                                  reasoning_content=reasoning_text)
        disaggregated_params = to_disaggregated_params(
            output.disaggregated_params)
        choice = ChatCompletionResponseChoice(
            index=output.index,
            message=message,
            finish_reason=output.finish_reason,
            stop_reason=output.stop_reason,
            disaggregated_params=disaggregated_params,
            avg_decoded_tokens_per_iter=getattr(rsp,
                                                'avg_decoded_tokens_per_iter',
                                                None),
        )

        if args.return_logprobs:
            choice.logprobs = create_logprobs(output.token_ids, args.tokenizer,
                                              output.logprobs,
                                              args.top_logprobs)
        choices.append(choice)

    if args.echo and args.last_message_content:
        for choice in choices:
            full_message = args.last_message_content + choice.message.content
            choice.message.content = full_message

    num_prompt_tokens = args.num_prompt_tokens
    num_generated_tokens = sum(len(output.token_ids) for output in rsp.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
        prompt_tokens_details=PromptTokensDetails(
            cached_tokens=rsp.cached_tokens),
    )
    response = ChatCompletionResponse(
        model=args.model,
        choices=choices,
        usage=usage,
    )
    return response


@dataclass(kw_only=True)
class CompletionPostprocArgs(PostprocArgs):
    echo: bool = False
    model: str = None
    num_choices: int = 1
    prompt_idx: int = 0
    detokenize: bool = True
    prompt: Optional[str] = None
    stream_options: Optional[StreamOptions] = None

    @classmethod
    def from_request(cls, request: CompletionRequest):
        return cls(
            echo=request.echo,
            model=request.model,
            num_choices=request.n if request.n else 1,
            stream_options=request.stream_options,
            detokenize=request.detokenize,
        )


@nvtx_range_debug("completion_stream_post_processor")
def completion_stream_post_processor(rsp: DetokenizedGenerationResultBase,
                                     args: CompletionPostprocArgs) -> List[str]:
    res: List[str] = []
    prompt_tokens = args.num_prompt_tokens
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
        chunk = CompletionStreamResponse(model=args.model, choices=[choice])
        if include_continuous_usage:
            chunk.usage = UsageInfo(prompt_tokens=prompt_tokens,
                                    completion_tokens=output.length,
                                    total_tokens=output.length + prompt_tokens,
                                    prompt_tokens_details=PromptTokensDetails(
                                        cached_tokens=rsp.cached_tokens))
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

        final_usage_chunk = ChatCompletionStreamResponse(choices=[],
                                                         model=args.model,
                                                         usage=final_usage)
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
    return response


@dataclass(kw_only=True)
class ChatCompletionPostprocArgs(PostprocArgs):
    model: str
    tools: Optional[List[ChatCompletionToolsParam]]
    tool_choice: Optional[Union[Literal["none", "auto"],
                                ChatCompletionNamedToolChoiceParam]]
    request_id: Optional[int] = None

    @classmethod
    def from_request(cls, request: ChatCompletionRequest):
        return cls(
            model=request.model,
            tools=request.tools,
            tool_choice=request.tool_choice,
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
    )
    return response


@nvtx_range_debug("chat_harmony_streaming_post_processor")
def chat_harmony_streaming_post_processor(
        rsp: GenerationResult, args: ChatCompletionPostprocArgs) -> List[str]:
    response = handle_streaming_response(
        tools=args.tools,
        tool_choice=args.tool_choice,
        result=rsp,
        model=args.model,
        request_id=args.request_id,
        done=rsp._done,
        num_prompt_tokens=args.num_prompt_tokens,
    )
    return response
