from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union

from openai_harmony import (HarmonyEncodingName, Role, StreamableParser,
                            load_harmony_encoding)

from .._utils import nvtx_range_debug
from ..executor import (DetokenizedGenerationResultBase, GenerationResult,
                        GenerationResultBase)
from ..executor.postproc_worker import PostprocArgs
from ..llmapi.reasoning_parser import (BaseReasoningParser,
                                       ReasoningParserFactory)
from ..llmapi.tokenizer import TransformersTokenizer
# yapf: disable
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
                              FunctionCall, StreamOptions, ToolCall, UsageInfo,
                              to_disaggregated_params)

# yapf: enale

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
    stream_options: Optional[StreamOptions] = None
    last_message_content: Optional[str] = None
    reasoning_parser: Optional[str] = None
    reasoning_parser_dict: dict[int, BaseReasoningParser] = field(
        default_factory=dict)
    use_harmony: bool = False
    harmony_parsers: list[StreamableParser] = field(default_factory=list)

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
            return_logprobs=request.logprobs,
            use_harmony=request.use_harmony,
        )


def create_logprobs(token_ids: List[int],
                    tokenizer: TransformersTokenizer,
                    logprobs: List[float]) -> ChatCompletionLogProbs:
    assert len(token_ids) == len(logprobs), \
            "token_ids and logprobs have different lengths"
    content: List[ChatCompletionLogProbsContent] = []
    for token_id, logprob in zip(token_ids, logprobs):
        token = tokenizer.decode(token_id)
        # returning multiple logprobs is not supported
        first_logprob = ChatCompletionLogProbsContent(
            token=token,
            logprob=max(logprob, -9999.0),
            bytes=list(token.encode("utf-8", errors="replace")))
        content.append(first_logprob)
    chat_logprobs = ChatCompletionLogProbs(content=content)
    return chat_logprobs


def apply_reasoning_parser(args: ChatPostprocArgs, output_index: int, text: str, streaming: bool) -> Tuple[bool, str, str]:
    reasoning_parser = None
    if args.reasoning_parser is not None:
        if output_index not in args.reasoning_parser_dict:
            args.reasoning_parser_dict[output_index] = ReasoningParserFactory.create_reasoning_parser(
                args.reasoning_parser)
        reasoning_parser = args.reasoning_parser_dict[output_index]

    in_reasoning = False
    if reasoning_parser is not None:
        if not streaming:
            result = reasoning_parser.parse(text)
        else:
            result = reasoning_parser.parse_delta(text)
        in_reasoning, content, reasoning_text = result.in_reasoning, result.content, result.reasoning_text
    else:
        in_reasoning, content, reasoning_text = False, text, None

    return in_reasoning, content, reasoning_text


def init_harmony_parsers(args: ChatPostprocArgs):
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    args.harmony_parsers = [
        StreamableParser(encoding, Role.ASSISTANT)
        for _ in range(args.num_choices)
    ]


@nvtx_range_debug("chat_stream_post_processor")
def chat_stream_post_processor(rsp: GenerationResultBase, args: ChatPostprocArgs) -> List[str]:

    def yield_first_chat(num_tokens: int,
                         idx: int,
                         role: str = None,
                         content: str = None):
        choice_data = ChatCompletionResponseStreamChoice(index=idx,
                                                         delta=DeltaMessage(
                                                             role=role,
                                                             content=content),
                                                         finish_reason=None)
        chunk = ChatCompletionStreamResponse(choices=[choice_data],
                                             model=args.model)
        if include_continuous_usage:
            chunk.usage = UsageInfo(prompt_tokens=num_tokens,
                                    total_tokens=num_tokens,
                                    completion_tokens=0)
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
            res.append(f"data: {yield_first_chat(prompt_tokens, i, role=args.role)} \n\n")
            if args.echo and args.last_message_content:
                res.append(f"data: {yield_first_chat(prompt_tokens, i, content=args.last_message_content)} \n\n")
        args.first_iteration = False

    if args.use_harmony and not args.harmony_parsers:
        init_harmony_parsers(args)

    for output in rsp.outputs:
        i = output.index

        if finish_reason_sent[i]:
            continue

        delta_text = output.text_diff

        in_reasoning, delta_text, reasoning_delta_text = apply_reasoning_parser(
            args, i, delta_text, True)

        if args.tool_choice and type(
                args.tool_choice) is ChatCompletionNamedToolChoiceParam:
            delta_message = DeltaMessage(tool_calls=[
                ToolCall(function=FunctionCall(
                    name=args.tool_choice.function.name, arguments=delta_text))
            ])
        else:
            if in_reasoning:
                delta_message = DeltaMessage(
                    reasoning_text=reasoning_delta_text)
            else:
                delta_message = DeltaMessage(
                    content=delta_text, reasoning_text=reasoning_delta_text)

        choice = ChatCompletionResponseStreamChoice(index=i,
                                                    delta=delta_message,
                                                    finish_reason=None)
        if args.return_logprobs:
            logprobs = output.logprobs_diff
            token_ids = output.token_ids_diff
            choice.logprobs = create_logprobs(token_ids, args.tokenizer, logprobs)
        if output.finish_reason is not None:
            choice.finish_reason = output.finish_reason
            choice.stop_reason = output.stop_reason
            finish_reason_sent[i] = True
        chunk = ChatCompletionStreamResponse(choices=[choice], model=args.model)
        if include_continuous_usage:
            chunk.usage = UsageInfo(prompt_tokens=prompt_tokens,
                                    completion_tokens=output.length,
                                    total_tokens=output.length + prompt_tokens)
        data = chunk.model_dump_json(exclude_none=True)
        res.append(f"data: {data}\n\n")

    if include_usage and rsp._done:
        completion_tokens = sum(output.length
                                for output in rsp.outputs)
        final_usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        final_usage_chunk = ChatCompletionStreamResponse(
            choices=[], model=args.model, usage=final_usage)
        final_usage_data = final_usage_chunk.model_dump_json()
        res.append(f"data: {final_usage_data}\n\n")
    return res


@nvtx_range_debug("chat_response_post_processor")
def chat_response_post_processor(rsp: GenerationResultBase, args: ChatPostprocArgs) -> ChatCompletionResponse:
    choices: List[ChatCompletionResponseChoice] = []
    role = args.role
    if args.use_harmony and not args.harmony_parsers:
        init_harmony_parsers(args)

    for output in rsp.outputs:
        if args.use_harmony:
            parser = args.harmony_parsers[output.index]
            for token in output.token_ids:
                parser.process(token)
            msg = parser.messages
            if len(msg) == 0:
                # The generation has stopped during reasoning.
                reasoning_text = parser.current_content
                text = None
            elif len(msg) == 1:
                # The generation has stopped during final message.
                reasoning_text = msg[0].content[0].text
                text = parser.current_content
            else:
                if len(msg) != 2:
                    raise ValueError(
                        "Expected 2 output messages (reasoning and final), "
                        f"but got {len(msg)}.")
                reasoning_msg, final_msg = msg
                reasoning_text = reasoning_msg.content[0].text
                text = final_msg.content[0].text
        else:
            _, text, reasoning_text = apply_reasoning_parser(
                args, output.index, output.text, False)

        if args.tool_choice and isinstance(
                args.tool_choice,
                ChatCompletionNamedToolChoiceParam):
            message = ChatMessage(
                role=role,
                content="",
                tool_calls=[
                    ToolCall(function=FunctionCall(
                        name=args.tool_choice.function.name,
                        arguments=text))
                ])
        else:
            if text is None:
                text = ""
            message = ChatMessage(
                role=role, content=text, reasoning_content=reasoning_text)
        disaggregated_params = to_disaggregated_params(output.disaggregated_params)
        choice = ChatCompletionResponseChoice(
            index=output.index,
            message=message,
            finish_reason=output.finish_reason,
            stop_reason=output.stop_reason,
            disaggregated_params=disaggregated_params,
        )

        if args.return_logprobs:
            choice.logprobs = create_logprobs(output.token_ids, args.tokenizer, output.logprobs)
        choices.append(choice)

    if args.echo and args.last_message_content:
        for choice in choices:
            full_message = args.last_message_content + choice.message.content
            choice.message.content = full_message

    num_prompt_tokens = args.num_prompt_tokens
    num_generated_tokens = sum(
        len(output.token_ids) for output in rsp.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
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
def completion_stream_post_processor(rsp: DetokenizedGenerationResultBase, args: CompletionPostprocArgs) -> List[str]:
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
            finish_reason = output.finish_reason,
            stop_reason = output.stop_reason,
        )
        chunk = CompletionStreamResponse(model=args.model, choices=[choice])
        if include_continuous_usage:
            chunk.usage = UsageInfo(prompt_tokens=prompt_tokens,
                                    completion_tokens=output.length,
                                    total_tokens=output.length + prompt_tokens)
        data = chunk.model_dump_json(exclude_unset=False)
        res.append(f"data: {data}\n\n")

    if include_usage and rsp._done:
        completion_tokens = sum(output.length
                                for output in rsp.outputs)
        final_usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        final_usage_chunk = ChatCompletionStreamResponse(
            choices=[], model=args.model, usage=final_usage)
        final_usage_data = final_usage_chunk.model_dump_json()
        res.append(f"data: {final_usage_data}\n\n")
    args.first_iteration = False
    return res


@nvtx_range_debug("completion_response_post_processor")
def completion_response_post_processor(rsp: GenerationResult, args: CompletionPostprocArgs) -> CompletionResponse:
    prompt_tokens = args.num_prompt_tokens
    completion_tokens = 0
    choices = []
    for output in rsp.outputs:
        text = output.text
        if args.echo:
            text = args.prompt + text
        disaggregated_params = to_disaggregated_params(output.disaggregated_params)
        choice = CompletionResponseChoice(
            text=text if args.detokenize else "",
            token_ids=None if args.detokenize else output.token_ids,
            index=args.prompt_idx * args.num_choices + output.index,
            disaggregated_params=disaggregated_params,
            context_logits=None if rsp.context_logits is None else rsp.context_logits.tolist(),
            stop_reason=output.stop_reason,
            finish_reason=output.finish_reason,
        )

        completion_tokens += output.length
        choices.append(choice)

    usage = UsageInfo(prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=completion_tokens + prompt_tokens)
    response = CompletionResponse(choices=choices, model=args.model, usage=usage)
    return response
