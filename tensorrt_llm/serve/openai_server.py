#!/usr/bin/env python
import asyncio
import signal
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import AsyncGenerator, AsyncIterator, List, Tuple, TypedDict

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openai.types.chat import ChatCompletionMessageParam
from transformers import PreTrainedTokenizer

# yapf: disable
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.llmapi import LLM
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionLogProbs, ChatCompletionLogProbsContent,
    ChatCompletionNamedToolChoiceParam, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse,
    ChatMessage, CompletionRequest, CompletionResponse,
    CompletionResponseChoice, CompletionResponseStreamChoice,
    CompletionStreamResponse, DeltaMessage, ErrorResponse, FunctionCall,
    ModelCard, ModelList, ToolCall, UsageInfo)
from tensorrt_llm.version import __version__ as VERSION

# yapf: enale
TIMEOUT_KEEP_ALIVE = 5  # seconds.


class ConversationMessage(TypedDict):
    role: str
    content: str


def parse_chat_message_content(
    message: ChatCompletionMessageParam, ) -> ConversationMessage:
    role = message["role"]
    content = message.get("content")

    if content is None:
        return []
    if isinstance(content, str):
        return [ConversationMessage(role=role, content=content)]

    # for Iterable[ChatCompletionContentPartTextParam]
    texts: List[str] = []
    for part in content:
        part_type = part["type"]
        if part_type == "text":
            text = part["text"]
            texts.append(text)
        else:
            raise NotImplementedError(f"{part_type} is not supported")

    text_prompt = "\n".join(texts)
    return [ConversationMessage(role=role, content=text_prompt)]


class OpenAIServer:

    def __init__(self,
                 llm: LLM,
                 model: str,
                 hf_tokenizer: PreTrainedTokenizer = None):
        self.llm = llm
        self.tokenizer = hf_tokenizer

        model_dir = Path(model)
        if model_dir.exists() and model_dir.is_dir():
            self.model = model_dir.name
        else:
            self.model = model

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # terminate rank0 worker
            yield
            self.llm.shutdown()

        self.app = FastAPI(lifespan=lifespan)

        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(_, exc):
            return self.create_error_response(message=str(exc))

        self.register_routes()

    @staticmethod
    def create_error_response(
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
        error_response = ErrorResponse(message=message,
                                       type=err_type,
                                       code=status_code.value)
        return JSONResponse(content=error_response.model_dump(),
                            status_code=error_response.code)

    def register_routes(self):
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/version", self.version, methods=["GET"])
        self.app.add_api_route("/v1/models", self.get_model, methods=["GET"])
        self.app.add_api_route("/v1/completions",
                               self.openai_completion,
                               methods=["POST"])
        self.app.add_api_route("/v1/chat/completions",
                               self.openai_chat,
                               methods=["POST"])

    async def health(self) -> Response:
        return Response(status_code=200)

    async def version(self) -> JSONResponse:
        ver = {"version": VERSION}
        return JSONResponse(content=ver)

    async def get_model(self) -> JSONResponse:
        model_list = ModelList(data=[ModelCard(id=self.model)])
        return JSONResponse(content=model_list.model_dump())

    async def openai_chat(self, request: ChatCompletionRequest) -> Response:

        def get_role() -> str:
            if request.add_generation_prompt:
                role = "assistant"
            else:
                role = request.messages[-1]["role"]
            return role

        def stream_usage_info(prompt_tokens: int, completion_tokens: int):
            if request.stream_options and request.stream_options.include_usage and \
                request.stream_options.continuous_usage_stats:
                usage = UsageInfo(prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens,
                                    total_tokens=prompt_tokens +
                                    completion_tokens)
            else:
                usage = None
            return usage

        def create_logprobs(token_ids: List[int],
                            logprobs: List[float]) -> ChatCompletionLogProbs:
            assert len(token_ids) == len(logprobs), \
                   "token_ids and logprobs have different lengths"
            content: List[ChatCompletionLogProbsContent] = []
            for token_id, logprob in zip(token_ids, logprobs):
                token = self.tokenizer.decode(token_id)
                # returning multiple logprobs is not supported
                first_logprob = ChatCompletionLogProbsContent(
                    token=token, logprob=max(logprob, -9999.0),
                    bytes=list(token.encode("utf-8", errors="replace"))
                )
                content.append(first_logprob)
            chat_logprobs = ChatCompletionLogProbs(content=content)
            return chat_logprobs

        async def chat_stream_generator(promise: RequestOutput) -> AsyncGenerator[str, None]:
            first_iteration = True
            num_choices = 1 if request.n is None else request.n
            finish_reason_sent = [False] * num_choices
            role = get_role()

            def yield_first_chat(num_tokens: int, role: str = None, content: str = None):
                for i in range(num_choices):
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=DeltaMessage(
                            role=role, content=content),
                        finish_reason=None)
                    chunk = ChatCompletionStreamResponse(
                        choices=[choice_data], model=self.model)
                    chunk.usage = stream_usage_info(num_tokens, 0)

                    data = chunk.model_dump_json(exclude_unset=True)
                    return data

            async for res in promise:
                prompt_tokens = len(res.prompt_token_ids)
                if first_iteration:
                    yield f"data: {yield_first_chat(prompt_tokens, role=role)} \n\n"

                    if request.echo:
                        last_msg_content = ""
                        if conversation and conversation[-1].get(
                                "content") and conversation[-1].get(
                                    "role") == role:
                            last_msg_content = conversation[-1][
                                "content"]

                        if last_msg_content:
                            yield f"data: {yield_first_chat(prompt_tokens, content=last_msg_content)}\n\n"
                first_iteration = False

                for output in res.outputs:
                    i = output.index

                    if finish_reason_sent[i]:
                        continue

                    delta_text = output.text_diff
                    if request.tool_choice and type(
                            request.tool_choice
                    ) is ChatCompletionNamedToolChoiceParam:
                        delta_message = DeltaMessage(tool_calls=[
                            ToolCall(function=FunctionCall(
                                name=request.tool_choice.function.name,
                                arguments=delta_text))
                        ])
                    else:
                        delta_message = DeltaMessage(content=delta_text)

                    choice = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=delta_message,
                        finish_reason=None)
                    if request.logprobs:
                        logprobs = output.logprobs_diff
                        token_ids = output.token_ids_diff
                        choice.logprobs = create_logprobs(token_ids, logprobs)
                    if output.finish_reason is not None:
                        choice.finish_reason = output.finish_reason
                        choice.stop_reason = output.stop_reason
                        finish_reason_sent[i] = True
                    chunk = ChatCompletionStreamResponse(
                        choices=[choice], model=self.model)
                    chunk.usage = stream_usage_info(
                        prompt_tokens, output.length)
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            if (request.stream_options
                    and request.stream_options.include_usage):
                completion_tokens = sum(output.length
                                        for output in promise.outputs)
                final_usage = UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )

                final_usage_chunk = ChatCompletionStreamResponse(
                    choices=[], model=self.model, usage=final_usage)
                final_usage_data = final_usage_chunk.model_dump_json()
                yield f"data: {final_usage_data}\n\n"
            yield f"data: [DONE]\n\n"

        async def create_chat_response(promise: RequestOutput) -> ChatCompletionResponse:
            await promise.aresult()
            choices: List[ChatCompletionResponseChoice] = []
            role = get_role()
            for output in promise.outputs:
                if request.tool_choice and isinstance(
                        request.tool_choice,
                        ChatCompletionNamedToolChoiceParam):
                    message = ChatMessage(
                        role=role,
                        content="",
                        tool_calls=[
                            ToolCall(function=FunctionCall(
                                name=request.tool_choice.function.name,
                                arguments=output.text))
                        ])
                else:
                    message = ChatMessage(role=role, content=output.text)
                choice = ChatCompletionResponseChoice(
                    index=output.index,
                    message=message,
                    finish_reason=output.finish_reason,
                    stop_reason=output.stop_reason,
                )

                if request.logprobs:
                    choice.logprobs = create_logprobs(output.token_ids, output.logprobs)
                choices.append(choice)

            if request.echo:
                last_msg_content = ""
                if conversation and conversation[-1].get(
                        "content") and conversation[-1].get("role") == role:
                    last_msg_content = conversation[-1]["content"]
                for choice in choices:
                    full_message = last_msg_content + choice.message.content
                    choice.message.content = full_message

            num_prompt_tokens = len(promise.prompt_token_ids)
            num_generated_tokens = sum(
                len(output.token_ids) for output in promise.outputs)
            usage = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_generated_tokens,
                total_tokens=num_prompt_tokens + num_generated_tokens,
            )
            response = ChatCompletionResponse(
                model=self.model,
                choices=choices,
                usage=usage,
            )
            return response

        try:
            conversation: List[ConversationMessage] = []
            for msg in request.messages:
                conversation.extend(parse_chat_message_content(msg))
            tool_dicts = None if request.tools is None else [
                tool.model_dump() for tool in request.tools
            ]
            prompt: str = self.tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt,
                tools=tool_dicts,
                documents=request.documents,
                chat_template=request.chat_template,
                **(request.chat_template_kwargs or {}),
            )
            sampling_params = request.to_sampling_params()

            promise = self.llm.generate_async(
                inputs=prompt,
                sampling_params=sampling_params,
                streaming=request.stream,
            )
            if request.stream:
                response_generator = chat_stream_generator(promise)
                return StreamingResponse(content=response_generator,
                                            media_type="text/event-stream")
            else:
                response = await create_chat_response(promise)
                return JSONResponse(content=response.model_dump())
        except CppExecutorError:
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            return self.create_error_response(str(e))

    async def openai_completion(self, request: CompletionRequest) -> Response:

        def merge_promises(promises: List[RequestOutput]) -> AsyncIterator[Tuple[int, RequestOutput]]:
            outputs = asyncio.Queue()
            finished = [False] * len(promises)

            async def producer(i: int, promise: RequestOutput):
                async for output in promise:
                    await outputs.put((i, output))
                finished[i] = True

            _tasks = [asyncio.create_task(producer(i, promise))
                for i, promise in enumerate(promises)
            ]

            async def consumer():
                while not all(finished) or not outputs.empty():
                    item = await outputs.get()
                    yield item
                await asyncio.gather(*_tasks)

            return consumer()

        async def create_completion_generator(generator: AsyncIterator[Tuple[int, RequestOutput]],
                                              num_choices: int):
            num_repsonse_per_request = 1 if request.n is None else request.n
            echoed = [False] * num_choices
            async for prompt_idx, requst_output in generator:
                prompt = requst_output.prompt
                for gen_idx, output in enumerate(requst_output.outputs):
                    response_idx = prompt_idx * num_repsonse_per_request + gen_idx
                    delta_text = output.text_diff
                    if request.echo and not echoed[response_idx]:
                        delta_text = prompt + delta_text
                        echoed[response_idx] = True
                    response = CompletionStreamResponse(
                        model=self.model,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=response_idx,
                                text=delta_text,
                                stop_reason=output.stop_reason,
                                finish_reason=output.finish_reason,
                            )
                        ])
                    response_json = response.model_dump_json(
                        exclude_unset=False)
                    yield f"data: {response_json}\n\n"
            yield f"data: [DONE]\n\n"

        async def create_completion_response(generator: AsyncIterator[Tuple[int, RequestOutput]],
                                             num_choices: int):
            choices = [None] * num_choices
            num_repsonse_per_request = 1 if request.n is None else request.n
            num_prompt_tokens = num_gen_tokens = 0
            async for prompt_idx, request_output in generator:
                num_prompt_tokens += len(request_output.prompt_token_ids)
                for gen_idx, output in enumerate(request_output.outputs):
                    num_gen_tokens += len(output.token_ids)
                    output_text = output.text
                    if request.echo:
                        output_text = request_output.prompt + output_text
                    idx = prompt_idx * num_repsonse_per_request + gen_idx

                    disaggregated_params = CompletionResponseChoice.to_disaggregated_params(output.disaggregated_params)
                    choice = CompletionResponseChoice(
                        index=idx,
                        text=output_text,
                        stop_reason=output.stop_reason,
                        finish_reason=output.finish_reason,
                        disaggregated_params=disaggregated_params,
                    )
                    choices[idx] = choice

            usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_gen_tokens,
                total_tokens=num_gen_tokens + num_prompt_tokens,
            )
            response = CompletionResponse(
                model=self.model,
                choices=choices,
                usage=usage_info,
            )
            return response

        try:
            if isinstance(request.prompt, str) or \
                (isinstance(request.prompt, list) and isinstance(request.prompt[0], int)):
                prompts = [request.prompt]
            else:
                prompts = request.prompt

            promises: List[RequestOutput] = []
            sampling_params = request.to_sampling_params()
            disaggregated_params = request.to_llm_disaggregated_params()
            for prompt in prompts:
                promise = self.llm.generate_async(
                    inputs=prompt,
                    sampling_params=sampling_params,
                    streaming=request.stream,
                    disaggregated_params=disaggregated_params
                )
                promises.append(promise)
            generator = merge_promises(promises)
            num_choices = len(prompts) if request.n is None else len(prompts) * request.n
            if request.stream:
                response_generator = create_completion_generator(generator, num_choices)
                return StreamingResponse(content=response_generator,
                                         media_type="text/event-stream")
            else:
                response = await create_completion_response(generator, num_choices)
                return JSONResponse(content=response.model_dump())
        except CppExecutorError:
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            return self.create_error_response(str(e))

    async def __call__(self, host, port):
        config = uvicorn.Config(self.app,
                                host=host,
                                port=port,
                                log_level="info",
                                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
        await uvicorn.Server(config).serve()
