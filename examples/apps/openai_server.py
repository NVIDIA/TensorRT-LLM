#!/usr/bin/env python
import asyncio
import logging
from http import HTTPStatus
from pathlib import Path
from typing import (AsyncGenerator, AsyncIterator, List, Optional, Tuple,
                    TypedDict)

import click
import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openai.types.chat import ChatCompletionMessageParam
from transformers import AutoTokenizer, PreTrainedTokenizer

# yapf: disable
from tensorrt_llm.hlapi import LLM, BuildConfig, KvCacheConfig
from tensorrt_llm.hlapi.llm import RequestOutput
from tensorrt_llm.hlapi.openai_protocol import (
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


class OpenaiServer:

    def __init__(self,
                 llm: LLM,
                 model: str,
                 kv_cache_config: KvCacheConfig,
                 hf_tokenizer: PreTrainedTokenizer = None):
        self.llm = llm
        self.model = model
        self.kv_cache_config = kv_cache_config
        self.tokenizer = hf_tokenizer

        self.app = FastAPI()

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
        model_dir = Path(self.model)
        if model_dir.exists() and model_dir.is_dir():
            model = model_dir.name
        else:
            model = self.model
        model_list = ModelList(data=[ModelCard(id=model)])
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

        async def chat_stream_generator(promise: RequestOutput) -> AsyncGenerator[str, None]:
            first_iteration = True
            num_choices = 1 if request.n is None else request.n
            finish_reason_sent = [False] * num_choices
            role = get_role()
            async for res in promise:
                prompt_tokens = len(res.prompt_token_ids)
                if first_iteration:
                    # Send first response for each request.n (index) with
                    # the role
                    for i in range(num_choices):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(role=role),
                            logprobs=None,
                            finish_reason=None)
                        chunk = ChatCompletionStreamResponse(
                            choices=[choice_data], model=request.model)
                        chunk.usage = stream_usage_info(
                            prompt_tokens, 0)

                        data = chunk.model_dump_json()
                        yield f"data: {data}\n\n"

                    if request.echo:
                        last_msg_content = ""
                        if conversation and conversation[-1].get(
                                "content") and conversation[-1].get(
                                    "role") == role:
                            last_msg_content = conversation[-1][
                                "content"]

                        if last_msg_content:
                            for i in range(num_choices):
                                choice_data = (
                                    ChatCompletionResponseStreamChoice(
                                        index=i,
                                        delta=DeltaMessage(
                                            content=last_msg_content),
                                        logprobs=None,
                                        finish_reason=None))
                                chunk = ChatCompletionStreamResponse(
                                    choices=[choice_data],
                                    model=request.model)
                                chunk.usage = stream_usage_info(
                                    prompt_tokens, 0)
                                data = chunk.model_dump_json(
                                    exclude_unset=True)
                            yield f"data: {data}\n\n"
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

                    if delta_text:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            finish_reason=None)
                        chunk = ChatCompletionStreamResponse(
                            choices=[choice_data], model=request.model)
                        chunk.usage = stream_usage_info(
                            prompt_tokens, output.length)
                        data = chunk.model_dump_json()
                        yield f"data: {data}\n\n"
                    else:
                        finish_reason_sent[i] = True

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
                    choices=[], model=request.model, usage=final_usage)
                final_usage_data = final_usage_chunk.model_dump_json()
                yield f"data: {final_usage_data}\n\n"

        async def create_chat_response(promise: RequestOutput) -> JSONResponse:
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
                )
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
                model=request.model,
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

            promise = self.llm.generate_async(prompt, sampling_params,
                                              request.stream)
            if request.stream:
                response_generator = chat_stream_generator(promise)
                return StreamingResponse(content=response_generator,
                                         media_type="text/event-stream")
            else:
                response = await create_chat_response(promise)
                return JSONResponse(content=response.model_dump())

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
                        model=request.model,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=response_idx, text=delta_text)
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
                    choice = CompletionResponseChoice(
                        index=idx,
                        text=output_text,
                    )
                    choices[idx] = choice

            usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_gen_tokens,
                total_tokens=num_gen_tokens + num_prompt_tokens,
            )
            response = CompletionResponse(
                model=request.model,
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
            for prompt in prompts:
                promise = self.llm.generate_async(prompt, sampling_params,
                                                    request.stream)
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
        except Exception as e:
            return self.create_error_response(str(e))

    async def __call__(self, host, port):
        config = uvicorn.Config(self.app,
                                host=host,
                                port=port,
                                log_level="info",
                                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
        await uvicorn.Server(config).serve()


@click.command()
@click.argument("model_dir")
@click.option("--tokenizer", type=str, default=None)
@click.option("--host", type=str, default=None)
@click.option("--port", type=int, default=8000)
@click.option("--max_beam_width", type=int, default=1)
@click.option("--tp_size", type=int, default=1)
@click.option("--pp_size", type=int, default=1)
def entrypoint(model_dir: str,
               tokenizer: Optional[str] = None,
               host: Optional[str] = None,
               port: int = 8000,
               max_beam_width: int = 1,
               tp_size: int = 1,
               pp_size: int = 1):
    host = host or "0.0.0.0"
    port = port or 8000
    logging.info(f"Starting server at {host}:{port}")

    build_config = BuildConfig(max_batch_size=10, max_beam_width=max_beam_width)

    llm = LLM(model_dir,
              tokenizer,
              tensor_parallel_size=tp_size,
              pipeline_parallel_size=pp_size,
              build_config=build_config)

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8)
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer or model_dir)

    server = OpenaiServer(llm=llm,
                          model=model_dir,
                          kv_cache_config=kv_cache_config,
                          hf_tokenizer=hf_tokenizer)

    asyncio.run(server(host, port))


if __name__ == "__main__":
    entrypoint()
