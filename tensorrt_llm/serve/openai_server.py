#!/usr/bin/env python
import asyncio
import signal
import traceback
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import AsyncGenerator, AsyncIterator, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers import AutoConfig, AutoProcessor

# yapf: disable
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.executor.postproc_worker import PostprocParams
from tensorrt_llm.inputs import prompt_inputs
from tensorrt_llm.llmapi import LLM
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.chat_utils import (ConversationMessage,
                                           apply_chat_template,
                                           parse_chat_messages_coroutines)
from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                ChatCompletionResponse,
                                                CompletionRequest,
                                                CompletionResponse,
                                                CompletionResponseChoice,
                                                ErrorResponse, ModelCard,
                                                ModelList, UsageInfo,
                                                to_llm_disaggregated_params)
from tensorrt_llm.serve.postprocess_handlers import (
    ChatPostprocArgs, CompletionPostprocArgs, chat_response_post_processor,
    chat_stream_post_processor, completion_response_post_processor,
    completion_stream_post_processor)
from tensorrt_llm.version import __version__ as VERSION

from .._utils import nvtx_mark

# yapf: enale
TIMEOUT_KEEP_ALIVE = 5  # seconds.


class OpenAIServer:

    def __init__(self,
                 llm: LLM,
                 model: str):
        self.llm = llm
        self.tokenizer = llm.tokenizer
        try:
            hf_tokenizer_path = llm._hf_model_dir or self.tokenizer.tokenizer.name_or_path
            self.processor = AutoProcessor.from_pretrained(hf_tokenizer_path)
            self.model_config = AutoConfig.from_pretrained(hf_tokenizer_path)
        except Exception:
            logger.debug("Failed to load AutoProcessor or AutoConfig for %s", hf_tokenizer_path)
            self.processor = None
            self.model_config = None

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

    async def await_disconnected(self, raw_request: Request, promise):
        while not await raw_request.is_disconnected():
            await asyncio.sleep(1)
        if not promise.finished:
            promise.abort()
            logger.info(
                f"{raw_request.client} is disconnected, abort {promise.request_id}")

    @property
    def postproc_worker_enabled(self) -> bool:
        return True if self.llm.args.num_postprocess_workers > 0 else False

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
        # TODO: the metrics endpoint only reports iteration stats, not the runtime stats for now
        self.app.add_api_route("/metrics", self.get_iteration_stats, methods=["GET"])
        # TODO: workaround before ETCD support
        self.app.add_api_route("/kv_cache_events", self.get_kv_cache_events, methods=["POST"])
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

    async def get_iteration_stats(self) -> JSONResponse:
        stats = []
        async for stat in self.llm.get_stats_async(2):
            stats.append(stat)
        return JSONResponse(content=stats)

    async def get_kv_cache_events(self) -> JSONResponse:
        events = []
        try:
            async for event in self.llm.get_kv_cache_events_async(2):
                events.append(event)
        except IndexError:
            # queue is empty, no more events
            pass
        return JSONResponse(content=events)

    async def openai_chat(self, request: ChatCompletionRequest, raw_request: Request) -> Response:

        def get_role() -> str:
            if request.add_generation_prompt:
                role = "assistant"
            else:
                role = request.messages[-1]["role"]
            return role

        async def chat_stream_generator(
                promise: RequestOutput, postproc_params: PostprocParams) -> AsyncGenerator[str, None]:
            if not self.postproc_worker_enabled:
                post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
            async for res in promise:
                pp_results = res.outputs[0]._postprocess_result if self.postproc_worker_enabled else post_processor(res, args)
                for pp_res in pp_results:
                    yield pp_res
            yield f"data: [DONE]\n\n"
            nvtx_mark("generation ends")

        async def create_chat_response(
                promise: RequestOutput, postproc_params: PostprocParams) -> ChatCompletionResponse:
            await promise.aresult()
            if self.postproc_worker_enabled:
                return promise.outputs[0]._postprocess_result
            else:
                post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                return post_processor(promise, args)

        try:
            conversation: List[ConversationMessage] = []
            tool_dicts = None if request.tools is None else [
                tool.model_dump() for tool in request.tools
            ]
            sampling_params = request.to_sampling_params()
            postproc_args = ChatPostprocArgs.from_request(request)
            disaggregated_params = to_llm_disaggregated_params(request.disaggregated_params)

            conversation, mm_coroutines = parse_chat_messages_coroutines(request.messages, self.model_config)

            prompt: str = apply_chat_template(
                tokenizer=self.tokenizer,
                processor=self.processor,
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt,
                tools=tool_dicts,
                documents=request.documents,
                chat_template=request.chat_template,
                chat_template_kwargs=request.chat_template_kwargs or {})
            prompt = prompt_inputs(prompt)

            mm_data = await mm_coroutines
            if mm_data is not None:
                prompt["multi_modal_data"] = mm_data

            postproc_args.reasoning_parser = self.llm.args.reasoning_parser
            if conversation and conversation[-1].get(
                    "content") and conversation[-1].get("role") == get_role():
                postproc_args.last_message_content = conversation[-1]["content"]
            postproc_params = PostprocParams(
                post_processor=chat_stream_post_processor
                if request.stream else chat_response_post_processor,
                postproc_args=postproc_args,
            )

            promise = self.llm.generate_async(
                inputs=prompt,
                sampling_params=sampling_params,
                _postproc_params=postproc_params if self.postproc_worker_enabled else None,
                streaming=request.stream,
                disaggregated_params=disaggregated_params
            )
            asyncio.create_task(self.await_disconnected(raw_request, promise))
            if not self.postproc_worker_enabled:
                postproc_args.tokenizer = self.tokenizer
                postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)

            if request.stream:
                response_generator = chat_stream_generator(promise, postproc_params)
                return StreamingResponse(content=response_generator,
                                         media_type="text/event-stream")
            else:
                response = await create_chat_response(promise, postproc_params)
                return JSONResponse(content=response.model_dump())
        except CppExecutorError:
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            return self.create_error_response(str(e))

    async def openai_completion(self, request: CompletionRequest, raw_request: Request) -> Response:

        def merge_promises(
            promises: List[RequestOutput],
            postproc_params_collections: List[Optional[PostprocParams]]
        ) -> AsyncIterator[Tuple[RequestOutput, Optional[PostprocParams]]]:
            outputs = asyncio.Queue()
            finished = [False] * len(promises)

            async def producer(i: int, promise: RequestOutput, postproc_params: Optional[PostprocParams]):
                async for output in promise:
                    await outputs.put((output, postproc_params))
                finished[i] = True

            _tasks = [
                asyncio.create_task(producer(i, promise, postproc_params))
                for i, (promise, postproc_params) in enumerate(zip(promises, postproc_params_collections))
            ]

            async def consumer():
                while not all(finished) or not outputs.empty():
                    item = await outputs.get()
                    yield item
                await asyncio.gather(*_tasks)

            return consumer()

        async def create_completion_generator(
                generator: AsyncIterator[Tuple[RequestOutput, Optional[PostprocParams]]]):
            async for request_output, postproc_params in generator:
                if not self.postproc_worker_enabled:
                    post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                    pp_result = post_processor(request_output, args)
                else:
                    pp_result = request_output.outputs[0]._postprocess_result
                for pp_res in pp_result:
                    yield pp_res
            yield f"data: [DONE]\n\n"

        async def create_completion_response(
                generator: AsyncIterator[Tuple[RequestOutput, Optional[PostprocParams]]]) -> CompletionResponse:
            all_choices: List[CompletionResponseChoice] = []
            num_prompt_tokens = num_gen_tokens = 0
            async for request_output, postproc_params in generator:
                pp_result: CompletionResponse
                if not self.postproc_worker_enabled:
                    post_processor, args = postproc_params.post_processor, postproc_params.postproc_args
                    pp_result = post_processor(request_output, args)
                else:
                    pp_result = request_output.outputs[0]._postprocess_result

                choices, usage = pp_result.choices, pp_result.usage
                all_choices.extend(choices)
                num_prompt_tokens += usage.prompt_tokens
                num_gen_tokens += usage.completion_tokens

            usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_gen_tokens,
                total_tokens=num_gen_tokens + num_prompt_tokens,
            )
            response = CompletionResponse(
                model=self.model,
                choices=all_choices,
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
            postproc_params_collection: List[Optional[PostprocParams]] = []
            sampling_params = request.to_sampling_params()
            disaggregated_params = to_llm_disaggregated_params(request.disaggregated_params)
            for idx, prompt in enumerate(prompts):
                postproc_args = CompletionPostprocArgs.from_request(request)
                postproc_args.prompt_idx = idx
                if request.echo:
                    postproc_args.prompt = prompt
                postproc_params = PostprocParams(
                    post_processor=completion_stream_post_processor
                    if request.stream else completion_response_post_processor,
                    postproc_args=postproc_args,
                )
                promise = self.llm.generate_async(
                    inputs=prompt,
                    sampling_params=sampling_params,
                    _postproc_params=postproc_params,
                    streaming=request.stream,
                    disaggregated_params=disaggregated_params
                )
                asyncio.create_task(self.await_disconnected(raw_request, promise))
                if not self.postproc_worker_enabled:
                    postproc_args.tokenizer = self.tokenizer
                    postproc_args.num_prompt_tokens = len(promise.prompt_token_ids)
                promises.append(promise)
                postproc_params_collection.append(None if self.postproc_worker_enabled else postproc_params)

            generator = merge_promises(promises, postproc_params_collection)
            if request.stream:
                response_generator = create_completion_generator(
                    generator)
                return StreamingResponse(content=response_generator,
                                            media_type="text/event-stream")
            else:
                response = await create_completion_response(
                    generator)
                return JSONResponse(content=response.model_dump())
        except CppExecutorError:
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            print(f"Encountered an exception: {str(e)}")
            traceback.print_exc()
            return self.create_error_response(str(e))

    async def __call__(self, host, port):
        config = uvicorn.Config(self.app,
                                host=host,
                                port=port,
                                log_level="info",
                                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
        await uvicorn.Server(config).serve()
