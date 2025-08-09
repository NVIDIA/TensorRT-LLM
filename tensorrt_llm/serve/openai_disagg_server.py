#!/usr/bin/env python
import asyncio
import copy
import os
import signal
import traceback
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import List, Optional, Type, Union

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

# yapf: disable
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.llmapi.disagg_utils import (ConditionalDisaggConfig,
                                              MetadataServerConfig,
                                              RouterConfig)
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.metadata_server import create_metadata_server
from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                ChatCompletionResponse,
                                                CompletionRequest,
                                                CompletionResponse,
                                                DisaggregatedParams,
                                                ErrorResponse)
from tensorrt_llm.serve.router import KvCacheAwareRouter, create_router
from tensorrt_llm.version import __version__ as VERSION

# yapf: enale
TIMEOUT_KEEP_ALIVE = 10  # seconds.

class OpenAIDisaggServer:

    def __init__(self,
                 ctx_servers: List[str],
                 gen_servers: List[str],
                 req_timeout_secs: int = 180,
                 server_start_timeout_secs: int = 180,
                 max_retries: int = 3,
                 ctx_router_config: Optional[RouterConfig] = None,
                 gen_router_config: Optional[RouterConfig] = None,
                 conditional_disagg_config: Optional[ConditionalDisaggConfig] = None,
                 metadata_server_cfg: Optional[MetadataServerConfig] = None):

        self.ctx_servers = ctx_servers
        self.gen_servers = gen_servers
        self.metadata_server = create_metadata_server(metadata_server_cfg)
        self.ctx_router = create_router(ctx_router_config, ctx_servers, metadata_server_cfg, self.metadata_server)
        self.gen_router = create_router(gen_router_config, gen_servers, metadata_server_cfg, self.metadata_server)
        self.conditional_disagg_config = conditional_disagg_config

        if max_retries < 0:
            raise ValueError(f"Max retries {max_retries} must be greater than or equal to 0")
        self.max_retries = max_retries
        logger.info(f"Server max retries: {self.max_retries}")

        if (len(self.gen_servers) == 0):
            raise ValueError("At least one generation server must be provided")

        if os.getenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY") != "1" and len(ctx_servers) == 0:
            raise ValueError("At least one context server must be provided")

        if self.conditional_disagg_config is not None and not isinstance(self.gen_router, KvCacheAwareRouter):
            raise ValueError("Generation router must be a KvCacheAwareRouter to enable conditional disaggregation")

        # Session will be initialized in lifespan
        self.session: Optional[aiohttp.ClientSession] = None

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Create a persistent aiohttp ClientSession
            self.session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0, limit_per_host=0, force_close=True),
                timeout=aiohttp.ClientTimeout(total=req_timeout_secs))

            logger.info("Waiting for context and generation servers to be ready")
            await self.wait_for_servers_ready(server_start_timeout_secs)

            if self.metadata_server:
                logger.info("Starting server monitoring via metadata service")
                await self.ctx_router.start_server_monitoring(metadata_server_cfg.refresh_interval)
                await self.gen_router.start_server_monitoring(metadata_server_cfg.refresh_interval)

            yield

            if self.metadata_server:
                logger.info("Stopping server monitoring via metadata service")
                await self.ctx_router.stop_server_monitoring()
                await self.gen_router.stop_server_monitoring()

            await self.session.close()  # Ensure session cleanup

        self.app = FastAPI(lifespan=lifespan)

        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(_, exc):
            return JSONResponse(status_code=400, content={"error": str(exc)})

        self.register_routes()

    @staticmethod
    def create_error_response(
            message: str,
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
        raise HTTPException(status_code=500, detail=f"Internal server error {message}")

    def register_routes(self):
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/version", self.version, methods=["GET"])
        self.app.add_api_route("/v1/completions",
                               self.openai_completion,
                               methods=["POST"])
        self.app.add_api_route("/v1/chat/completions",
                               self.openai_chat_completion,
                               methods=["POST"])

    async def health(self) -> Response:
        return Response(status_code=200)

    async def version(self) -> JSONResponse:
        ver = {"version": VERSION}
        return JSONResponse(content=ver)

    async def merge_streaming_responses(self, ctx_response,
                                        gen_server: str,
                                        gen_req: Union[CompletionRequest, ChatCompletionRequest]):
        try:

            if ctx_response is not None and len(ctx_response.choices) != 1:
                raise ValueError("Context server did not return a single choice. This is not expected")

            #If request finished after first token not due to length, return right away and skip gen
            if ctx_response is not None and ctx_response.choices[0].finish_reason not in ["length", "not_finished"]:
                yield f"data: [DONE]\n\n".encode('utf-8')
            else:
                # Then yield the generation responses
                if isinstance(gen_req, CompletionRequest):
                    gen_response = await self.send_completion_request(gen_server, gen_req)
                elif isinstance(gen_req, ChatCompletionRequest):
                    gen_response = await self.send_chat_request(gen_server, gen_req)
                else:
                    raise TypeError("Invalid request type: {type(gen_req).__name__}")

                async for chunk in gen_response.body_iterator:
                    yield chunk

        finally:
            await self.gen_router.finish_request(gen_req)

    async def openai_completion(self, req: CompletionRequest) -> Response:
        try:
            if not isinstance(req.prompt, str):
                # Check if it's a list and contains integers
                if type(req.prompt) is list and len(req.prompt) == 1:
                    req.prompt = req.prompt[0]
                elif not isinstance(req.prompt, list) or not all(isinstance(x, int) for x in req.prompt):
                    raise ValueError("Disaggregated server currently only supports single string prompt or list of integers in request")

            return await self._send_disagg_request(req)

        except Exception as e:
            await self._handle_exception(e)

    async def openai_chat_completion(self, req: ChatCompletionRequest) -> Response:

        try:
            return await self._send_disagg_request(req)
        except Exception as e:
            await self._handle_exception(e)

    async def _handle_exception(self, exception):
        if isinstance(exception, CppExecutorError):
            logger.error(traceback.format_exc())
            signal.raise_signal(signal.SIGINT)
        elif isinstance(exception, HTTPException):
            raise exception  # Re-raise HTTP exceptions properly
        else:
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Internal server error {str(exception)}")

    async def _send_context_request(self, ctx_server: str, ctx_req: Union[CompletionRequest, ChatCompletionRequest]):

        ctx_req.disaggregated_params = DisaggregatedParams(request_type="context_only")
        ctx_req.stream = False
        ctx_req.stream_options = None

        logger.debug("Sending request to ctx server: %s", ctx_server)
        try:
            if isinstance(ctx_req, ChatCompletionRequest):
                ctx_response = await self.send_chat_request(ctx_server, ctx_req)
            else:
                assert isinstance(ctx_req, CompletionRequest)
                ctx_response = await self.send_completion_request(ctx_server, ctx_req)
        finally:
            await self.ctx_router.finish_request(ctx_req)

        choices = ctx_response.choices
        if len(choices) > 1:
            raise ValueError("Disagg server returned more than one choice. This is currently not supported in disaggregated server.")
        if choices[0].disaggregated_params is None:
            raise ValueError("Context server did not return disaggregated params")
        if choices[0].disaggregated_params.ctx_request_id is None:
            raise ValueError("Invalid disaggregated params in context phase response.")

        return ctx_response

    async def _send_disagg_request(self, req: Union[CompletionRequest, ChatCompletionRequest]):
        gen_server = None
        need_ctx = False
        try:
            # Determine if need context server
            condition = self.conditional_disagg_config
            if condition is not None:
                assert isinstance(self.gen_router, KvCacheAwareRouter)
                # Query kv cache status and select a best gen_server.
                # The server is reserved for generation request
                gen_server, info = await self.gen_router.get_next_server(req)
                match_length = sum(info["matches"])
                total_length = sum(len(token_list) for token_list in info["token_lists"])
                if match_length == 0 or total_length - match_length > condition.max_local_prefill_length:
                    need_ctx = True
            elif os.getenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY") == "1":
                # Hard-code first token, ctx_request_id for testing
                req.disaggregated_params = DisaggregatedParams(
                    request_type="generation_only",
                    first_gen_tokens=[7],
                    ctx_request_id=1,
                    encoded_opaque_state=None,
                    draft_tokens=None)
                # Since KV cache for prompt tokens will be uninitialized, need to ignore eos
                req.ignore_eos = True
            else:
                need_ctx = True

            if need_ctx:
                ctx_req = copy.deepcopy(req)
                ctx_server, _ = await self.ctx_router.get_next_server(ctx_req)
                # TODO: add ctx_server info into generation request for pre-registration
                ctx_response = await self._send_context_request(ctx_server, ctx_req)

                if ctx_response is not None and len(ctx_response.choices) != 1:
                    raise ValueError("Context server did not return a single choice. This is not expected")

                # Append disaggregates parameters to generation request
                req.disaggregated_params = ctx_response.choices[0].disaggregated_params
                req.disaggregated_params.request_type = "generation_only"

                # Replace the string prompt with prompt_tokens_ids
                if isinstance(req, CompletionRequest):
                    req.prompt = ctx_response.prompt_token_ids
                elif isinstance(req, ChatCompletionRequest):
                    req.prompt_token_ids = ctx_response.prompt_token_ids
                else:
                    raise ValueError("Invalid request type: {type(req).__name__}")
            else:
                ctx_response = None

            # Pick a generation server if haven't reserved one, and send request
            if gen_server is None:
                gen_server, _ = await self.gen_router.get_next_server(req)
            logger.debug("Sending request to gen server: %s", gen_server)

            if not req.stream:
                try:
                    #If request finished after first token for reason other than length, return right away and skip gen
                    if ctx_response is not None and ctx_response.choices[0].finish_reason not in ["length","not_finished"]:
                        del ctx_response.choices[0].disaggregated_params
                        return ctx_response
                    else:
                        if isinstance(req, CompletionRequest):
                            gen_response = await self.send_completion_request(gen_server, req)
                        else:
                            assert isinstance(req, ChatCompletionRequest)
                            gen_response = await self.send_chat_request(gen_server, req)
                        return gen_response
                finally:
                    if gen_server is not None:
                        await self.gen_router.finish_request(req)

            else:
                # Return a streaming response that combines both context and generation responses
                return StreamingResponse(
                    self.merge_streaming_responses(ctx_response, gen_server, req),
                    media_type="text/event-stream"
                )
        except:
            if gen_server is not None:
                await self.gen_router.finish_request(req)
            raise


    async def __call__(self, host, port):
        config = uvicorn.Config(self.app,
                                host=host,
                                port=port,
                                log_level="info",
                                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
        await uvicorn.Server(config).serve()

    async def create_generator(self, url: str, request: Union[CompletionRequest, ChatCompletionRequest], end_point: str):
        async with self.session.post(url + end_point, json=request.model_dump(exclude_unset=True)) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" in content_type:
                if not request.stream:
                    raise ValueError("Received an event-stream although request stream was False")

                try:
                    async for line in response.content.iter_any():
                        if line:
                            yield line
                            await asyncio.sleep(0)
                except Exception as e:
                    logger.error(f"Unexpected error in stream: {e}")
                    raise

    async def create_completion_generator(self, url: str, request: CompletionRequest):
        async for chunk in self.create_generator(url, request, "/v1/completions"):
            yield chunk

    async def create_chat_generator(self, url: str, request: ChatCompletionRequest):
        async for chunk in self.create_generator(url, request, "/v1/chat/completions"):
            yield chunk

    async def send_request(self, url: str,
                           request: Union[CompletionRequest, ChatCompletionRequest],
                           endpoint: str,
                           response_type: Type[Union[CompletionResponse, ChatCompletionResponse]],
                           create_generator: callable) -> Union[CompletionResponse, ChatCompletionResponse, StreamingResponse]:
        for attempt in range(self.max_retries + 1):
            try:
                if request.stream:
                    response_generator = create_generator(url, request)
                    return StreamingResponse(content=response_generator, media_type="text/event-stream")
                else:
                    async with self.session.post(url + endpoint, json=request.model_dump(exclude_unset=True)) as response:
                        content_type = response.headers.get("Content-Type", "")
                        if "text/event-stream" in content_type:
                            raise ValueError("Received an event-stream although request stream was False")

                        response_dict = await response.json()
                        if not response.ok:
                            logger.error(f"Received failed response {response_dict}")
                            response.raise_for_status()
                        return response_type(**response_dict)
            except (aiohttp.ClientError, OSError) as e:
                if attempt == self.max_retries:
                    raise HTTPException(status_code=HTTP_429_TOO_MANY_REQUESTS, detail=f"Too many requests") from e
                logger.error(f"Client error: {e} - retry {attempt} of {self.max_retries}")
                # TODO : add a configurable retry interval
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error encountered while processing request to {url+endpoint}: {e}")
                raise


    async def send_completion_request(self, url: str, request: CompletionRequest) -> Union[CompletionResponse, StreamingResponse]:
        return await self.send_request(url, request, "/v1/completions", CompletionResponse, self.create_completion_generator)

    async def send_chat_request(self, url: str, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return await self.send_request(url, request, "/v1/chat/completions", ChatCompletionResponse, self.create_chat_generator)

    @classmethod
    async def check_server_ready(cls, session: aiohttp.ClientSession, server_url: str) -> bool:
        try:
            async with session.get(server_url+"/health") as response:
                return response.status == 200
        except Exception:
            return False

    @classmethod
    async def wait_for_all_servers_ready(cls, session: aiohttp.ClientSession,
                                         ctx_servers: list[str],
                                         gen_servers: list[str],
                                         server_start_timeout_secs: int = 180):
        async def get_unready_servers(servers: list[str]) -> list[str]:
            servers_ready = await asyncio.gather(*[cls.check_server_ready(session, server) for server in servers])
            return [server for server, ready in zip(servers, servers_ready) if not ready]

        async def check_all_servers_ready():
            iter = 0
            unready_servers = await get_unready_servers(ctx_servers + gen_servers)
            while len(unready_servers) > 0:
                wait_time = 3
                logger.info(
                    f"[{iter}] Servers are not ready. Waiting for {unready_servers}..."
                )
                await asyncio.sleep(wait_time)
                iter += 1
                unready_servers = await get_unready_servers(unready_servers)
        try:
            await asyncio.wait_for(check_all_servers_ready(), timeout=server_start_timeout_secs)
        except asyncio.CancelledError:
            raise TimeoutError("Timeout waiting for context and generation servers to be ready")
        logger.info("Context and generation servers are ready")

    async def wait_for_servers_ready(self, server_start_timeout_secs: int = 180):
        await self.wait_for_all_servers_ready(self.session, self.ctx_servers, self.gen_servers, server_start_timeout_secs)
