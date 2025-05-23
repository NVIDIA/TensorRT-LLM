#!/usr/bin/env python
import asyncio
import copy
import json
import logging
import os
import signal
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import List, Optional, Type, Union, Dict, Any

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse

# yapf: disable
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.llmapi.disagg_utils import RouterConfig
from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                ChatCompletionResponse,
                                                CompletionRequest,
                                                CompletionResponse,
                                                ErrorResponse)
from tensorrt_llm.multimodal_params import MultimodalParams

from tensorrt_llm.serve.router import create_router
from tensorrt_llm.version import __version__ as VERSION

logging.basicConfig(level=logging.INFO)

# yapf: enale
TIMEOUT_KEEP_ALIVE = 10  # seconds.

class OpenAIMultiModalDisaggServer:

    def __init__(self,
                 gen_servers: List[str] = None,
                 mm_servers: List[str] = None,
                 req_timeout_secs: int = 180,
                 server_start_timeout_secs: int = 180,
                 ctx_router_config: Optional[RouterConfig] = None,
                 gen_router_config: Optional[RouterConfig] = None):

        self.ctx_servers = None
        self.gen_servers = gen_servers
        self.mm_servers = mm_servers
        assert len(mm_servers) == 1, "Currently only one multimodal server is supported"
        # We should remove this restriction pretty soon (also need to modify the broadcast mm_embed logic in model runner)
        assert len(gen_servers) == 1, "Currently only one generation server is supported"
        #self.gen_router = create_router(gen_router_config, gen_servers)
        #self.ctx_router = create_router(ctx_router_config, ctx_servers)

        #self.mm_router = create_router(mm_router_config, mm_servers)

        assert os.getenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY") != "1", "Multimodal disaggregated mode is not supported in disaggregated_gen benchmark mode"

        # Session will be initialized in lifespan
        self.session: Optional[aiohttp.ClientSession] = None

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Create a persistent aiohttp ClientSession
            self.session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0, limit_per_host=0, keepalive_timeout=300),
                timeout=aiohttp.ClientTimeout(total=req_timeout_secs))

            logging.info("Waiting for context and generation servers to be ready")
            await self.wait_for_servers_ready(server_start_timeout_secs)
            yield
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
            # First yield the context response if it's not None
            if ctx_response is not None:
                # Remove the disaggregated params from the context response
                data = ctx_response.model_dump()
                del data['choices'][0]['disaggregated_params']
                data = json.dumps(data)
                yield f"data: {data}\n\n".encode('utf-8')

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
            #await self.gen_router.finish_request(gen_req)
            pass

    async def openai_completion(self, req: CompletionRequest) -> Response:
        # TODO: support completion mode later
        assert len(self.mm_servers) == 0, "Multimodal disaggregated mode is not supported in completion mode yet"
        try:
            gen_req = copy.deepcopy(req)
            if not isinstance(req.prompt, str):
                # Check if it's a list and contains integers
                if type(req.prompt) is list and len(req.prompt) == 1:
                    req.prompt = req.prompt[0]
                elif not isinstance(req.prompt, list) or not all(isinstance(x, int) for x in req.prompt):
                    raise ValueError("Disaggregated server currently only supports single string prompt or list of integers in request")

            ctx_response = await self._process_context_server_request(req, "completion")

            return await self._process_generation_server_request(gen_req, ctx_response)

        except Exception as e:
            await self._handle_exception(e)

    async def openai_chat_completion(self, req: ChatCompletionRequest) -> Response:
        try:
            # Step 1: Process multimodal request and get response
            mm_req = copy.deepcopy(req)
            mm_response = await self._process_multimodal_server_request(mm_req)

            # Step 2: Append multimodal response directly to the original request
            if mm_response and 'embeddings' in mm_response:
                req.mm_params = MultimodalParams(**mm_response)

            return await self._process_generation_server_request(req)

        except Exception as e:
            await self._handle_exception(e)

    async def _handle_exception(self, exception):
        if isinstance(exception, CppExecutorError):
            logging.exception(exception)
            signal.raise_signal(signal.SIGINT)
        elif isinstance(exception, HTTPException):
            raise exception  # Re-raise HTTP exceptions properly
        else:
            logging.exception(exception)
            raise HTTPException(status_code=500, detail=f"Internal server error {str(exception)}")

    async def _process_multimodal_server_request(self, mm_req: ChatCompletionRequest) -> Optional[Dict[str, Any]]:
        """
        Process multimodal request and return response from multimodal server.

        Returns:
            Optional[Dict[str, Any]]: Response dictionary from multimodal server or None if processing failed
        """
        try:
            # Disable streaming for multimodal requests
            mm_req.stream = False

            # Send request to multimodal server
            async with self.session.post(
                self.mm_servers[0] + "/v1/multimodal_encoder",
                json=mm_req.model_dump(exclude_unset=True)
            ) as response:
                if not response.ok:
                    error_msg = f"Multimodal server returned error: {response.status} {response.reason}"
                    logging.error(error_msg)
                    raise HTTPException(status_code=response.status, detail=error_msg)
                return await response.json()

        except Exception as e:
            logging.error(f"Unexpected error in multimodal processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Multimodal processing failed: {str(e)}")

    async def _process_generation_server_request(self, gen_req, ctx_response=None):
        if ctx_response is not None:
            choices = ctx_response.choices
            if len(choices) > 1:
                raise ValueError("Disagg server returned more than one choice. This is currently not supported in disaggregated server.")
            if choices[0].disaggregated_params is None:
                raise ValueError("Context server did not return disaggregated params")

            # Append disaggregates parameters to generation request
            gen_req.disaggregated_params = choices[0].disaggregated_params
            gen_req.disaggregated_params.request_type = "generation_only"
        else:
            if gen_req.disaggregated_params is not None:
                # TODO: support E+PD for now; later we can support E+P+D
                del gen_req.disaggregated_params

        # Pick a generation server and send request
        # gen_server, _ = await self.gen_router.get_next_server(gen_req)
        # TODO: support gen_server routing
        gen_server = self.gen_servers[0]

        if not gen_req.stream:
            try:
                if isinstance(gen_req, CompletionRequest):
                    # TODO: support completion mode later
                    assert 0, "Completion mode is not supported in multimodal disaggregated mode yet"
                    gen_response = await self.send_completion_request(gen_server, gen_req)
                elif isinstance(gen_req, ChatCompletionRequest):
                    gen_response = await self.send_chat_request(gen_server, gen_req)
                return gen_response
            finally:
                # TODO: support gen_router
                #await self.gen_router.finish_request(gen_req)
                pass
        else:
            # Return a streaming response that combines both context and generation responses
            return StreamingResponse(
                self.merge_streaming_responses(ctx_response, gen_server, gen_req),
                media_type="text/event-stream"
            )


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
                    logging.error(f"Unexpected error in stream: {e}")
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
        if request.stream:
            response_generator = create_generator(url, request)
            return StreamingResponse(content=response_generator, media_type="text/event-stream")
        else:
            request_json = request.model_dump(exclude_unset=True)
            async with self.session.post(url + endpoint, json=request_json) as response:
                content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" in content_type:
                    raise ValueError("Received an event-stream although request stream was False")

                response_dict = await response.json()
                if not response.ok:
                    logging.error(f"Request failed with status {response.status}")
                    logging.error(f"Response body: {response_dict}")
                    response.raise_for_status()
                return response_type(**response_dict)

    async def send_completion_request(self, url: str, request: CompletionRequest) -> Union[CompletionResponse, StreamingResponse]:
        return await self.send_request(url, request, "/v1/completions", CompletionResponse, self.create_completion_generator)

    async def send_chat_request(self, url: str, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return await self.send_request(url, request, "/v1/chat/completions", ChatCompletionResponse, self.create_chat_generator)

    async def check_server_ready(self, server_url: str) -> bool:
        try:
            async with self.session.get(server_url+"/health") as response:
                return response.status == 200
        except Exception:
            return False

    async def wait_for_servers_ready(self, server_start_timeout_secs: int = 180):
        async def are_servers_ready():
            context_ready = True
            if self.ctx_servers is not None:
                context_ready = all([await self.check_server_ready(url) for url in self.ctx_servers])
            generation_ready = all([await self.check_server_ready(url) for url in self.gen_servers])
            return context_ready and generation_ready

        async def check_all_servers_ready():
            while not await are_servers_ready():
                wait_time = 3
                logging.info("Context and generation servers are not ready. Waiting...")
                await asyncio.sleep(wait_time)

        try:
            await asyncio.wait_for(check_all_servers_ready(), timeout=server_start_timeout_secs)
        except asyncio.CancelledError:
            raise TimeoutError("Timeout waiting for context and generation servers to be ready")
