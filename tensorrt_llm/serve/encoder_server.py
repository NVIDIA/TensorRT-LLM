#!/usr/bin/env python
import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
import uvicorn

from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.version import __version__ as VERSION
from tensorrt_llm._torch.multimodal.mm_encoder import MultimodalEncoder
from tensorrt_llm.executor.multimodal import MultimodalRequest
from tensorrt_llm.multimodal_params import MultimodalParams
from pathlib import Path
from tensorrt_llm.serve.openai_protocol import ModelList, ModelCard
from tensorrt_llm.serve.openai_server import OpenAIServer
from dataclasses import asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# yapf: enale
TIMEOUT_KEEP_ALIVE = 5  # seconds.


class OpenAIEncoderServer:
    """
    Encoder server that processes image URLs and returns structured embeddings
    Compatible with the OpenAI disaggregated server architecture.
    """

    def __init__(self, encoder: MultimodalEncoder, model: str):
        """
        Initialize the encoder server.

        Args:
            encoder: ImageEncoder instance specialized for encoding images
            model: Name or identifier for the encoder model
        """
        self.encoder = encoder
        self.model = model

        model_dir = Path(model)
        if model_dir.exists() and model_dir.is_dir():
            self.model = model_dir.name
        else:
            self.model = model

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # terminate rank0 worker
            yield
            self.encoder.shutdown()

        self.app = FastAPI(lifespan=lifespan)

        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(_, exc):
            return OpenAIServer.create_error_response(message=str(exc))

        self.register_routes()

    async def await_disconnected(self, raw_request: Request, promise):
        while not await raw_request.is_disconnected():
            await asyncio.sleep(1)
        if not promise.finished:
            promise.abort()
            logger.info(
                f"{raw_request.client} is disconnected, abort {promise.request_id}")

    def register_routes(self):
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/version", self.version, methods=["GET"])
        self.app.add_api_route("/v1/models", self.get_model, methods=["GET"])
        self.app.add_api_route("/v1/multimodal_encoder",
                               self.encode_image,
                               methods=["POST"])

    async def health(self) -> Response:
        """Health check endpoint."""
        return Response(status_code=200)

    async def version(self) -> JSONResponse:
        """Version information endpoint."""
        ver = {"version": VERSION}
        return JSONResponse(content=ver)

    async def get_model(self) -> JSONResponse:
        model_list = ModelList(data=[ModelCard(id=self.model)])
        return JSONResponse(content=model_list.model_dump())

    async def encode_image(self, request: ChatCompletionRequest, raw_request: Request) -> Response:

        async def create_mm_embedding_response(
                promise) -> MultimodalParams:
            await promise.aresult()
            return promise.multimodal_params

        try:
            mm_request = MultimodalRequest.from_chat_messages(request.messages)
            if len(mm_request.items) == 0:
                return JSONResponse(content={})
            promise = await self.encoder.generate_async(mm_request)
            asyncio.create_task(self.await_disconnected(raw_request, promise))
            response = await create_mm_embedding_response(promise)
            return JSONResponse(content=asdict(response))

        except CppExecutorError:
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)
        except Exception as e:
            return OpenAIServer.create_error_response(str(e))

    async def __call__(self, host: str, port: int):
        """Run the server."""
        config = uvicorn.Config(self.app,
                               host=host,
                               port=port,
                               log_level="info",
                               timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
        await uvicorn.Server(config).serve()


