"""
NOTE: This FastAPI-based server is only an example for demonstrating the usage
of TensorRT LLM LLM API. It is not intended for production use.
For production, use the `trtllm-serve` command. The server exposes OpenAI compatible API endpoints.
"""

#!/usr/bin/env python
import asyncio
import json
import logging
import signal
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import AsyncGenerator, Optional

import click
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.executor import CppExecutorError, RequestError
from tensorrt_llm.llmapi import BuildConfig, KvCacheConfig, SamplingParams

TIMEOUT_KEEP_ALIVE = 5  # seconds.


class LlmServer:

    def __init__(self, llm: LLM):
        self.llm = llm

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # terminate rank0 worker
            yield
            self.llm.shutdown()

        self.app = FastAPI(lifespan=lifespan)
        self.register_routes()

    def register_routes(self):
        self.app.add_api_route("/stats", self.stats, methods=["GET"])
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/generate", self.generate, methods=["POST"])

    async def stats(self) -> Response:
        content = await self.llm.aget_stats()
        return JSONResponse(json.loads(content))

    async def health(self) -> Response:
        return Response(status_code=200)

    async def generate(self, request: Request) -> Response:
        ''' Generate completion for the request.

        The request should be a JSON object with the following fields:
        - prompt: the prompt to use for the generation.
        - stream: whether to stream the results or not.
        - other fields: the sampling parameters (See `SamplingParams` for details).
        '''
        request_dict = await request.json()

        prompt = request_dict.pop("prompt", "")
        streaming = request_dict.pop("streaming", False)

        sampling_params = SamplingParams(**request_dict)

        try:
            promise = self.llm.generate_async(prompt,
                                              streaming=streaming,
                                              sampling_params=sampling_params)

            async def stream_results() -> AsyncGenerator[bytes, None]:
                async for output in promise:
                    yield output.outputs[0].text_diff.encode("utf-8")

            if streaming:
                return StreamingResponse(stream_results())

            # Non-streaming case
            await promise.aresult()
            return JSONResponse({"text": promise.outputs[0].text})
        except RequestError as e:
            return JSONResponse(content=str(e),
                                status_code=HTTPStatus.BAD_REQUEST)
        except CppExecutorError:
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)

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
@click.option("--cp_size", type=int, default=1)
@click.option("--kv_cache_free_gpu_memory_fraction", type=float, default=0.8)
def entrypoint(model_dir: str,
               tokenizer: Optional[str] = None,
               host: Optional[str] = None,
               port: int = 8000,
               max_beam_width: int = 1,
               tp_size: int = 1,
               pp_size: int = 1,
               cp_size: int = 1,
               kv_cache_free_gpu_memory_fraction: float = 0.8):
    host = host or "0.0.0.0"
    port = port or 8000
    logging.info(f"Starting server at {host}:{port}")

    build_config = BuildConfig(max_batch_size=10, max_beam_width=max_beam_width)

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction)

    llm = LLM(model_dir,
              tokenizer,
              tensor_parallel_size=tp_size,
              pipeline_parallel_size=pp_size,
              build_config=build_config,
              kv_cache_config=kv_cache_config)

    server = LlmServer(llm=llm)

    asyncio.run(server(host, port))


if __name__ == "__main__":
    entrypoint()
