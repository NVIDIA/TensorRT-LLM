#!/usr/bin/env python
import asyncio
import json
import logging
from typing import AsyncGenerator, Optional

import click
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from tensorrt_llm.hlapi import LLM, BuildConfig, KvCacheConfig, SamplingParams

TIMEOUT_KEEP_ALIVE = 5  # seconds.


class LlmServer:

    def __init__(self, llm: LLM, kv_cache_config: KvCacheConfig):
        self.llm = llm
        self.kv_cache_config = kv_cache_config

        self.app = FastAPI()
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

    server = LlmServer(llm=llm, kv_cache_config=kv_cache_config)

    asyncio.run(server(host, port))


if __name__ == "__main__":
    entrypoint()
