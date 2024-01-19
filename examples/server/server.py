import argparse
import asyncio
import json
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from tensorrt_llm.engine import AsyncLLMEngine

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
async_engine: AsyncLLMEngine = None


@app.get("/stats")
async def stats() -> Response:
    """Health check."""
    if async_engine.stats.async_q.empty():
        result = {}
    else:
        result = json.loads(await async_engine.stats.async_q.get())

    return JSONResponse(result)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()

    streaming = request_dict.pop("streaming", False)
    generator = async_engine.generate(request_dict.pop("prompt"),
                                      request_dict.pop("max_num_tokens", 8),
                                      streaming)

    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for output in generator:
            yield (json.dumps(output.text) + "\0").encode("utf-8")

    if streaming:
        return StreamingResponse(stream_results())

    # Non-streaming case
    return JSONResponse({"text": await anext(generator)})


async def main(args):
    global async_engine

    async_engine = AsyncLLMEngine(args.model_dir, args.tokenizer_type,
                                  args.max_beam_width, args.max_num_sequences)
    config = uvicorn.Config(app,
                            host=args.host,
                            port=args.port,
                            log_level="info",
                            timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
    await uvicorn.Server(config).serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("tokenizer_type")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_beam_width", type=int, default=1)
    parser.add_argument("--max_num_sequences", type=int, default=10)
    args = parser.parse_args()

    asyncio.run(main(args))
