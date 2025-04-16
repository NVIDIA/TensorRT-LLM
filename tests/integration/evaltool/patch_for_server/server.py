import argparse
import asyncio

import httpx
import uvicorn
from fastapi import Body, FastAPI
from openai_schema import (ChatCompletionRequest, ChatCompletionResponse,
                           CompletionRequest, CompletionResponse)
from trt_llm_model import TRTLLMModel

import tensorrt_llm

runtime_rank = tensorrt_llm.mpi_rank()
world_size = tensorrt_llm.mpi_world_size()

# API Endpoints
app = FastAPI(
    title="TRT-LLM Inference Server",
    version="0.0.1",
)


@app.post("/v1/chat/completions/forward")
async def create_chat_completion(req: ChatCompletionRequest = Body(...), ):

    # TODO: Update the following logic to align `llm_name` with the `mtbench` convention:
    #  \w+\/\w+ (e.g., meta-llama/Llama-2-7b).
    #
    # Currently, the `trtllm` server accepts `llm_name` in a format not supported by `mtbench`.
    # We can replace `/` with `-` to match the Hugging Face model naming conventions.
    #
    # if req.model != model.engine_path:
    #     raise HTTPException(
    #         status_code=400,
    #         detail=f"Invalid model name specified. Please use the correct model: `{model.engine_path}`.",
    #     )

    context = ""
    for m in req.messages:
        if m.role == "system":
            context += m.content + "\n\n"
        elif m.role == "user":
            context += "User:\n" + m.content + "\n\n"
        elif m.role == "assistant":
            context += "Assistant:\n" + m.content + "\n\n"
        else:
            raise NotImplementedError("Unsupported role: " + str(m.role))

    kwargs = {"context": [context]}
    if req.max_tokens is not None:
        kwargs.update({"max_output_len": req.max_tokens})
    if req.temperature is not None:
        kwargs.update({"temperature": req.temperature})
    if req.top_p is not None:
        kwargs.update({"top_p": req.top_p})
    if req.stop is not None:
        kwargs.update({"stop": req.stop})

    prediction = model.generate_text(**kwargs)

    choice = {
        "index": 0,
        "message": {
            "role": "assistant",
            "content": prediction["predictions"][0],
        },
    }

    if req.logprobs:
        choice.update({
            "logprobs": {
                "content": [{
                    "token": token,
                    "logprob": logprob
                } for token, logprob in zip(prediction["tokens"][0],
                                            prediction["logprobs"][0])]
            },
        })

    response = {"model": req.model, "choices": [choice]}

    return response


@app.post("/v1/completions/forward")
async def create_completion(req: CompletionRequest = Body(...), ):

    # TODO: Update the following logic to align `llm_name` with the `mtbench` convention:
    #  \w+\/\w+ (e.g., meta-llama/Llama-2-7b).
    #
    # Currently, the `trtllm` server accepts `llm_name` in a format not supported by `mtbench`.
    # We can replace `/` with `-` to match the Hugging Face model naming conventions.
    #
    # if req.model != model.engine_path:
    #     raise HTTPException(
    #         status_code=400,
    #         detail=f"Invalid model name specified. Please use the correct model: `{model.engine_path}`.",
    #     )

    kwargs = {"context": [req.prompt]}
    if req.max_tokens is not None:
        kwargs.update({"max_output_len": req.max_tokens})
    if req.temperature is not None:
        kwargs.update({"temperature": req.temperature})
    if req.top_p is not None:
        kwargs.update({"top_p": req.top_p})
    if req.echo is not None:
        kwargs.update({"echo": req.echo})
    if req.stop is not None:
        kwargs.update({"stop": req.stop})

    prediction = model.generate_text(**kwargs)

    choice = {
        "index": 0,
        "text": prediction["predictions"][0],
    }

    if req.logprobs:
        assert ("logprobs" in prediction.keys()
                ), "TRT LLM engine hasn't returned context_logits"
        choice.update({
            "logprobs": {
                "token_logprobs": prediction["logprobs"][0],
                "tokens": prediction["tokens"][0],
            },
        })

    response = {"model": req.model, "choices": [choice]}

    return response


def run_server(port):
    headers = {
        "Content-Type": "application/json",
    }
    app2 = FastAPI(
        title="TRT-LLM Inference Server",
        version="0.0.1",
    )

    @app2.post("/v1/chat/completions")
    async def create_chat_completion(
            req: ChatCompletionRequest = Body(...), ) -> ChatCompletionResponse:
        async with httpx.AsyncClient(timeout=httpx.Timeout(36000.0)) as client:
            tasks = []
            for i in range(0, world_size):
                _port = 12479 + i
                task = client.post(
                    f"http://0.0.0.0:{_port}/v1/chat/completions/forward",
                    headers=headers,
                    json=req.dict())
                tasks.append(task)

            try:
                responses = await asyncio.gather(*tasks)
            except httpx.RequestError as e:
                print(f"Network error: {e}")
                return {"error": "Network error occurred during request"}

            return ChatCompletionResponse(**responses[0].json())

    @app2.post("/v1/completions")
    async def create_completion(
            req: CompletionRequest = Body(...), ) -> CompletionResponse:
        async with httpx.AsyncClient(timeout=httpx.Timeout(36000.0)) as client:
            tasks = []
            for i in range(0, world_size):
                _port = 12479 + i
                task = client.post(
                    f"http://0.0.0.0:{_port}/v1/completions/forward",
                    headers=headers,
                    json=req.dict())
                tasks.append(task)

            try:
                responses = await asyncio.gather(*tasks)
            except httpx.RequestError as e:
                print(f"Network error: {e}")
                return {"error": "Network error occurred during request"}

            return CompletionResponse(**responses[0].json())

    uvicorn.run(app2, host="0.0.0.0", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--max_output_len", type=int, default=256)
    parser.add_argument("--lookahead_config", type=str, default=None)
    args = parser.parse_args()
    print(args)

    model = TRTLLMModel(engine_path=args.engine_path,
                        tokenizer_path=args.tokenizer_path,
                        max_output_len=args.max_output_len,
                        lookahead_config=args.lookahead_config)

    if runtime_rank == 0:
        import multiprocessing
        p = multiprocessing.Process(target=run_server, args=(12478, ))
        p.start()

    uvicorn.run(app, host="0.0.0.0", port=12479 + runtime_rank)

    p.join()
