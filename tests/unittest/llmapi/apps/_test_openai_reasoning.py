from typing import List, Tuple

import openai
import pytest

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)


# yapf: disable
@pytest.fixture(scope="module",
                params=["DeepSeek-R1-Distill-Qwen-1.5B",
                        "Qwen3/Qwen3-0.6B"])
def model_name(request) -> str:
    return request.param
# yapf: enable


@pytest.fixture(scope="module", params=["trt", "pytorch"])
def backend(request):
    return request.param


@pytest.fixture(scope="module")
def server(model_name: str, backend: str):
    # Skip specific model/backend combinations
    if model_name == "Qwen3/Qwen3-0.6B" and backend == "trt":
        pytest.skip("Qwen3 model not supported with trt backend")

    model_path = get_model_path(model_name)
    args = ["--backend", f"{backend}"]
    max_beam_width = 1 if backend == "pytorch" else 2
    args.extend(["--max_beam_width", str(max_beam_width)])
    args.extend(["--max_batch_size", "2", "--max_seq_len", "1024"])
    if model_name.startswith("Qwen3"):
        args.extend(["--reasoning_parser", "qwen3"])
    else:
        args.extend(["--reasoning_parser", "deepseek-r1"])
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer) -> openai.OpenAI:
    return server.get_client()


def test_reasoning_parser(client: openai.OpenAI, model_name: str, backend: str):
    messages = [{"role": "user", "content": "hi"}]
    if backend == "pytorch":
        n, extra_body = 1, None
    else:
        n, extra_body = 2, dict(use_beam_search=True)
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=1000,
        temperature=0.0,
        n=n,
        extra_body=extra_body,
    )

    assert len(resp.choices) == n
    for resp_choice in resp.choices:
        assert len(resp_choice.message.content) > 0
        assert len(resp_choice.message.reasoning_content) > 0


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer) -> openai.AsyncOpenAI:
    return server.get_async_client()


async def process_stream(
        stream: openai.AsyncStream) -> Tuple[List[str], List[str]]:
    content_chunks: List[str] = []
    reasoning_content_chunks: List[str] = []
    async for chunk in stream:
        assert len(chunk.choices) == 1
        choice = chunk.choices[0]
        delta = choice.delta.dict()
        content = delta.get("content", None)
        reasoning_content = delta.get("reasoning_content", None)
        if content:
            content_chunks.append(content)
        if reasoning_content:
            reasoning_content_chunks.append(reasoning_content)
    return (content_chunks, reasoning_content_chunks)


@pytest.mark.asyncio(loop_scope="module")
async def test_reasoning_parser_streaming(async_client: openai.AsyncOpenAI,
                                          model_name: str):
    messages = [{"role": "user", "content": "hi"}]
    stream = await async_client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=1000,
        temperature=0.0,
        stream=True,
    )

    content_chunks, reasoning_content_chunks = await process_stream(
        stream=stream)
    assert len(content_chunks) > 0
    assert len(reasoning_content_chunks) > 0

    stream = await async_client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=2,
        temperature=0.0,
        stream=True,
    )

    content_chunks, reasoning_content_chunks = await process_stream(
        stream=stream)
    assert len(content_chunks) == 0
    if model_name.startswith("Qwen3"):
        # First token would be <think>
        assert len(reasoning_content_chunks) == 1
    else:
        # <think> is in chat template
        assert len(reasoning_content_chunks) == 2
