import asyncio

import openai
import pytest
import requests

from tensorrt_llm.version import __version__ as VERSION

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer


@pytest.fixture(scope="module")
def model_name():
    return "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module", params=[None, 'pytorch'])
def backend(request):
    return request.param


@pytest.fixture(scope="module", params=['8'])
def max_batch_size(request):
    return request.param


@pytest.fixture(scope="module", params=['80000'])
def max_seq_len(request):
    return request.param


@pytest.fixture(scope="module")
def server(model_name: str, backend: str, max_batch_size: str,
           max_seq_len: str):
    model_path = get_model_path(model_name)
    args = ["--max_beam_width", "4"]
    if backend is not None:
        args.append("--backend")
        args.append(backend)
    if max_batch_size is not None:
        args.append("--max_batch_size")
        args.append(max_batch_size)
    if max_seq_len is not None:
        args.append("--max_seq_len")
        args.append(max_seq_len)
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


def test_version(server: RemoteOpenAIServer):
    version_url = server.url_for("version")
    response = requests.get(version_url)
    assert response.status_code == 200
    assert response.json()["version"] == VERSION


def test_health(server: RemoteOpenAIServer):
    health_url = server.url_for("health")
    response = requests.get(health_url)
    assert response.status_code == 200


def test_model(client: openai.OpenAI, model_name: str):
    model = client.models.list().data[0]
    assert model.id == model_name.split('/')[-1]


# reference: https://github.com/vllm-project/vllm/blob/44f990515b124272f87954fc763d90697d8aa1db/tests/entrypoints/openai/test_basic.py#L123
@pytest.mark.asyncio
async def test_request_cancellation(server: RemoteOpenAIServer,
                                    model_name: str):
    # clunky test: send an ungodly amount of load in with short timeouts
    # then ensure that it still responds quickly afterwards

    chat_input = [{"role": "user", "content": "Write a long story"}]
    client = server.get_async_client(timeout=0.5, max_retries=3)
    tasks = []
    # Request about 2 million tokens
    for _ in range(200):
        task = asyncio.create_task(
            client.chat.completions.create(messages=chat_input,
                                           model=model_name,
                                           max_tokens=10000,
                                           extra_body={"min_tokens": 10000}))
        tasks.append(task)

    done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

    # Make sure all requests were sent to the server and timed out
    # (We don't want to hide other errors like 400s that would invalidate this
    # test)
    assert len(pending) == 0
    for d in done:
        with pytest.raises(openai.APITimeoutError):
            d.result()

    # If the server had not cancelled all the other requests, then it would not
    # be able to respond to this one within the timeout
    client = server.get_async_client(timeout=5)
    response = await client.chat.completions.create(messages=chat_input,
                                                    model=model_name,
                                                    max_tokens=10)

    assert len(response.choices) == 1
