import os
import socket

import openai
import pytest

from ..test_llm import get_model_path
from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

RANK = int(os.environ.get("SLURM_PROCID", 0))
NODE_RANK = int(os.environ.get("SLURM_NODEID", 0))
NODE_LIST = os.environ.get("SLURM_NODELIST", "").split(",")

pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture(scope="module")
def model_name():
    return "llama-3.1-model/Llama-3.1-8B-Instruct"


@pytest.fixture(scope="module", params=['pytorch'], ids=["pytorch"])
def backend(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[(2, 1), (1, 2)],
    ids=lambda tp_pp_size: f'ctx_tp{tp_pp_size[0]}pp{tp_pp_size[1]}')
def ctx_tp_pp_size(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[(2, 1), (1, 2)],
    ids=lambda tp_pp_size: f'gen_tp{tp_pp_size[0]}pp{tp_pp_size[1]}')
def gen_tp_pp_size(request):
    return request.param


@pytest.fixture(scope="module")
def worker(model_name: str, ctx_tp_pp_size: tuple, gen_tp_pp_size: tuple):
    host = socket.gethostname()
    assert len(NODE_LIST) == 2
    assert host in NODE_LIST
    if NODE_RANK == 0:
        model_path = get_model_path(model_name)
        tp_size, pp_size = ctx_tp_pp_size
        args = ["--tp_size", str(tp_size), "--pp_size", str(pp_size)]
        with RemoteOpenAIServer(model_path,
                                port=8001,
                                cli_args=args,
                                host="localhost") as server:
            yield server
    elif NODE_RANK == 1:
        model_path = get_model_path(model_name)
        tp_size, pp_size = gen_tp_pp_size
        args = ["--tp_size", str(tp_size), "--pp_size", str(pp_size)]
        with RemoteOpenAIServer(model_path, port=8002, cli_args=args,
                                host=host) as server:
            yield server


@pytest.fixture(scope="module")
def disagg_server(worker: RemoteOpenAIServer):
    if NODE_RANK == 0 and RANK == 0:
        ctx_host = socket.gethostname()  # start ctx worker on NODE_RANK 0
        assert len(NODE_LIST) == 2
        assert ctx_host in NODE_LIST
        print(f"ctx_host: {ctx_host} {NODE_LIST}")
        gen_host = NODE_LIST.remove(ctx_host)[0]
        ctx_url = f"http://{ctx_host}:8001"
        gen_url = f"http://{gen_host}:8002"
        with RemoteDisaggOpenAIServer(ctx_servers=[ctx_url],
                                      gen_servers=[gen_url],
                                      port=8000) as server:
            yield server
    else:
        yield None


@pytest.fixture(scope="module")
def client(disagg_server: RemoteDisaggOpenAIServer):
    if NODE_RANK == 0:
        return disagg_server.get_client()
    else:
        return None


@pytest.fixture(scope="module")
def async_client(disagg_server: RemoteDisaggOpenAIServer):
    if NODE_RANK == 0:
        return disagg_server.get_async_client()
    else:
        return None


def test_chat(client: openai.OpenAI, model_name: str):
    if NODE_RANK == 0:
        messages = [{
            "role": "system",
            "content": "you are a helpful assistant"
        }, {
            "role": "user",
            "content": "What is the result of 1+1? Answer in one word: "
        }]
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1,
        )
        assert chat_completion.id is not None
        assert len(chat_completion.choices) == 1
        assert chat_completion.usage.completion_tokens == 1
        message = chat_completion.choices[0].message

        print(f"Output: {message.content}")
        assert message.content == 'Two'
    else:
        time.sleep(30)
        assert True
