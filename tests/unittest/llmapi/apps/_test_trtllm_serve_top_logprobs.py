import openai
import pytest
import yaml

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture(scope="module", ids=["TinyLlama-1.1B-Chat"])
def model_name():
    return "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module", params=["trt", "pytorch"])
def backend(request):
    return request.param


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file(tmp_path_factory):
    extra_llm_api_options_dict = {
        "enable_chunked_prefill": False,
        "gather_generation_logits": True,
        "kv_cache_config": {
            "enable_block_reuse": False,
        }
    }

    temp_file_path = tmp_path_factory.mktemp(
        "config") / "extra_llm_api_options.yaml"
    with open(temp_file_path, 'w') as f:
        yaml.dump(extra_llm_api_options_dict, f)
    return temp_file_path


@pytest.fixture(scope="module")
def server(model_name: str, backend: str, temp_extra_llm_api_options_file: str):
    model_path = get_model_path(model_name)
    args = ["--backend", f"{backend}"]
    if backend == "trt":
        args += ["--extra_llm_api_options", temp_extra_llm_api_options_file]
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer):
    return server.get_async_client()


@pytest.mark.asyncio(loop_scope="module")
async def test_chat_completion_top5_logprobs(async_client: openai.AsyncOpenAI,
                                             model_name: str):
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "What is the capital of France?"
    }]

    # Test top_logprobs
    chat_completion = await async_client.chat.completions.create(
        model=model_name,
        messages=messages,  # type: ignore[arg-type]
        max_completion_tokens=10,
        temperature=0.0,
        logprobs=True,
        top_logprobs=5,
        extra_body={
            "ignore_eos": True,
        })
    logprobs = chat_completion.choices[0].logprobs
    assert logprobs is not None and logprobs.content is not None
    assert len(logprobs.content) == 10
    for logprob_content in logprobs.content:
        assert logprob_content.token is not None
        assert logprob_content.logprob is not None
        assert logprob_content.bytes is not None
        assert logprob_content.top_logprobs is not None
        assert len(logprob_content.top_logprobs) == 5


@pytest.mark.asyncio(loop_scope="module")
async def test_completion_top5_logprobs(async_client: openai.AsyncOpenAI,
                                        model_name: str):
    prompt = "Hello, my name is"

    completion = await async_client.completions.create(model=model_name,
                                                       prompt=prompt,
                                                       max_tokens=5,
                                                       temperature=0.0,
                                                       logprobs=5,
                                                       extra_body={
                                                           "ignore_eos": True,
                                                       })

    choice = completion.choices[0]
    logprobs = choice.logprobs
    assert logprobs is not None
    assert logprobs.tokens is not None
    assert logprobs.token_logprobs is not None
    assert logprobs.top_logprobs is not None

    assert len(logprobs.tokens) == len(logprobs.token_logprobs) == len(
        logprobs.top_logprobs)
    assert len(logprobs.tokens) > 0

    for token, token_logprob, token_top_logprobs in zip(logprobs.tokens,
                                                        logprobs.token_logprobs,
                                                        logprobs.top_logprobs):
        assert token is not None
        assert token_logprob is not None
        assert token_logprob <= 0
        assert token_top_logprobs is not None
        assert len(token_top_logprobs) == 5
