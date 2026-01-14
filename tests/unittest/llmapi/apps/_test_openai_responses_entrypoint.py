import openai
import pytest

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture(scope="module", params=["Qwen3/Qwen3-0.6B"])
def model(request):
    return request.param


@pytest.fixture(scope="module")
def server(model: str):
    model_path = get_model_path(model)

    args = []
    if model.startswith("Qwen3"):
        args.extend(["--reasoning_parser", "qwen3"])
    elif model.startswith("DeepSeek-R1"):
        args.extend(["--reasoning_parser", "deepseek-r1"])

    if not model.startswith("gpt_oss"):
        args.extend(["--tool_parser", "qwen3"])

    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_async_client()


@pytest.mark.asyncio(loop_scope="module")
async def test_get(client: openai.AsyncOpenAI, model: str):
    response = await client.responses.create(
        model=model, input="Which one is larger as numeric, 9.9 or 9.11?", max_output_tokens=1024
    )

    response_get = await client.responses.retrieve(response.id)
    assert response_get.id == response.id
    assert response_get.model_dump() == response.model_dump()


@pytest.mark.asyncio(loop_scope="module")
async def test_get_invalid_response_id(client: openai.AsyncOpenAI):
    with pytest.raises(openai.BadRequestError):
        await client.responses.retrieve("invalid_response_id")


@pytest.mark.asyncio(loop_scope="module")
async def test_get_non_existent_response_id(client: openai.AsyncOpenAI):
    with pytest.raises(openai.NotFoundError):
        await client.responses.retrieve("resp_non_existent_response_id")


@pytest.mark.asyncio(loop_scope="module")
async def test_delete(client: openai.AsyncOpenAI, model: str):
    response = await client.responses.create(
        model=model, input="Which one is larger as numeric, 9.9 or 9.11?", max_output_tokens=1024
    )

    await client.responses.delete(response.id)
    with pytest.raises(openai.NotFoundError):
        await client.responses.retrieve(response.id)


@pytest.mark.asyncio(loop_scope="module")
async def test_delete_invalid_response_id(client: openai.AsyncOpenAI):
    with pytest.raises(openai.BadRequestError):
        await client.responses.delete("invalid_response_id")


@pytest.mark.asyncio(loop_scope="module")
async def test_delete_non_existent_response_id(client: openai.AsyncOpenAI):
    with pytest.raises(openai.NotFoundError):
        await client.responses.delete("resp_non_existent_response_id")
