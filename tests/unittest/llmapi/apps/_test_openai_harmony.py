import openai
import pytest

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer


@pytest.fixture(scope="module", ids=["gpt-oss-20b"])
def model_name():
    return "gpt_oss/gpt-oss-20b"


@pytest.fixture(scope="module")
def server(model_name: str):
    model_path = get_model_path(model_name)
    with RemoteOpenAIServer(model_path) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer):
    return server.get_async_client()


@pytest.mark.asyncio(loop_scope="module")
async def test_chat_completion_gpt_oss(async_client: openai.AsyncOpenAI):
    response = await async_client.chat.completions.create(
        model="gpt_oss/gpt-oss-20b",
        messages=[{
            "role":
            "system",
            "content":
            "You are ChatGPT, a large language model trained by OpenAI"
        }, {
            "role": "user",
            "content": "Which one is larger, 9.11 or 9.9?"
        }])
    assert response.choices[0].message.content is not None
    assert response.choices[0].message.reasoning_content is not None
