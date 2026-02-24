import io
import os
import sys
import tempfile
from base64 import b64encode
from pathlib import Path
from typing import List

import openai
import pytest
import torch
import yaml
from PIL import Image

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.inputs import encode_base64_image

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)

from utils.llm_data import llm_models_root

from ._test_openai_mmencoder import RemoteMMEncoderServer
from ._test_openai_mmencoder import server as mm_encoder_server
from ._test_openai_mmencoder import \
    test_multimodal_content_mm_encoder as _test_multimodal_content_mm_encoder

assert mm_encoder_server is not None  # keep 'mm_encoder_server' fixture visible in this module


@pytest.fixture(scope="module", ids=["Qwen2.5-VL-3B-Instruct"])
def model_name():
    return "Qwen2.5-VL-3B-Instruct"


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file(request):
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, "extra_llm_api_options.yaml")
    try:
        extra_llm_api_options_dict = {
            "kv_cache_config": {
                "enable_block_reuse": False,
                "free_gpu_memory_fraction": 0.6,
            },
        }

        with open(temp_file_path, 'w') as f:
            yaml.dump(extra_llm_api_options_dict, f)

        yield temp_file_path
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@pytest.fixture(scope="module")
def server(model_name: str, temp_extra_llm_api_options_file: str):
    model_path = get_model_path(model_name)
    args = [
        "--extra_llm_api_options",
        temp_extra_llm_api_options_file,
        "--max_batch_size",
        "64",
        "--max_num_tokens",
        "16384",
    ]
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer):
    return server.get_async_client()


@pytest.mark.asyncio(loop_scope="module")
def test_single_chat_session_image(client: openai.OpenAI, model_name: str):
    content_text = "Describe the natural environment in the image."
    image_url = str(llm_models_root() / "multimodals" / "test_data" /
                    "seashore.png")
    messages = [{
        "role":
        "user",
        "content": [{
            "type": "text",
            "text": content_text
        }, {
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }],
    }]

    max_completion_tokens = 10
    # test single completion
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        temperature=0.0,
        logprobs=False)
    assert chat_completion.id is not None
    assert len(chat_completion.choices) == 1
    message = chat_completion.choices[0].message
    assert message.content is not None
    assert message.role == "assistant"
    # test finish_reason
    finish_reason = chat_completion.choices[0].finish_reason
    completion_tokens = chat_completion.usage.completion_tokens
    if finish_reason == "length":
        assert completion_tokens == 10
    elif finish_reason == "stop":
        assert completion_tokens <= 10
    else:
        raise RuntimeError(
            f"finish_reason {finish_reason} not in [length, stop]")
    # test max_tokens
    legacy = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
        temperature=0.0,
        logprobs=False,
    )
    assert legacy.choices[0].message.content \
        == chat_completion.choices[0].message.content


# used by mm_encoder_server
@pytest.fixture(scope="module")
def extra_encoder_options() -> bool:
    return False


# used by mm_encoder_server
@pytest.fixture(scope="module")
def temp_extra_encoder_options_file() -> str:
    return "/dummy/path"


@pytest.fixture(scope="module")
def server_patched(model_name: str, temp_extra_llm_api_options_file: str):
    # Custom module implements missing 'attach_multimodal_embeddings' to intercept
    # embeddings.
    model_path = get_model_path(model_name)
    args = [
        "--extra_llm_api_options",
        temp_extra_llm_api_options_file,
        "--max_batch_size",
        "64",
        "--max_num_tokens",
        "16384",
        "--custom_module_dirs",
        str(
            Path(sys.modules[test_single_chat_session_image_embeds.__module__].
                 __file__).parent / "_attach_multimodal_embeddings_patch"),
    ]
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.mark.needs_l40s
@pytest.mark.asyncio(loop_scope="module")
def test_single_chat_session_image_embeds(
    server_patched: RemoteOpenAIServer,
    model_name: str,
    mm_encoder_server: RemoteMMEncoderServer,
):
    client = server_patched.get_client()
    messages, mm_embed_handle = _test_multimodal_content_mm_encoder(
        mm_encoder_server.get_client(), model_name)

    max_completion_tokens = 10

    chat_completion_image = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        temperature=0.0,
        logprobs=False)

    mm_embed = SharedTensorContainer.from_dict(mm_embed_handle).get_local_view()
    with io.BytesIO() as buf:
        torch.save(mm_embed, buf)
        mm_embed_bytes = buf.getvalue()

    image_content = messages[0]["content"][1]
    assert image_content["type"] == "image_url"
    image_content.clear()
    image_content["type"] = "image_embeds"
    image_content["image_embeds"] = {
        "data": b64encode(mm_embed_bytes).decode("ascii")
    }

    # test single completion
    #
    # FIXME: Remove try-except and use 'server' instead of 'server_patched',
    #        once Qwen2VLInputProcessorBase implements attach_multimodal_embeddings.
    try:
        chat_completion_embeds = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            temperature=0.0,
            logprobs=False)

        assert chat_completion_embeds.choices[
            0].message == chat_completion_image.choices[0].message
    except openai.BadRequestError as e:
        assert isinstance(e.body, dict)
        with open(Path(e.body["message"]), "rb") as f:
            intercepted_embeddings = torch.load(f, weights_only=True)
        assert list(intercepted_embeddings.keys()) == ["image"]
        assert len(intercepted_embeddings["image"]) == 1
        torch.testing.assert_close(intercepted_embeddings["image"][0],
                                   mm_embed.cpu())
        pytest.xfail(
            reason="Model does not implement 'attach_multimodal_embeddings'")


@pytest.mark.asyncio(loop_scope="module")
def test_single_chat_session_multi_image(client: openai.OpenAI,
                                         model_name: str):
    content_text = "Tell me the difference between two images"
    image_url1 = str(llm_models_root() / "multimodals" / "test_data" /
                     "inpaint.png")
    image_url2 = str(llm_models_root() / "multimodals" / "test_data" /
                     "seashore.png")
    messages = [{
        "role":
        "user",
        "content": [{
            "type": "text",
            "text": content_text
        }, {
            "type": "image_url",
            "image_url": {
                "url": image_url1
            }
        }, {
            "type": "image_url",
            "image_url": {
                "url": image_url2
            }
        }],
    }]

    max_completion_tokens = 10
    # test single completion
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        temperature=0.0,
        logprobs=False)
    assert chat_completion.id is not None
    assert len(chat_completion.choices) == 1
    message = chat_completion.choices[0].message
    assert message.content is not None
    assert message.role == "assistant"
    # test finish_reason
    finish_reason = chat_completion.choices[0].finish_reason
    completion_tokens = chat_completion.usage.completion_tokens
    if finish_reason == "length":
        assert completion_tokens == 10
    elif finish_reason == "stop":
        assert completion_tokens <= 10
    else:
        raise RuntimeError(
            f"finish_reason {finish_reason} not in [length, stop]")
    # test max_tokens
    legacy = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
        temperature=0.0,
        logprobs=False,
    )
    assert legacy.choices[0].message.content \
        == chat_completion.choices[0].message.content


@pytest.mark.asyncio(loop_scope="module")
def test_single_chat_session_video(client: openai.OpenAI, model_name: str):
    content_text = "Tell me what you see in the video briefly."
    video_url = str(llm_models_root() / "multimodals" / "test_data" /
                    "OAI-sora-tokyo-walk.mp4")
    messages = [{
        "role":
        "user",
        "content": [{
            "type": "text",
            "text": content_text
        }, {
            "type": "video_url",
            "video_url": {
                "url": video_url
            }
        }],
    }]

    max_completion_tokens = 10
    # test single completion
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        temperature=0.0,
        logprobs=False)
    assert chat_completion.id is not None
    assert len(chat_completion.choices) == 1
    message = chat_completion.choices[0].message
    assert message.content is not None
    assert message.role == "assistant"
    # test finish_reason
    finish_reason = chat_completion.choices[0].finish_reason
    completion_tokens = chat_completion.usage.completion_tokens
    if finish_reason == "length":
        assert completion_tokens == 10
    elif finish_reason == "stop":
        assert completion_tokens <= 10
    else:
        raise RuntimeError(
            f"finish_reason {finish_reason} not in [length, stop]")
    # test max_tokens
    legacy = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
        temperature=0.0,
        logprobs=False,
    )
    assert legacy.choices[0].message.content \
        == chat_completion.choices[0].message.content


@pytest.mark.asyncio(loop_scope="module")
def test_single_chat_session_image_embed(client: openai.OpenAI,
                                         model_name: str):
    test_data_root = Path(
        os.path.join(llm_models_root(), "multimodals", "test_data"))
    content_text = "Describe the natural environment in the image."
    image_url = str(llm_models_root() / "multimodals" / "test_data" /
                    "seashore.png")
    image64 = encode_base64_image(Image.open(image_url))
    messages = [{
        "role":
        "user",
        "content": [{
            "type": "text",
            "text": content_text
        }, {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64," + image64
            }
        }],
    }]

    max_completion_tokens = 10
    # test single completion
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        temperature=0.0,
        logprobs=False)
    assert chat_completion.id is not None
    assert len(chat_completion.choices) == 1
    message = chat_completion.choices[0].message
    assert message.content is not None
    assert message.role == "assistant"
    # test finish_reason
    finish_reason = chat_completion.choices[0].finish_reason
    completion_tokens = chat_completion.usage.completion_tokens
    if finish_reason == "length":
        assert completion_tokens == 10
    elif finish_reason == "stop":
        assert completion_tokens <= 10
    else:
        raise RuntimeError(
            f"finish_reason {finish_reason} not in [length, stop]")
    # test max_tokens
    legacy = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
        temperature=0.0,
        logprobs=False,
    )
    assert legacy.choices[0].message.content \
        == chat_completion.choices[0].message.content


@pytest.mark.asyncio(loop_scope="module")
async def test_chat_streaming_image(async_client: openai.AsyncOpenAI,
                                    model_name: str):
    content_text = "Describe the natural environment in the image."
    image_url = str(llm_models_root() / "multimodals" / "test_data" /
                    "seashore.png")
    messages = [{
        "role":
        "user",
        "content": [{
            "type": "text",
            "text": content_text
        }, {
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }],
    }]

    chat_completion = await async_client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
        logprobs=False,
    )
    output = chat_completion.choices[0].message.content
    _finish_reason = chat_completion.choices[0].finish_reason

    # test streaming
    stream = await async_client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
        logprobs=False,
        stream=True,
    )
    str_chunks: List[str] = []

    finish_reason_counter = 0
    finish_reason: str = None
    async for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta
        if choice.finish_reason is not None:
            finish_reason_counter += 1
            finish_reason = choice.finish_reason
        if delta.role:
            assert delta.role == "assistant"
        if delta.content:
            str_chunks.append(delta.content)
    # test finish_reason
    if delta.content == "":
        assert finish_reason == "stop"
    assert finish_reason_counter == 1
    assert finish_reason == _finish_reason
    num_tokens = len(str_chunks)
    if finish_reason == "length":
        assert num_tokens == 10
    elif finish_reason == "stop":
        assert num_tokens <= 10
    else:
        raise RuntimeError(
            f"finish_reason {finish_reason} not in [length, stop]")
    # test generated tokens
    assert "".join(str_chunks) == output
