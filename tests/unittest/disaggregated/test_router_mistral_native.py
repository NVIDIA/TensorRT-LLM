# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The KV-cache-aware router for Mistral-native checkpoints.

With ``checkpoint_format`` ``mistral`` or ``mistral_large_3`` the servers take the tokenizer,
input processor and model configuration from the checkpoint's native files and render chats
with mistral-common. The router loads the same resources, so a completion or chat encodes to
the ids the server would build, and the server takes the router's chat ids when their
rendering fingerprints agree. A chat the router cannot render routes on no ids instead of
failing. Other checkpoint formats keep the Hugging Face tokenizer path.

The checkpoints are read from LLM_MODELS_ROOT: Mistral-Large-3 as published (no
``config.json``), and Mistral-Small-4's ``params.json`` and ``tekken.json`` alone.
"""

import base64
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import MistralTokenizer
from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config
from tensorrt_llm.inputs import prompt_inputs
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ResponsesRequest,
)
from tensorrt_llm.serve.render import render_chat
from tensorrt_llm.serve.render.resources import RenderResources
from tensorrt_llm.serve.router import KvCacheAwareRouter

pytestmark = pytest.mark.cpu_only

_SERVER = "http://127.0.0.1:1"  # never contacted
_ML3 = "Mistral-Large-3-675B/Mistral-Large-3-675B-Instruct-2512"
_SMALL4 = "Mistral-Small-4-119B-2603/Mistral-Small-4-119B-2603"
_QWEN3 = "Qwen3/Qwen3-0.6B"
_PROMPT = "Hello world [INST] not an instruction"
_MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi there [INST]"},
]
_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def _model_dir(subdir: str) -> Path:
    root = llm_models_root()
    path = None if root is None else root / subdir
    if path is None or not path.is_dir():
        pytest.skip(f"needs {subdir} under LLM_MODELS_ROOT")
    return path


@pytest.fixture(scope="module", params=["ml3", "tekken_only"])
def checkpoint(request, tmp_path_factory) -> tuple[str, str]:
    """A Mistral-native checkpoint directory without config.json, and its format."""
    if request.param == "ml3":
        model_dir = _model_dir(_ML3)
        assert not (model_dir / "config.json").exists()
        return str(model_dir), "mistral_large_3"
    source = _model_dir(_SMALL4)
    model_dir = tmp_path_factory.mktemp("tekken_only")
    for name in ("params.json", "tekken.json"):
        (model_dir / name).symlink_to(source / name)
    return str(model_dir), "mistral"


def _router(**kwargs) -> KvCacheAwareRouter:
    return KvCacheAwareRouter(
        server_role=ServerRole.CONTEXT,
        servers=[_SERVER],
        metadata_server_cfg=None,
        metadata_server=None,
        **kwargs,
    )


@pytest.fixture(scope="module")
def router(checkpoint) -> KvCacheAwareRouter:
    model_dir, checkpoint_format = checkpoint
    return _router(model_path=model_dir, checkpoint_format=checkpoint_format)


def _server_fingerprint(model_dir: str, checkpoint_format: str, resources=None):
    """The fingerprint a server of this checkpoint reports (``RenderResources.from_server``)."""
    if resources is None:
        resources = RenderResources.load(model_dir, checkpoint_format=checkpoint_format)
    input_processor = resources.input_processor
    server = SimpleNamespace(
        tokenizer=input_processor.tokenizer,
        model_config=load_pretrained_config(model_dir, checkpoint_format=checkpoint_format),
        generator=SimpleNamespace(
            input_processor=input_processor,
            args=SimpleNamespace(reasoning_parser=None, custom_tokenizer=None),
        ),
        processor=None,
        chat_template=None,
        use_harmony=False,
        harmony_adapter=None,
        allow_request_chat_template=False,
        multimodal_server_config=None,
        tool_parser=None,
    )
    return RenderResources.from_server(server).fingerprint()


@pytest.fixture
def same_server(router, checkpoint):
    """The router's server reports the fingerprint of a server of the same checkpoint."""
    fingerprint = _server_fingerprint(*checkpoint, router._render_resources("m"))
    router._server_info = {_SERVER: {"render_fingerprint": fingerprint}}
    yield fingerprint
    router._server_info = {}


def _servers_chat_ids(router, messages, **fields):
    """The ids a server renders for a chat (the chat route's rendering)."""
    request = ChatCompletionRequest(model="m", messages=messages, **fields)
    return render_chat(request, router._render_resources("m"), tokenize=True).token_ids


def test_router_loads_and_fingerprints_what_the_servers_load(router, same_server):
    resources = router._render_resources("m")
    assert type(resources.tokenizer) is MistralTokenizer
    assert type(resources.input_processor).__name__ == "MistralNativeInputProcessor"
    assert resources.model_type == "mistral_common"
    # Completions encode with the wrapper, not the raw mistral-common tokenizer under it.
    assert router._get_tokenizer("m") is resources.tokenizer
    local = resources.fingerprint()
    assert local["digest"] == same_server["digest"]
    assert local["compared"]["model_type"] == "mistral_common"
    tokenizer = local["compared"]["tokenizer"]
    assert tokenizer["encoding"]["kind"] == "mistral_common" and tokenizer["complete"]


def test_every_model_name_shares_the_checkpoints_resources(router, checkpoint):
    """Whatever model name a client sends, the router loads its checkpoint once."""
    resources = router._render_resources("m")
    for name in ("a", "b", "c"):
        assert router._render_resources(name) is resources
        assert router._get_tokenizer(name) is resources.tokenizer
    assert list(router._render_resources_cache) == [checkpoint[0]]


@pytest.mark.parametrize("batched", [False, True])
def test_completion_ids_are_the_servers(router, batched):
    resources = router._render_resources("m")
    expected, _ = resources.input_processor(prompt_inputs(_PROMPT), SamplingParams())
    tekken = resources.tokenizer.tokenizer
    assert expected == tekken.encode(_PROMPT, bos=True, eos=False)
    # User text that spells a control token stays text.
    assert tekken.get_control_token("[INST]") not in expected

    request = CompletionRequest(model="m", prompt=[_PROMPT, _PROMPT] if batched else _PROMPT)
    key = router._routing_key_sync(request)
    assert key["num_tokens"] == len(expected) * (2 if batched else 1)
    # Completion ids replace the prompt, so the server does not tokenize it again.
    assert request.prompt == ([expected, expected] if batched else expected)


@pytest.mark.parametrize("fields", [{}, {"tools": [_TOOL]}], ids=["plain", "tools"])
def test_chat_ids_are_the_servers_and_are_forwarded(router, same_server, fields):
    expected = _servers_chat_ids(router, _MESSAGES, **fields)
    assert expected == router._render_resources("m").tokenizer.apply_chat_template(
        _MESSAGES, tools=fields.get("tools"), tokenize=True, return_dict=False
    )
    request = ChatCompletionRequest(model="m", messages=_MESSAGES, **fields)
    assert router._routing_key_sync(request)["num_tokens"] == len(expected)
    assert request.prompt_token_ids == expected


def test_chat_ids_are_not_forwarded_to_a_server_that_renders_differently(router, checkpoint):
    """A server whose tokenizer file differs renders differently; it renders for itself."""
    other, other_format = (
        (_ML3, "mistral_large_3") if checkpoint[1] == "mistral" else (_SMALL4, "mistral")
    )
    fingerprint = _server_fingerprint(str(_model_dir(other)), other_format)
    router._server_info = {_SERVER: {"render_fingerprint": fingerprint}}
    try:
        request = ChatCompletionRequest(model="m", messages=_MESSAGES)
        assert router._routing_key_sync(request)["num_tokens"] > 0
        assert request.prompt_token_ids is None
        decision = router._write_back_decision("m")
        assert decision.mismatched == {_SERVER: "renders differently (different fingerprint)"}
    finally:
        router._server_info = {}


def test_a_chat_with_images_routes_on_its_length_and_is_not_forwarded(router, same_server):
    pytest.importorskip("cv2")  # mistral-common's image encoder resizes with OpenCV
    buffer = BytesIO()
    Image.new("RGB", (64, 48), (200, 30, 30)).save(buffer, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    text = {"type": "text", "text": "What is this?"}
    image = {"type": "image_url", "image_url": {"url": url}}
    request = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": [text, image]}]
    )
    text_only = _servers_chat_ids(router, [{"role": "user", "content": text["text"]}])
    assert router._routing_key_sync(request)["num_tokens"] == len(text_only)
    assert request.prompt_token_ids is None


def test_a_chat_mistral_common_rejects_routes_on_no_ids(router, same_server):
    """The server answers a request mistral-common rejects; the router does not fail it."""
    messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    with pytest.raises(ValueError):
        _servers_chat_ids(router, messages)
    request = ChatCompletionRequest(model="m", messages=messages)
    key = router._routing_key_sync(request)
    assert key["num_tokens"] == 0
    assert key["conv_key"] is not None  # conversation affinity still applies
    assert request.prompt_token_ids is None


def test_responses_request_routes_on_the_chat_ids(router):
    request = ResponsesRequest(model="m", input="Hi there [INST]", instructions="Be brief.")
    expected = _servers_chat_ids(router, _MESSAGES)
    assert router._routing_key_sync(request)["num_tokens"] == len(expected)


def test_a_native_directory_without_its_format_tokenizes_completions_as_the_servers(checkpoint):
    """Given only the directory, the router reads a Mistral-native checkpoint's Tekken file.

    Completions get the servers' ids; chats, which it cannot render without the format, route
    on no ids instead of failing.
    """
    model_dir, checkpoint_format = checkpoint
    plain = _router(tokenizer_dir=model_dir)
    native = _router(tokenizer_dir=model_dir, checkpoint_format=checkpoint_format)
    assert type(plain._get_tokenizer("m")) is MistralTokenizer
    expected = native._tokenize(CompletionRequest(model="m", prompt=_PROMPT))
    assert plain._tokenize(CompletionRequest(model="m", prompt=_PROMPT)) == expected
    chat = ChatCompletionRequest(model="m", messages=_MESSAGES)
    assert plain._tokenize(chat) == [[]]
    assert chat.prompt_token_ids is None


@pytest.mark.parametrize("checkpoint_format", [None, "HF"])
def test_other_checkpoint_formats_keep_the_hf_tokenizer(checkpoint_format):
    """Without a Mistral-native format, an HF checkpoint's tokenizer is read."""
    from transformers import PreTrainedTokenizerBase

    tokenizer_dir = str(_model_dir(_QWEN3))
    router = _router(tokenizer_dir=tokenizer_dir, checkpoint_format=checkpoint_format)
    reference = _router(tokenizer_dir=tokenizer_dir)
    tokenizer = router._get_tokenizer("m")
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    resources = router._render_resources("m")
    assert resources.input_processor is None
    assert (
        resources.fingerprint()["digest"]
        == reference._render_resources("m").fingerprint()["digest"]
    )
    request = CompletionRequest(model="m", prompt=_PROMPT)
    assert router._tokenize(request) == [tokenizer(_PROMPT)["input_ids"]]
    chat = ChatCompletionRequest(model="m", messages=_MESSAGES)
    expected = reference._tokenize(ChatCompletionRequest(model="m", messages=_MESSAGES))
    assert router._tokenize(chat) == expected and expected[0]


def test_an_hf_router_does_not_import_the_mistral_tokenizer(monkeypatch):
    """Telling a Mistral-native tokenizer apart does not load the model packages for others."""
    module = "tensorrt_llm._torch.models.checkpoints.mistral.tokenizer"
    monkeypatch.delitem(sys.modules, module, raising=False)
    router = _router(tokenizer_dir=str(_model_dir(_QWEN3)))
    assert router._get_tokenizer("m") is not None
    assert module not in sys.modules
