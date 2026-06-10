# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for the trtllm-serve post-processing hook (TRTLLM-12622).

Launches a real ``trtllm-serve`` with ``--post_processor`` pointing at one of
the sample hooks in ``_postproc_hook_samples`` and asserts the client-visible
effect (rewrite / suppress / terminate) across the chat and completions
endpoints, streaming and non-streaming, with the postproc worker pool both
disabled (in-proxy detok) and enabled (worker-process detok).

The hooks are stateless and deterministic so assertions hold regardless of the
model's (non-deterministic) output:
  - UppercaseHook: every chunk is upper-cased  -> output == output.upper()
  - SuppressHook:  every chunk is withheld     -> output == ""
  - TerminateHook: first chunk terminates       -> output == "", stops early
"""

import os

import openai
import pytest

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)

# Dotted import paths into the sample-hook module shipped alongside this test.
_HOOKS = {
    "uppercase": "_postproc_hook_samples.UppercaseHook",
    "suppress": "_postproc_hook_samples.SuppressHook",
    "terminate": "_postproc_hook_samples.TerminateHook",
}


@pytest.fixture(scope="module")
def model_name():
    return "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module", params=[0, 2], ids=["disable_processpool", "enable_processpool"])
def num_postprocess_workers(request):
    return request.param


@pytest.fixture(scope="module", params=list(_HOOKS), ids=list(_HOOKS))
def hook(request):
    return request.param


@pytest.fixture(scope="module")
def server(model_name: str, num_postprocess_workers: int, hook: str):
    model_path = get_model_path(model_name)
    args = [
        "--backend",
        "pytorch",
        # co-exist with other servers
        "--kv_cache_free_gpu_memory_fraction",
        "0.2",
        "--num_postprocess_workers",
        f"{num_postprocess_workers}",
        "--post_processor",
        _HOOKS[hook],
    ]
    # Make the sample-hook module importable by the server (and its postproc
    # worker) subprocesses.
    apps_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["PYTHONPATH"] = apps_dir + os.pathsep + env.get("PYTHONPATH", "")
    with RemoteOpenAIServer(model_path, args, env=env) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer):
    return server.get_async_client()


def _assert_text_matches_hook(hook: str, text: str):
    if hook == "uppercase":
        assert text == text.upper(), f"expected upper-cased text, got {text!r}"
    elif hook == "suppress":
        assert text == "", f"expected suppressed (empty) text, got {text!r}"
    elif hook == "terminate":
        assert text == "", f"expected terminated (empty) text, got {text!r}"
    else:
        raise AssertionError(f"unknown hook {hook}")


def test_completions_non_streaming(client: openai.OpenAI, model_name: str, hook: str):
    completion = client.completions.create(
        model=model_name,
        prompt="Hello, my name is",
        max_tokens=16,
        temperature=0.0,
    )
    text = completion.choices[0].text
    _assert_text_matches_hook(hook, text)
    if hook == "terminate":
        assert completion.choices[0].finish_reason == "stop"


@pytest.mark.asyncio(loop_scope="module")
async def test_completions_streaming(async_client: openai.AsyncOpenAI, model_name: str, hook: str):
    stream = await async_client.completions.create(
        model=model_name,
        prompt="Hello, my name is",
        max_tokens=16,
        temperature=0.0,
        stream=True,
    )
    text = ""
    async for chunk in stream:
        token = chunk.choices[0].text
        if token:
            text += token
    _assert_text_matches_hook(hook, text)


def test_chat_non_streaming(client: openai.OpenAI, model_name: str, hook: str):
    chat = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Hello, tell me a short story."}],
        max_tokens=16,
        temperature=0.0,
    )
    content = chat.choices[0].message.content or ""
    _assert_text_matches_hook(hook, content)
    if hook == "terminate":
        assert chat.choices[0].finish_reason == "stop"


@pytest.mark.asyncio(loop_scope="module")
async def test_chat_streaming(async_client: openai.AsyncOpenAI, model_name: str, hook: str):
    stream = await async_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Hello, tell me a short story."}],
        max_tokens=16,
        temperature=0.0,
        stream=True,
    )
    content = ""
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            content += delta
    _assert_text_matches_hook(hook, content)
