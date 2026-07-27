# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /perf_metrics population on the Harmony (GPT-OSS) chat path.

Asserts two behaviors of the Harmony chat endpoint:
  * The streaming generator stamps `server_first_token_time` and calls
    `_extract_metrics(...)` after `data: [DONE]`, so the `/perf_metrics`
    deque is populated for streaming chat requests.
  * When `TRTLLM_KVCACHE_TIME_OUTPUT_PATH` is set, the per-request
    `sampling_params.return_perf_metrics` flag is enabled so the engine
    emits metrics for each request. Without it, the deque stays empty
    even with `return_perf_metrics: True` at the LLM-args level.
"""

import json
import os
import tempfile
from urllib.request import urlopen

import openai
import pytest
import yaml
from utils.llm_data import llm_datasets_root

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)
os.environ["TIKTOKEN_RS_CACHE_DIR"] = os.path.join(llm_datasets_root(), "tiktoken_vocab")
os.environ["TIKTOKEN_ENCODINGS_BASE"] = os.path.join(llm_datasets_root(), "tiktoken_vocab")


@pytest.fixture(scope="module", ids=["GPT-OSS-20B"])
def model():
    return "gpt_oss/gpt-oss-20b/"


@pytest.fixture(scope="module", params=[0, 2], ids=["disable_processpool", "enable_processpool"])
def num_postprocess_workers(request):
    return request.param


@pytest.fixture(scope="module")
def kv_cache_time_output_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("kv_cache_time_output"))


@pytest.fixture(scope="module")
def extra_llm_api_options_file():
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="extra_llm_api_options_")
    os.close(fd)
    try:
        with open(path, "w") as f:
            yaml.dump(
                {
                    "return_perf_metrics": True,
                    "perf_metrics_max_requests": 16,
                },
                f,
            )
        yield path
    finally:
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture(scope="module")
def server(
    model: str,
    num_postprocess_workers: int,
    kv_cache_time_output_dir: str,
    extra_llm_api_options_file: str,
):
    model_path = get_model_path(model)
    args = [
        "--num_postprocess_workers",
        f"{num_postprocess_workers}",
        "--extra_llm_api_options",
        extra_llm_api_options_file,
    ]
    env = os.environ.copy()
    # Non-empty value flips `sampling_params.return_perf_metrics` per request
    # inside the Harmony chat path; the value itself is also consumed by the
    # C++ KV cache layer for CSV dumps.
    env["TRTLLM_KVCACHE_TIME_OUTPUT_PATH"] = kv_cache_time_output_dir
    with RemoteOpenAIServer(model_path, args, env=env) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer):
    return server.get_async_client()


def _drain_perf_metrics(server: RemoteOpenAIServer):
    response = urlopen(f"{server.url_root}/perf_metrics")
    assert response.status == 200
    return json.loads(response.read())


def _assert_perf_metrics_entry_well_formed(entry: dict):
    assert "request_id" in entry
    assert "perf_metrics" in entry
    pm = entry["perf_metrics"]
    assert "first_iter" in pm and "last_iter" in pm
    assert pm["first_iter"] <= pm["last_iter"]

    tm = pm["timing_metrics"]
    for key in ("arrival_time", "first_scheduled_time", "first_token_time", "last_token_time"):
        assert key in tm, f"missing timing_metrics.{key}"
    assert tm["arrival_time"] <= tm["first_scheduled_time"]
    assert tm["first_scheduled_time"] <= tm["first_token_time"]
    assert tm["first_token_time"] <= tm["last_token_time"]


@pytest.mark.asyncio(loop_scope="module")
async def test_non_streaming_perf_metrics(
    async_client: openai.AsyncOpenAI, server: RemoteOpenAIServer, model: str
):
    # Drain anything from prior tests in this module session.
    _drain_perf_metrics(server)
    response = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly the single word: PONG."}],
        extra_body={"top_k": 1},
    )
    assert response.choices[0].message.content is not None

    entries = _drain_perf_metrics(server)
    assert len(entries) == 1, (
        "Expected exactly one /perf_metrics entry after a single non-streaming "
        f"harmony chat completion, got {len(entries)}: {entries}"
    )
    _assert_perf_metrics_entry_well_formed(entries[0])


@pytest.mark.asyncio(loop_scope="module")
async def test_streaming_perf_metrics(
    async_client: openai.AsyncOpenAI, server: RemoteOpenAIServer, model: str
):
    # Drain anything from prior tests in this module session.
    _drain_perf_metrics(server)
    stream = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Explain transformers in one sentence."}],
        stream=True,
        extra_body={"top_k": 1},
    )
    saw_done = False
    async for _chunk in stream:
        # We don't need to inspect content; just consume the stream so the
        # server-side generator runs through `[DONE]` and `_extract_metrics`.
        saw_done = True
    assert saw_done, "Streaming chat returned no chunks"

    entries = _drain_perf_metrics(server)
    assert len(entries) == 1, (
        "Expected exactly one /perf_metrics entry after a single streaming "
        f"harmony chat completion, got {len(entries)}: {entries}. "
        "This usually means _extract_metrics did not run after [DONE]."
    )
    _assert_perf_metrics_entry_well_formed(entries[0])
