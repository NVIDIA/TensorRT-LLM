# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Regression tests for issue #13949.

Ensures /v1/responses perf metrics parity with chat/completions, validated
through the JSONL output interface (``perf_metrics_output_dir``).
"""

import asyncio
import inspect
import logging
import os
import re
import tempfile
from urllib.request import urlopen

import pytest
import yaml
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseTextDeltaEvent,
)
from test_common.perf_metrics_utils import read_perf_metrics_jsonl, wait_for_perf_metrics_jsonl

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# A small reasoning-capable model that the CI already uses for responses tests.
_MODEL = "Qwen3/Qwen3-0.6B"
_INPUT = "What is 1+1? Answer briefly."


@pytest.fixture(scope="module")
def perf_metrics_output_dir(tmp_path_factory):
    """Shared directory where the server writes perf_metrics-*.jsonl records."""
    return tmp_path_factory.mktemp("perf_metrics")


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file(perf_metrics_output_dir):
    """Write a temporary YAML file enabling JSONL perf metrics output and yield its path."""
    fd, temp_file_path = tempfile.mkstemp(
        prefix="responses_perf_metrics_options_",
        suffix=".yaml",
    )
    os.close(fd)
    try:
        extra_llm_api_options_dict = {
            "return_perf_metrics": True,
            "perf_metrics_output_dir": str(perf_metrics_output_dir),
        }
        with open(temp_file_path, "w", encoding="utf-8") as f:
            yaml.dump(extra_llm_api_options_dict, f)
        yield temp_file_path
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@pytest.fixture(scope="module")
def server(temp_extra_llm_api_options_file: str) -> RemoteOpenAIServer:
    """Start a RemoteOpenAIServer with perf-metrics enabled and yield it to tests."""
    model_path = get_model_path(_MODEL)
    args = [
        "--reasoning_parser",
        "qwen3",
        "--tool_parser",
        "qwen3",
        "--extra_llm_api_options",
        temp_extra_llm_api_options_file,
    ]
    logger.info(f"Starting responses perf-metrics server: model={_MODEL}")
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server
        logger.info("Tests completed, shutting down server")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_perf_metrics_entry_valid(entry: dict, context: str = ""):
    """Assert that a single JSONL perf_metrics record has the expected structure."""
    assert "request_id" in entry, f"{context}: missing 'request_id'"
    assert entry["status"] == "complete", f"{context}: status != 'complete'"
    assert "perf_metrics" in entry, f"{context}: missing 'perf_metrics'"

    data = entry["perf_metrics"]
    assert "first_iter" in data, f"{context}: missing 'first_iter'"
    assert "last_iter" in data, f"{context}: missing 'last_iter'"
    assert data["first_iter"] <= data["last_iter"], f"{context}: first_iter > last_iter"

    timing = data["timing_metrics"]
    for key in ("arrival_time", "first_scheduled_time", "first_token_time", "last_token_time"):
        assert key in timing, f"{context}: missing timing key '{key}'"

    assert timing["arrival_time"] < timing["first_scheduled_time"], (
        f"{context}: arrival_time not before first_scheduled_time"
    )
    assert timing["first_scheduled_time"] < timing["first_token_time"], (
        f"{context}: first_scheduled_time not before first_token_time"
    )
    assert timing["first_token_time"] <= timing["last_token_time"], (
        f"{context}: first_token_time after last_token_time"
    )

    kv_cache = data["kv_cache_metrics"]
    assert kv_cache["num_new_allocated_blocks"] <= kv_cache["num_total_allocated_blocks"], (
        f"{context}: num_new_allocated_blocks > num_total_allocated_blocks"
    )


def _parse_prometheus_counter(data: str, metric_name: str) -> float | None:
    """Return the value of a Prometheus counter/gauge line, or None."""
    pattern = re.compile(
        r"^" + re.escape(metric_name) + r"(?:\{[^}]*\})?" + r"\s+(\S+)",
        re.MULTILINE,
    )
    match = pattern.search(data)
    return float(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_non_streaming_responses_populates_perf_metrics(
    server: RemoteOpenAIServer,
    perf_metrics_output_dir,
):
    """Non-streaming /v1/responses must append one JSONL record."""
    # Establish baseline count (other tests may have run first in the module).
    count_before = len(read_perf_metrics_jsonl(perf_metrics_output_dir))

    client = server.get_client()
    client.responses.create(
        model=_MODEL,
        input=_INPUT,
        max_output_tokens=32,
        stream=False,
    )

    records = wait_for_perf_metrics_jsonl(perf_metrics_output_dir, count_before + 1)
    assert len(records) == count_before + 1, (
        f"Expected exactly one new JSONL record after non-streaming "
        f"/v1/responses; got {len(records) - count_before} new records"
    )

    new_entry = records[-1]
    _assert_perf_metrics_entry_valid(new_entry, "non-streaming responses")
    # Disaggregated-only fields must be absent.
    assert "ctx_request_id" not in new_entry


def test_streaming_responses_populates_perf_metrics(
    server: RemoteOpenAIServer,
    perf_metrics_output_dir,
):
    """Streaming /v1/responses must append one JSONL record after stream completes."""
    count_before = len(read_perf_metrics_jsonl(perf_metrics_output_dir))

    client = server.get_client()
    # Consume the entire stream so that create_streaming_generator runs to
    # completion and _extract_metrics is called.
    stream = client.responses.create(
        model=_MODEL,
        input=_INPUT,
        max_output_tokens=32,
        stream=True,
    )
    for _ in stream:
        pass

    records = wait_for_perf_metrics_jsonl(perf_metrics_output_dir, count_before + 1)
    assert len(records) == count_before + 1, (
        f"Expected exactly one new JSONL record after streaming "
        f"/v1/responses; got {len(records) - count_before} new records"
    )

    new_entry = records[-1]
    _assert_perf_metrics_entry_valid(new_entry, "streaming responses")
    assert "ctx_request_id" not in new_entry


@pytest.mark.asyncio(loop_scope="module")
async def test_incomplete_streaming_responses_does_not_populate_perf_metrics(
    server: RemoteOpenAIServer,
    perf_metrics_output_dir,
):
    """Partially consumed /v1/responses streams must not append completed perf_metrics records."""
    count_before = len(read_perf_metrics_jsonl(perf_metrics_output_dir))

    client = server.get_async_client()
    cancel_input = (
        "Write 12 numbered bullet points about how a streamed LLM server should "
        "handle request cancellation, with each bullet as a full sentence."
    )
    stream = await client.responses.create(
        model=_MODEL,
        input=cancel_input,
        max_output_tokens=256,
        stream=True,
    )

    found_delta = False
    try:
        async for event in stream:
            assert not isinstance(event, ResponseCompletedEvent), (
                "Expected a non-terminal delta event before cancellation, "
                "but received the completed response event instead"
            )
            if isinstance(event, (ResponseTextDeltaEvent, ResponseReasoningTextDeltaEvent)):
                found_delta = True
                assert len(read_perf_metrics_jsonl(perf_metrics_output_dir)) == count_before, (
                    "JSONL perf metrics changed before the stream was cancelled, "
                    "which indicates the request may have already completed"
                )
                break
            # Legitimate lifecycle events emitted before the first model-output delta.
            assert isinstance(
                event,
                (
                    ResponseCreatedEvent,
                    ResponseInProgressEvent,
                    ResponseOutputItemAddedEvent,
                    ResponseContentPartAddedEvent,
                ),
            ), f"Unexpected event before first delta: {type(event).__name__}: {event!r}"
    finally:
        # Explicitly close the stream early to simulate client cancellation.
        if hasattr(stream, "aclose"):
            await stream.aclose()
        elif hasattr(stream, "close"):
            maybe_awaitable = stream.close()
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable

        # Close async client deterministically.
        close_method = getattr(client, "close", None)
        if close_method is not None:
            maybe_awaitable = close_method()
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable

    assert found_delta, "Expected at least one text/reasoning delta before cancellation"

    # Observe for a short window to ensure no completed record is appended later.
    for _ in range(6):
        await asyncio.sleep(0.5)
        assert len(read_perf_metrics_jsonl(perf_metrics_output_dir)) == count_before, (
            "Cancelled /v1/responses stream should not append perf metrics JSONL records"
        )


def test_repeated_requests_append_multiple_entries(
    server: RemoteOpenAIServer,
    perf_metrics_output_dir,
):
    """Each /v1/responses request must append exactly one record (no duplicates)."""
    count_before = len(read_perf_metrics_jsonl(perf_metrics_output_dir))

    client = server.get_client()
    n = 3
    for _ in range(n):
        client.responses.create(
            model=_MODEL,
            input=_INPUT,
            max_output_tokens=16,
            stream=False,
        )

    records = wait_for_perf_metrics_jsonl(perf_metrics_output_dir, count_before + n)
    assert len(records) == count_before + n, (
        f"Expected {n} new JSONL records after {n} non-streaming /v1/responses "
        f"requests; got {len(records) - count_before}"
    )

    # Every new record must be structurally valid.
    for i, entry in enumerate(records[count_before:], start=1):
        _assert_perf_metrics_entry_valid(entry, f"repeated request {i}")


def test_responses_request_id_present_in_metrics(
    server: RemoteOpenAIServer,
    perf_metrics_output_dir,
):
    """Each JSONL perf_metrics record for /v1/responses must contain a non-empty request_id."""
    count_before = len(read_perf_metrics_jsonl(perf_metrics_output_dir))

    client = server.get_client()
    client.responses.create(
        model=_MODEL,
        input=_INPUT,
        max_output_tokens=16,
        stream=False,
    )

    records = wait_for_perf_metrics_jsonl(perf_metrics_output_dir, count_before + 1)
    assert len(records) == count_before + 1, (
        "Expected exactly one new JSONL record after /v1/responses"
    )

    latest = records[-1]
    assert "request_id" in latest, "request_id missing from perf_metrics JSONL record"
    assert latest["request_id"], "request_id is empty in perf_metrics JSONL record"


def test_prometheus_counter_advances_for_responses(server: RemoteOpenAIServer):
    """Prometheus request_success_total must advance after /v1/responses."""
    METRIC = "trtllm_request_success_total"

    def _get_counter() -> float:
        resp = urlopen(f"{server.url_root}/prometheus/metrics", timeout=5)  # noqa: S310
        assert resp.status == 200
        data = resp.read().decode("utf-8")
        value = _parse_prometheus_counter(data, METRIC)
        return value if value is not None else 0.0

    before = _get_counter()

    client = server.get_client()
    client.responses.create(
        model=_MODEL,
        input=_INPUT,
        max_output_tokens=16,
        stream=False,
    )

    after = _get_counter()
    assert after > before, (
        f"Prometheus {METRIC} did not advance after /v1/responses (before={before}, after={after})"
    )


def test_parity_with_chat_completions(
    server: RemoteOpenAIServer,
    perf_metrics_output_dir,
):
    """Both /v1/responses and /v1/chat/completions must produce structurally identical JSONL records."""
    client = server.get_client()

    # Chat completions entry.
    count_before = len(read_perf_metrics_jsonl(perf_metrics_output_dir))
    client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": _INPUT}],
        max_tokens=16,
        stream=False,
    )
    records = wait_for_perf_metrics_jsonl(perf_metrics_output_dir, count_before + 1)
    assert len(records) == count_before + 1, (
        "chat/completions did not produce exactly one JSONL perf metrics record"
    )
    chat_entry = records[-1]

    # Responses API entry.
    count_before = len(read_perf_metrics_jsonl(perf_metrics_output_dir))
    client.responses.create(
        model=_MODEL,
        input=_INPUT,
        max_output_tokens=16,
        stream=False,
    )
    records = wait_for_perf_metrics_jsonl(perf_metrics_output_dir, count_before + 1)
    assert len(records) == count_before + 1, (
        "/v1/responses did not produce exactly one JSONL perf metrics record"
    )
    resp_entry = records[-1]

    # Both records must have the same top-level keys.
    chat_keys = set(chat_entry.keys())
    resp_keys = set(resp_entry.keys())
    assert chat_keys == resp_keys, (
        f"Key mismatch between chat/completions and /v1/responses records: "
        f"chat={chat_keys}, responses={resp_keys}"
    )

    # Both records must have structurally valid perf_metrics.
    _assert_perf_metrics_entry_valid(chat_entry, "chat/completions parity")
    _assert_perf_metrics_entry_valid(resp_entry, "responses parity")
