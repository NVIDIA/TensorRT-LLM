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
"""The validation-error answers of both trtllm-serve apps, pinned on the real servers.

Each test posts an invalid body to a server built through the real constructor and
checks status, body and side effects (the metrics call, the counter, the throttled log)
against today's answers. The validation text is read from the exception FastAPI raised
for that very request, so no test re-implements how a server renders it.

The two servers differ on purpose and the tests use values that show it: the aggregated
server caps the OpenAI-shaped text at 4096 characters and counts a rejected request on
its metrics collector; the disaggregated frontend keeps the whole text, counts on its
own Prometheus counter and logs the first and every 1000th OpenAI-shaped rejection.
"""

import asyncio
import logging
from unittest.mock import MagicMock, Mock

import fastapi.routing
import pytest
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from test_common.serve.app_factory import AggConfig, DisaggConfig

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.logger import logger

# threadleak is off: TestClients are created in the test body and closed by the serve_env
# fixture's teardown, which runs after the leak check.
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

# Rejected by every request model: the model name is not a string, and the batch request's
# list of requests is not a list.
INVALID_BODY = {"model": 1.5, "requests": "not-a-list"}
# A body whose echo in the validation text exceeds the aggregated server's cap.
LONG_INVALID_BODY = {"model": "m" * 10_000}
AGG_DETAIL_LIMIT = 4096
KEEP = "today's answer must be kept (status, body and side effects)"


def _openai_body(text):
    return {"error": text}


def _anthropic_body(text):
    return {"type": "error", "error": {"type": "invalid_request_error", "message": text}}


def _absent_collector():
    """A metrics collector the server treats as not configured (falsy), yet observable."""
    collector = MagicMock()
    collector.__bool__.return_value = False
    return collector


@pytest.fixture
def raised_validation_errors(monkeypatch):
    """The RequestValidationError objects FastAPI raised, in order.

    FastAPI raises the error from its routing module; a recording subclass installed there
    makes ``str(exc)`` of the very request available, so bodies are compared to the text the
    server saw, not to a re-implementation of it. The app's handler, registered for the base
    class, still receives the subclass.
    """
    raised = []

    class Recording(RequestValidationError):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            raised.append(self)

    monkeypatch.setattr(fastapi.routing, "RequestValidationError", Recording)
    return raised


@pytest.fixture
def warning_records(monkeypatch):
    """Messages the TRT-LLM logger emits at WARNING or above during the test.

    The library logger does not propagate, so ``caplog`` sees nothing; a handler is attached
    instead, and the logger is lowered to ``warning`` for the test because its default level
    drops warnings before any handler.
    """
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Capture(level=logging.WARNING)
    previous_level = logger.level
    monkeypatch.setattr(logger, "_set_from_env", False)
    logger.set_level("warning")
    logger._logger.addHandler(handler)
    try:
        yield records
    finally:
        logger._logger.removeHandler(handler)
        logger.set_level(previous_level)


def _validation_warnings(records):
    # The TRT-LLM logger prefixes each message with its own tags; match the text, not the start.
    return [message for message in records if "[validation]" in message]


def _counter_value(counter) -> float:
    return next(
        sample.value for sample in counter.collect()[0].samples if sample.name.endswith("_total")
    )


# ---------------------------------------------------------------------------
# Aggregated server
# ---------------------------------------------------------------------------

AGG_LLM_ROUTES = [
    ("/v1/chat/completions", _openai_body),
    ("/v1/completions", _openai_body),
    ("/v1/responses", _openai_body),
    ("/v1/messages", _anthropic_body),
    ("/v1/messages/count_tokens", _anthropic_body),
    ("/v1/messages/batches", _anthropic_body),
]


@pytest.mark.parametrize("metrics", ["collector", "none"])
@pytest.mark.parametrize("path, body_of", AGG_LLM_ROUTES, ids=[path for path, _ in AGG_LLM_ROUTES])
def test_agg_llm_bodies_and_metrics(serve_env, raised_validation_errors, path, body_of, metrics):
    server = serve_env.build_openai_server(AggConfig())
    collector = Mock() if metrics == "collector" else _absent_collector()
    server.metrics_collector = collector
    client = serve_env.client(server.app)

    response = client.post(path, json=INVALID_BODY)

    text = str(raised_validation_errors[-1])
    assert (response.status_code, response.json()) == (400, body_of(text)), f"{path}: {KEEP}"
    if metrics == "collector":
        collector.log_request_error.assert_called_once_with(http_code=400)
    else:
        collector.log_request_error.assert_not_called()


def test_agg_caps_only_the_openai_validation_text(serve_env, raised_validation_errors):
    server = serve_env.build_openai_server(AggConfig())
    client = serve_env.client(server.app)

    response = client.post("/v1/chat/completions", json=LONG_INVALID_BODY)
    text = str(raised_validation_errors[-1])
    assert len(text) > AGG_DETAIL_LIMIT, "the long body must make the validation text long"
    cut = text[:AGG_DETAIL_LIMIT] + " ... [truncated %d bytes]" % (len(text) - AGG_DETAIL_LIMIT)
    assert (response.status_code, response.json()) == (400, _openai_body(cut)), (
        f"OpenAI cap: {KEEP}"
    )

    response = client.post("/v1/messages", json=LONG_INVALID_BODY)
    text = str(raised_validation_errors[-1])
    assert len(text) > AGG_DETAIL_LIMIT
    assert (response.status_code, response.json()) == (400, _anthropic_body(text)), (
        f"the Anthropic envelope carries the whole text: {KEEP}"
    )


def test_agg_visual_gen_body(serve_env):
    server = serve_env.build_openai_server(AggConfig(role=ServerRole.VISUAL_GEN))
    server.metrics_collector = Mock()
    client = serve_env.client(server.app, lifespan=False)

    response = client.post("/v1/images/generations", json={"prompt": 123})
    assert (response.status_code, response.json()) == (
        422,
        {
            "object": "error",
            "message": "prompt: Input should be a valid string",
            "type": "BadRequestError",
            "param": None,
            "code": 422,
        },
    ), f"VisualGen envelope: {KEEP}"

    response = client.post("/v1/images/generations", json={"prompt": "a cat", "bogus": 1})
    assert response.status_code == 422
    assert response.json()["message"] == (
        "Unknown request field 'bogus'. Pass model-specific parameters via 'extra_params' instead."
    ), f"VisualGen unknown-field message: {KEEP}"

    server.metrics_collector.log_request_error.assert_not_called()


# ---------------------------------------------------------------------------
# Disaggregated frontend
# ---------------------------------------------------------------------------


def test_disagg_bodies_counter_and_throttled_log(
    serve_env, raised_validation_errors, warning_records
):
    server = serve_env.build_disagg_server(DisaggConfig())
    client = serve_env.client(server.app, lifespan=False)
    counter = server._perf_metrics_collector.validation_exceptions
    rejected_before = _counter_value(counter)

    # An OpenAI route: the whole text (no cap), counted, logged as the first rejection with
    # the compact error list rather than the echoed input.
    response = client.post("/v1/chat/completions", json=LONG_INVALID_BODY)
    text = str(raised_validation_errors[-1])
    assert len(text) > AGG_DETAIL_LIMIT
    assert (response.status_code, response.json()) == (400, _openai_body(text)), (
        f"OpenAI body: {KEEP}"
    )
    assert _counter_value(counter) == rejected_before + 1
    warnings = _validation_warnings(warning_records)
    assert len(warnings) == 1, warnings
    assert "[validation] POST /v1/chat/completions 400 (n=1): [{'loc':" in warnings[0], warnings
    assert "m" * 100 not in warnings[0], "the log carries the error list, not the echoed input"

    # Throttled: the 1000th rejection is logged, the 1001st is not.
    server._val_err_n = 999
    client.post("/v1/chat/completions", json=INVALID_BODY)
    client.post("/v1/chat/completions", json=INVALID_BODY)
    warnings = _validation_warnings(warning_records)
    assert len(warnings) == 2, warnings
    assert "(n=1000):" in warnings[1], warnings
    assert _counter_value(counter) == rejected_before + 3

    # An Anthropic route: the envelope, counted, never logged. The log counter is reset first,
    # so an OpenAI-shaped rejection in its place would be the first one and would log: the
    # Anthropic rejection neither logs nor advances that counter.
    server._val_err_n = 0
    response = client.post("/v1/messages", json=INVALID_BODY)
    text = str(raised_validation_errors[-1])
    assert (response.status_code, response.json()) == (400, _anthropic_body(text)), (
        f"Anthropic body: {KEEP}"
    )
    assert _counter_value(counter) == rejected_before + 4
    assert server._val_err_n == 0, "an Anthropic rejection must not advance the OpenAI log counter"
    assert len(_validation_warnings(warning_records)) == 2, "Anthropic rejections are not logged"
    # Positive control: the next OpenAI-shaped rejection is the first one again and is logged.
    client.post("/v1/chat/completions", json=INVALID_BODY)
    assert server._val_err_n == 1
    warnings = _validation_warnings(warning_records)
    assert len(warnings) == 3 and "(n=1):" in warnings[-1], warnings

    # Control characters in the method and the path are escaped in the log line. Such a path
    # matches no route over HTTP, so the app's own handler is called directly.
    server._val_err_n = 0
    handler = server.app.exception_handlers[RequestValidationError]
    scope = {
        "type": "http",
        "method": "P\x01ST",
        "path": "/v1/chat/completions\x7f",
        "headers": [],
        "query_string": b"",
        "app": server.app,
    }
    error = RequestValidationError(
        [{"type": "missing", "loc": ("body", "messages"), "msg": "Field required", "input": {}}]
    )
    response = asyncio.run(handler(Request(scope), error))
    assert response.status_code == 400
    assert (
        "[validation] P\\x01ST /v1/chat/completions\\x7f 400 (n=1): "
        in _validation_warnings(warning_records)[-1]
    ), warning_records
