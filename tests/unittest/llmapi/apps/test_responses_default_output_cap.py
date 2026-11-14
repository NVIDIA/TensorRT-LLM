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
"""A server-wide max_output_tokens for Responses requests that set none.

Agent clients rarely set max_output_tokens, and a request without one
generates until the model stops or the context is full.
TRTLLM_RESPONSES_DEFAULT_MAX_OUTPUT_TOKENS caps such requests for the whole
deployment: a request's own value always wins, the response reports the cap
it was generated under, and an unset variable changes nothing.
"""

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

import tensorrt_llm.serve.responses_utils as responses_utils
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ResponsesRequest
from tensorrt_llm.serve.responses_utils import (
    RESPONSES_DEFAULT_MAX_OUTPUT_TOKENS_ENV,
    ResponsesStreamingProcessor,
    request_preprocess,
    responses_default_max_output_tokens,
)

# The CPU-* CI stages run pytest with -m 'cpu_only'. Without this marker every
# test in the file is deselected, which pytest reports as exit code 5 and the
# stage reports as a failure.
pytestmark = pytest.mark.cpu_only


def _request(**overrides):
    fields = dict(model="test-model", input="hi", stream=False, store=False)
    fields.update(overrides)
    return ResponsesRequest(**fields)


def _set_env(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv(RESPONSES_DEFAULT_MAX_OUTPUT_TOKENS_ENV, raising=False)
    else:
        monkeypatch.setenv(RESPONSES_DEFAULT_MAX_OUTPUT_TOKENS_ENV, raw)


@pytest.fixture()
def logged_warnings(monkeypatch):
    """The warnings the TRT-LLM logger emits, with its one-shot warnings armed again.

    A warning passes the logger's own severity gate and then the Python logger's
    level, so both are opened for the test whatever TLLM_LOG_LEVEL says.
    """
    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = Capture(level=logging.WARNING)
    monkeypatch.setattr(logger, "_min_severity", "warning")
    monkeypatch.setattr(logger, "_appeared_keys", set())
    previous_level = logger._logger.level
    logger._logger.setLevel(logging.WARNING)
    logger._logger.addHandler(handler)
    try:
        yield messages
    finally:
        logger._logger.removeHandler(handler)
        logger._logger.setLevel(previous_level)


class TestDefaultOutputCap:
    def test_the_variable_name(self):
        assert (
            RESPONSES_DEFAULT_MAX_OUTPUT_TOKENS_ENV == "TRTLLM_RESPONSES_DEFAULT_MAX_OUTPUT_TOKENS"
        )

    def test_a_request_with_no_cap_inherits_the_default(self):
        params = _request().to_sampling_params(default_sampling_params={"max_tokens": 32768})
        assert params.max_tokens == 32768

    def test_the_request_wins_over_the_default(self):
        params = _request(max_output_tokens=10).to_sampling_params(
            default_sampling_params={"max_tokens": 32768}
        )
        assert params.max_tokens == 10

    def test_an_explicit_zero_is_the_request_s_value_not_the_default(self):
        # Only an absent value takes the default; 0 goes through as today and the
        # executor rejects it as it always has.
        params = _request(max_output_tokens=0).to_sampling_params(
            default_sampling_params={"max_tokens": 32768}
        )
        assert params.max_tokens == 0

    def test_no_default_leaves_the_request_uncapped(self):
        assert _request().to_sampling_params().max_tokens is None
        assert _request().to_sampling_params(default_sampling_params={}).max_tokens is None
        assert (
            _request().to_sampling_params(default_sampling_params={"max_tokens": None}).max_tokens
            is None
        )

    @pytest.mark.parametrize(
        "raw, expected",
        [(None, None), ("", None), ("  ", None), ("32768", 32768), (" 64 ", 64)],
    )
    def test_the_variable_is_read_as_a_positive_integer(
        self, monkeypatch, logged_warnings, raw, expected
    ):
        _set_env(monkeypatch, raw)
        assert responses_default_max_output_tokens() == expected
        assert logged_warnings == []

    @pytest.mark.parametrize("raw", ["0", "-5", "many", "1.5", "1e3"])
    def test_a_value_that_is_not_a_positive_integer_is_ignored_with_one_warning(
        self, monkeypatch, logged_warnings, raw
    ):
        _set_env(monkeypatch, raw)
        assert responses_default_max_output_tokens() is None
        assert responses_default_max_output_tokens() is None
        assert len(logged_warnings) == 1, logged_warnings
        assert RESPONSES_DEFAULT_MAX_OUTPUT_TOKENS_ENV in logged_warnings[0]
        assert repr(raw) in logged_warnings[0]


_HARMONY_STOP_TOKENS = [200002]


def _preprocess(monkeypatch, request, use_harmony=False):
    """The sampling params request_preprocess builds, with the prompt rendering stubbed out.

    The harmony adapter is stubbed too, so both rendering paths run without a model.
    """

    async def fake_create_input_tokens(**kwargs):
        return [1, 2, 3]

    monkeypatch.setattr(responses_utils, "_create_input_tokens", fake_create_input_tokens)
    monkeypatch.setattr(responses_utils, "_create_input_tokens_harmony", fake_create_input_tokens)
    monkeypatch.setattr(
        responses_utils,
        "get_harmony_adapter",
        lambda: SimpleNamespace(get_stop_tokens=lambda: list(_HARMONY_STOP_TOKENS)),
    )
    _, sampling_params = asyncio.run(
        request_preprocess(
            request=request,
            prev_response=None,
            conversation_store=None,
            enable_store=False,
            use_harmony=use_harmony,
        )
    )
    # The path that ran is visible in the stop tokens it contributed.
    if use_harmony:
        assert sampling_params.stop_token_ids == _HARMONY_STOP_TOKENS
    else:
        assert not sampling_params.stop_token_ids
    return sampling_params


@pytest.mark.parametrize("use_harmony", [False, True])
class TestServerPath:
    def test_an_unset_variable_changes_nothing(self, monkeypatch, use_harmony):
        _set_env(monkeypatch, None)
        assert _preprocess(monkeypatch, _request(), use_harmony).max_tokens is None
        assert (
            _preprocess(monkeypatch, _request(max_output_tokens=10), use_harmony).max_tokens == 10
        )

    def test_the_variable_caps_a_request_that_sets_no_max_output_tokens(
        self, monkeypatch, use_harmony
    ):
        _set_env(monkeypatch, "4096")
        assert _preprocess(monkeypatch, _request(), use_harmony).max_tokens == 4096

    def test_the_request_wins_on_the_server_path_too(self, monkeypatch, use_harmony):
        _set_env(monkeypatch, "4096")
        assert (
            _preprocess(monkeypatch, _request(max_output_tokens=10), use_harmony).max_tokens == 10
        )


def _created_response(request, default_max_tokens):
    """The response snapshot the stream opens with."""
    sampling_params = request.to_sampling_params(
        default_sampling_params={"max_tokens": default_max_tokens}
    )
    processor = ResponsesStreamingProcessor(
        request=request,
        sampling_params=sampling_params,
        model_name="test-model",
        use_harmony=False,
    )
    created = processor.get_initial_responses()[0]
    assert created.startswith("event: response.created\n")
    return json.loads(created.split("data: ", 1)[1])["response"]


class TestEcho:
    def test_the_response_reports_the_cap_it_was_generated_under(self):
        assert _created_response(_request(stream=True), 4096)["max_output_tokens"] == 4096

    def test_a_request_with_its_own_cap_sees_that_cap(self):
        assert (
            _created_response(_request(stream=True, max_output_tokens=10), 4096)[
                "max_output_tokens"
            ]
            == 10
        )

    def test_without_a_cap_the_response_reports_none(self):
        assert _created_response(_request(stream=True), None)["max_output_tokens"] is None
