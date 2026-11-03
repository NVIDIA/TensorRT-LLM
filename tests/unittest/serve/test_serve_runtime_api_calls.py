# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The HTTP server reads the engine through public LLM members only.

Each test calls a production ``OpenAIServer`` method on a stand-in object (not
``object.__new__(OpenAIServer)``) whose generator double defines exactly the
public members the method reads, so a server method reaching for a private
engine member again fails loudly here.
"""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from tensorrt_llm.llmapi.llm import EncodeLimits
from tensorrt_llm.serve import openai_server
from tensorrt_llm.serve._base.serving import _record_generator_termination
from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only


class _ServerStub(SimpleNamespace):
    """A SimpleNamespace whose unknown attributes are inert handlers.

    ``register_routes`` touches about thirty handler attributes; the routes
    under test here are only the resource-governor ones, so every other
    handler read resolves to a placeholder callable FastAPI can register.
    """

    def __getattr__(self, name):
        def _handler(*args, **kwargs):
            return None

        return _handler


def _generator(**kwargs):
    kwargs.setdefault("args", SimpleNamespace(return_perf_metrics=False))
    return SimpleNamespace(**kwargs)


# ---------------------------------------------------------------------------
# _init_embedding_batcher reads generator.encode_limits
# ---------------------------------------------------------------------------
def test_embedding_batcher_takes_the_limits_encode_checks():
    stub = _ServerStub(
        generator=_generator(
            encode_limits=EncodeLimits(max_batch_size=4, max_num_tokens=512, max_seq_len=128),
            encode=lambda batch: batch,
        ),
        _embedding_max_queue_delay=0.01,
        _embedding_max_queue_size=8,
    )

    OpenAIServer._init_embedding_batcher(stub)

    batcher = stub.embedding_batcher
    assert batcher._max_batch_size == 4
    assert batcher._max_num_tokens == 512
    assert batcher._max_seq_len == 128


# ---------------------------------------------------------------------------
# register_routes reads generator.resource_governor_queue
# ---------------------------------------------------------------------------
class _RecordingGovernor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.registered_app = None

    def register_routes(self, app):
        self.registered_app = app


def _register_routes_stub(resource_governor_queue):
    return _ServerStub(
        app=FastAPI(),
        generator=_generator(resource_governor_queue=resource_governor_queue),
        tokenizer=object(),
        model_config=None,
        processor=None,
        allow_request_chat_template=False,
        use_harmony=False,
        chat_template=None,
    )


def test_register_routes_builds_the_governor_from_the_public_queue(monkeypatch):
    queue = object()
    stub = _register_routes_stub(resource_governor_queue=queue)
    monkeypatch.setattr("tensorrt_llm.serve.resource_governor.ResourceGovernor", _RecordingGovernor)

    OpenAIServer.register_routes(stub)

    assert isinstance(stub.resource_governor, _RecordingGovernor)
    assert stub.resource_governor.kwargs["resource_governor_queue"] is queue
    assert stub.resource_governor.registered_app is stub.app


def test_register_routes_skips_the_governor_without_a_queue(monkeypatch):
    stub = _register_routes_stub(resource_governor_queue=None)
    monkeypatch.setattr("tensorrt_llm.serve.resource_governor.ResourceGovernor", _RecordingGovernor)

    OpenAIServer.register_routes(stub)

    assert stub.resource_governor is None
    assert not any(route.path.startswith("/_resource_governor") for route in stub.app.routes)


# ---------------------------------------------------------------------------
# _init_llm reads generator.hf_model_dir
# ---------------------------------------------------------------------------
def test_init_llm_loads_processor_and_config_from_the_public_model_dir(monkeypatch):
    class _Config:
        model_type = "llama"

    processor_calls, config_calls = [], []
    monkeypatch.setattr(
        openai_server.AutoProcessor,
        "from_pretrained",
        classmethod(lambda cls, path, **kwargs: processor_calls.append(path)),
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.config_utils.load_pretrained_config",
        lambda path, **kwargs: config_calls.append(path) or _Config(),
    )
    stub = _ServerStub(
        generator=_generator(
            tokenizer=object(),
            hf_model_dir="/models/resolved-dir",
            args=SimpleNamespace(
                trust_remote_code=False,
                checkpoint_format=None,
                post_processor_hook=None,
                return_perf_metrics=False,
            ),
        ),
        postproc_worker_enabled=False,
        _launcher_forwarder=None,
    )

    OpenAIServer._init_llm(stub)

    assert processor_calls == ["/models/resolved-dir"]
    assert config_calls == ["/models/resolved-dir"]


# ---------------------------------------------------------------------------
# _record_generator_termination reads generator.engine_in_worker_process
# ---------------------------------------------------------------------------
def _recorded_outcome(monkeypatch, generator):
    outcomes = []
    monkeypatch.setattr(openai_server, "record_termination_observation", outcomes.append)
    _record_generator_termination(generator)
    assert len(outcomes) == 1
    return outcomes[0]


def test_termination_telemetry_blames_the_worker_behind_the_proxy(monkeypatch):
    outcome = _recorded_outcome(monkeypatch, _generator(engine_in_worker_process=True))
    assert outcome.termination_kind == "worker_failure"
    assert outcome.component == "engine_worker"
    assert outcome.reporting_source == "executor_proxy"


def test_termination_telemetry_blames_the_llm_for_in_process_engines(monkeypatch):
    outcome = _recorded_outcome(monkeypatch, _generator(engine_in_worker_process=False))
    assert outcome.termination_kind == "exception"
    assert outcome.component == "llm"
    assert outcome.reporting_source == "self"


def test_termination_telemetry_defaults_for_generators_without_the_member(monkeypatch):
    """The telemetry guard must not raise for doubles lacking the property."""
    outcome = _recorded_outcome(monkeypatch, object())
    assert outcome.termination_kind == "exception"
    assert outcome.component == "llm"
    assert outcome.reporting_source == "self"
