# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The HTTP server reads the engine through public LLM members only.

Each test calls a production ``OpenAIServer`` or serving-class method on a
stand-in object (not ``object.__new__(OpenAIServer)``) whose generator double
defines exactly the public members the method reads, or builds the server
through the test factory, so a server method reaching for a private engine
member again fails loudly here.
"""

from types import SimpleNamespace

import pytest
from test_common.serve.app_factory import AggConfig
from test_common.serve.routes import effective_routes

from tensorrt_llm.llmapi.llm import EncodeLimits
from tensorrt_llm.serve import openai_server
from tensorrt_llm.serve._base import serving as base_serving
from tensorrt_llm.serve._base.serving import _record_generator_termination
from tensorrt_llm.serve.api.openai.embeddings.serving import EmbeddingServing
from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only


class _ServerStub(SimpleNamespace):
    """A SimpleNamespace whose unknown attributes are inert handlers.

    A production method under test may touch attributes the test does not
    care about; every unknown read resolves to a placeholder callable.
    """

    def __getattr__(self, name):
        def _handler(*args, **kwargs):
            return None

        return _handler


def _generator(**kwargs):
    kwargs.setdefault("args", SimpleNamespace(return_perf_metrics=False))
    return SimpleNamespace(**kwargs)


# ---------------------------------------------------------------------------
# EmbeddingServing builds its batcher from generator.encode_limits
# ---------------------------------------------------------------------------
def test_embedding_batcher_takes_the_limits_encode_checks():
    serving = EmbeddingServing(
        model="fake-model",
        generator=_generator(
            encode_limits=EncodeLimits(max_batch_size=4, max_num_tokens=512, max_seq_len=128),
            encode=lambda batch: batch,
        ),
        embedding_max_queue_delay=0.01,
        embedding_max_queue_size=8,
    )

    batcher = serving.embedding_batcher
    assert batcher._max_batch_size == 4
    assert batcher._max_num_tokens == 512
    assert batcher._max_seq_len == 128


# ---------------------------------------------------------------------------
# The composition root reads generator.resource_governor_queue
# ---------------------------------------------------------------------------
class _RecordingGovernor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.registered_app = None

    def register_routes(self, app):
        self.registered_app = app


def test_the_root_builds_the_governor_from_the_public_queue(monkeypatch, serve_env):
    monkeypatch.setattr("tensorrt_llm.serve.resource_governor.ResourceGovernor", _RecordingGovernor)

    server = serve_env.build_openai_server(AggConfig(governor_queue=True))

    assert isinstance(server.resource_governor, _RecordingGovernor)
    assert server.resource_governor.kwargs["resource_governor_queue"] is (
        server.generator.resource_governor_queue
    )
    # The control router hands the governor the app to register its routes on.
    assert server.resource_governor.registered_app is server.app
    assert server.app.state.ops_serving_control is server.resource_governor


def test_the_root_skips_the_governor_without_a_queue(monkeypatch, serve_env):
    monkeypatch.setattr("tensorrt_llm.serve.resource_governor.ResourceGovernor", _RecordingGovernor)

    server = serve_env.build_openai_server(AggConfig(governor_queue=False))

    assert server.resource_governor is None
    assert not hasattr(server.app.state, "ops_serving_control")
    assert not any(
        path.startswith("/_resource_governor") for path, _ in effective_routes(server.app)
    )


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
    # Patched where the helper looks the name up: it lives in serve/_base/serving.py.
    monkeypatch.setattr(base_serving, "record_termination_observation", outcomes.append)
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
