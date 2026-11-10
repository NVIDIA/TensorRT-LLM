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
"""Factory that builds every trtllm-serve app on CPU through the real constructors.

``isolated_serve_env`` is the only entry point: building a server touches
process-global state (the default Prometheus registry, ``BaseMediaIO``'s
executor, environment variables, the prometheus multiprocess directory, the
telemetry session), and the context manager both prepares and restores all of
it. Use it instead of ``object.__new__(OpenAIServer)``:

    with isolated_serve_env(tmp_path) as env:
        server = env.build_openai_server(AggConfig())
        client = env.client(server.app)
        ...

or the ``serve_env`` fixture from ``tests/unittest/serve/conftest.py``.

Inside an environment, change the state it saves through the environment
(``AggConfig.env``, ``record_terminations()``), not through ``monkeypatch``:
pytest may undo a monkeypatch change after the environment's exit, which then
overwrites the value the environment restored.
"""

import contextlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, List, Optional
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import tensorrt_llm._utils
from tensorrt_llm.inputs.media_io import BaseMediaIO
from tensorrt_llm.llmapi.disagg_utils import (
    CtxGenServerConfig,
    DisaggClusterConfig,
    DisaggServerConfig,
    RouterConfig,
    ServerRole,
)
from tensorrt_llm.serve.coordinator_server import CoordinatorServer
from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinatorService
from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.usage import usage_lib
from test_common.serve.fakes import (
    FakeAsyncLLM,
    FakeEncodeOnlyLLM,
    FakeLLM,
    FakeLlmArgs,
    FakeMultimodalEncoder,
)

__all__ = [
    "AggConfig",
    "CoordinatorConfig",
    "DisaggConfig",
    "ServeTestEnv",
    "isolated_serve_env",
]

_MISSING = object()

# Environment variables that change route registration or request handling;
# removed on entry so every configuration starts from the same baseline and
# sets what it needs through AggConfig.env.
_CLEARED_ENV_VARS = (
    "ENABLE_METRICS",
    "TRTLLM_ENABLE_RENDER_ENDPOINTS",
    "DISABLE_HARMONY_ADAPTER",
    "TRTLLM_RESPONSES_API_DISABLE_STORE",
    "TRTLLM_KVCACHE_TIME_OUTPUT_PATH",
    "TRTLLM_REQUEST_TRACE_DIR",
    "TRTLLM_REQUEST_TRACE_IDENTITY",
)


@dataclass(frozen=True)
class AggConfig:
    """One ``OpenAIServer`` configuration, built through the real constructor."""

    role: Optional[ServerRole] = None
    model_type: str = "llama"  # "gpt_oss" makes use_harmony true
    governor_queue: bool = False  # FakeExecutor(resource_governor_queue=object())
    rl_control: bool = False  # FakeAsyncLLM + enable_rl_control_endpoints
    runtime_control: bool = False  # sleep_config + enable_runtime_control_endpoints
    return_perf_metrics: bool = False
    perf_metrics_output_dir: bool = False  # sets args.perf_metrics_output_dir=<tmp>/perf
    env: tuple = ()  # (name, value) pairs applied before construction
    model_name: str = "fake-model"  # OpenAIServer(model=...): the /v1/models id
    tool_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    num_postprocess_workers: int = 0  # > 0 takes the postprocess-worker path
    # Attached frontend of a multi-frontend server: registers the launcher
    # forwarding routes ahead of the local handlers, which they shadow.
    multi_frontend_attached: bool = False


@dataclass(frozen=True)
class DisaggConfig:
    """One ``OpenAIDisaggServer`` configuration."""

    fleet: bool = False  # delegate to a remote coordinator (CoordinatorClient)
    http_cluster_storage: bool = False
    return_perf_metrics: bool = False


@dataclass(frozen=True)
class CoordinatorConfig:
    """One ``CoordinatorServer`` configuration (the fleet-mode coordinator app)."""

    http_cluster_storage: bool = False


def _make_disagg_server_config(
    *, return_perf_metrics: bool = False, http_cluster_storage: bool = False
) -> DisaggServerConfig:
    # Loopback addresses on closed ports: construction is network-free, and
    # nothing in these tests runs the disaggregated lifespan.
    return DisaggServerConfig(
        server_configs=[
            CtxGenServerConfig(type="ctx", hostname="127.0.0.1", port=9),
            CtxGenServerConfig(type="gen", hostname="127.0.0.1", port=10),
        ],
        hostname="127.0.0.1",
        port=0,
        ctx_router_config=RouterConfig(server_role=ServerRole.CONTEXT),
        gen_router_config=RouterConfig(server_role=ServerRole.GENERATION),
        node_id=0,
        return_perf_metrics=return_perf_metrics,
        disagg_cluster_config=(
            DisaggClusterConfig(cluster_uri="http://127.0.0.1:9", cluster_name="test")
            if http_cluster_storage
            else None
        ),
    )


class _StubTelemetrySession:
    """Collects termination observations for ``ServeTestEnv.record_terminations``."""

    def __init__(self, sink: list) -> None:
        self.owner_pid = os.getpid()
        self._sink = sink

    def record_termination_observation(self, outcome, lifecycle_phase=None) -> None:
        self._sink.append(outcome)


class ServeTestEnv:
    """Handle returned by ``isolated_serve_env``; builds servers and clients."""

    def __init__(self, tmp_dir: Path) -> None:
        self.tmp_dir = Path(tmp_dir)
        self._servers: List[OpenAIServer] = []
        self._entered_clients: List[TestClient] = []
        self._plain_clients: List[TestClient] = []

    # -- builders ---------------------------------------------------------

    def _write_fake_checkpoint(self, model_type: str) -> Path:
        """A directory with only a config.json, so load_pretrained_config works.

        No tokenizer files are written: they would make AutoProcessor succeed
        and change ``server.processor`` from its fake-generator value (None).
        """
        model_dir = self.tmp_dir / f"model-{model_type}"
        model_dir.mkdir(parents=True, exist_ok=True)
        architectures = {
            "llama": ["LlamaForCausalLM"],
            "gpt_oss": ["GptOssForCausalLM"],
        }[model_type]
        (model_dir / "config.json").write_text(
            json.dumps({"model_type": model_type, "architectures": architectures})
        )
        return model_dir

    def build_openai_server(self, config: AggConfig) -> OpenAIServer:
        from test_common.serve.fakes import FakeExecutor  # local: keeps the import graph flat

        for name, value in config.env:
            os.environ[name] = value

        args = FakeLlmArgs(
            model=str(self._write_fake_checkpoint(config.model_type)),
            return_perf_metrics=config.return_perf_metrics,
            perf_metrics_output_dir=(
                str(self.tmp_dir / "perf") if config.perf_metrics_output_dir else None
            ),
            sleep_config=SimpleNamespace() if config.runtime_control else None,
            encode_only=config.role is ServerRole.EMBEDDING,
            reasoning_parser=config.reasoning_parser,
            num_postprocess_workers=config.num_postprocess_workers,
        )
        executor = FakeExecutor(resource_governor_queue=object() if config.governor_queue else None)
        if config.role is ServerRole.VISUAL_GEN:
            # Lazy import: only VisualGen tests pay the visual_gen import cost,
            # and importing it is what lets _is_visual_gen_instance say yes.
            from test_common.serve.fakes_visual_gen import FakeVisualGen

            generator = FakeVisualGen(model=config.model_name)
        elif config.role is ServerRole.MM_ENCODER:
            generator = FakeMultimodalEncoder(args=args, executor=executor)
        elif config.role is ServerRole.EMBEDDING:
            generator = FakeEncodeOnlyLLM(args=args)
        elif config.rl_control:
            generator = FakeAsyncLLM(args=args, executor=executor)
        else:
            generator = FakeLLM(args=args, executor=executor)

        multi_frontend_serving = None
        if config.multi_frontend_attached:
            from tensorrt_llm.serve.multi_frontend import MultiFrontendServing

            ipc_dir = self.tmp_dir / "multi_frontend_ipc"
            ipc_dir.mkdir(parents=True, exist_ok=True)
            multi_frontend_serving = MultiFrontendServing(
                launcher_uds=str(ipc_dir / "launcher.sock"), is_launcher=False
            )

        server = OpenAIServer(
            generator=generator,
            model=config.model_name,
            tool_parser=config.tool_parser,
            server_role=config.role,
            metadata_server_cfg=None,
            enable_rl_control_endpoints=config.rl_control,
            rl_control_api_key="test-key" if config.rl_control else None,
            enable_runtime_control_endpoints=config.runtime_control,
            runtime_control_api_key="test-key" if config.runtime_control else None,
            multi_frontend_serving=multi_frontend_serving,
        )
        self._servers.append(server)
        return server

    def build_disagg_server(self, config: DisaggConfig) -> OpenAIDisaggServer:
        # At most one per environment: each registers the disaggregated server
        # counters in the default Prometheus registry, and a second one fails
        # with "Duplicated timeseries".
        server_config = _make_disagg_server_config(
            return_perf_metrics=config.return_perf_metrics,
            http_cluster_storage=config.http_cluster_storage,
        )
        return OpenAIDisaggServer(
            server_config,
            coordinator_url="http://127.0.0.1:9" if config.fleet else None,
        )

    def build_coordinator_server(self, config: CoordinatorConfig) -> CoordinatorServer:
        server_config = _make_disagg_server_config(http_cluster_storage=config.http_cluster_storage)
        coordinator = DisaggCoordinatorService(server_config, lambda *args, **kwargs: None)
        return CoordinatorServer(coordinator)

    # -- request plumbing -------------------------------------------------

    def client(self, app: FastAPI, *, lifespan: bool = True) -> TestClient:
        """A TestClient closed at environment exit; runs the lifespan by default."""
        client = TestClient(app)
        if lifespan:
            client.__enter__()
            self._entered_clients.append(client)
        else:
            self._plain_clients.append(client)
        return client

    def record_terminations(self) -> list:
        """Route termination observations of this process into the returned list.

        Installs a stub telemetry session; the environment's exit restores the
        session that existed before entry.
        """
        sink: list = []
        usage_lib._SESSION = _StubTelemetrySession(sink)
        return sink

    # -- teardown helpers (called by isolated_serve_env) -------------------

    def _close(self) -> None:
        for client in self._entered_clients:
            client.__exit__(None, None, None)
        for client in self._plain_clients:
            client.close()
        self._entered_clients.clear()
        self._plain_clients.clear()
        for server in self._servers:
            for pool_name in (
                "_input_proc_executor",
                "_media_load_executor",
                "_metrics_stats_executor",
            ):
                pool = getattr(server, pool_name, None)
                if pool is not None:
                    pool.shutdown(wait=False, cancel_futures=True)
        self._servers.clear()


@contextmanager
def isolated_serve_env(tmp_dir: Path) -> Iterator[ServeTestEnv]:
    """Process-global state that building trtllm-serve apps touches, restored on exit.

    The only supported forms are ``with isolated_serve_env(tmp_path) as env:``
    and the ``serve_env`` fixture. The context-manager object has no builder
    methods, so ``isolated_serve_env(tmp_path).build_openai_server(...)`` fails
    with ``AttributeError`` instead of leaving global state behind.

    Build one configuration per environment when configurations differ in
    environment variables or perf metrics.
    """
    tmp_dir = Path(tmp_dir)
    env_patcher = mock.patch.dict(os.environ)
    env_patcher.start()
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        prometheus_dir = tmp_dir / "prometheus"
        prometheus_dir.mkdir(parents=True, exist_ok=True)
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(prometheus_dir)
        os.environ["TRTLLM_MEDIA_STORAGE_PATH"] = str(tmp_dir / "media")
        for name in _CLEARED_ENV_VARS:
            os.environ.pop(name, None)

        saved_media_executor = BaseMediaIO._executor
        # The module global exists only after the first
        # set_prometheus_multiproc_dir() call; save it with a sentinel.
        saved_prometheus_dir = getattr(tensorrt_llm._utils, "prometheus_multiproc_dir", _MISSING)
        # An unhealthy /health records a termination observation on the
        # process-wide telemetry session, which keeps only its first
        # observation; detach any session so tests cannot poison it.
        saved_session = usage_lib._SESSION
        usage_lib._SESSION = None
        collectors_before = set(REGISTRY._collector_to_names.keys())

        env = ServeTestEnv(tmp_dir)
        try:
            yield env
        finally:
            env._close()
            for collector in list(REGISTRY._collector_to_names.keys()):
                if collector not in collectors_before:
                    with contextlib.suppress(KeyError):
                        REGISTRY.unregister(collector)
            BaseMediaIO._executor = saved_media_executor
            if saved_prometheus_dir is _MISSING:
                if hasattr(tensorrt_llm._utils, "prometheus_multiproc_dir"):
                    delattr(tensorrt_llm._utils, "prometheus_multiproc_dir")
            else:
                tensorrt_llm._utils.prometheus_multiproc_dir = saved_prometheus_dir
            usage_lib._SESSION = saved_session
    finally:
        env_patcher.stop()
