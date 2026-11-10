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
"""Environment variables read by trtllm-serve.

Every variable that ``tensorrt_llm/serve/`` (except ``scripts/``) and the
trtllm-serve CLI (``tensorrt_llm/commands/serve.py``,
``tensorrt_llm/commands/serve_cli/``) read is declared here once, with its
type, its default and what it does. Read a variable as ``envs.NAME`` after
``from tensorrt_llm.serve import envs``. A lint
(``tests/unittest/serve/test_envs.py``) rejects direct reads of ``os.environ``
in those directories and ``from tensorrt_llm.serve.envs import NAME``, which
would freeze the value at import time.

Values are read from ``os.environ`` on every access and never cached:
launchers write some of these variables at run time
(``TRTLLM_RESPONSES_API_DISABLE_STORE``, ``TRTLLM_DISAGG_ROLE``, the
fleet-worker variables) and code reads them later. Each reader reproduces the
parsing of the call site that read the variable before it was declared here
(``tests/unittest/serve/test_envs.py`` pins it); changing a reader or a default
is a behavior change. Where unset and empty differ today, they still differ.
A required variable raises ``KeyError`` when it is unset, also through
``hasattr(envs, NAME)`` and ``inspect.getmembers(envs)``; code that enumerates
the variables uses ``environment_variables`` instead of attribute access.

Three blocks, in this order: deployment switches (set by whoever runs the
server), launcher-internal variables (written by a launcher for a process it
starts, or for itself) and external variables (set by other software, such as
MPI or Slurm). ``also_read_by`` names the code outside the linted directories
that still reads ``os.environ`` for the same variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

if TYPE_CHECKING:
    # The type each ``envs.NAME`` read returns, one line per declared variable.
    # Deployment switches.
    TRTLLM_SERVER_DISABLE_GC: bool
    TRTLLM_DISAGG_SERVER_DISABLE_GC: bool
    TRTLLM_RESPONSES_API_DISABLE_STORE: bool
    TRTLLM_RUNTIME_CONTROL_API_KEY: Optional[str]
    TRTLLM_MEDIA_STORAGE_PATH: Optional[str]
    DISABLE_HARMONY_ADAPTER: bool
    TRTLLM_KVCACHE_TIME_OUTPUT_PATH: str
    TRTLLM_KIMI_PARAM_POLICY: bool
    TRTLLM_KIMI_K3_STRICT_TOOL_GRAMMAR: bool
    TRTLLM_DISAGG_NO_RETRY: bool
    TRTLLM_DISAGG_BENCHMARK_GEN_ONLY: bool
    TRTLLM_DISAGG_COORDINATOR_RESERVATION_TIMEOUT: float
    TRTLLM_DISAGG_COORDINATOR_PORT: Optional[int]
    TRTLLM_DISAGG_COORDINATOR_UDS: bool
    TLLM_FRONTEND_READY_TIMEOUT: float
    TRTLLM_ANTHROPIC_BATCH_CONCURRENCY: Optional[str]
    TRTLLM_ANTHROPIC_BATCH_MAX_RETAINED: Optional[str]
    TRTLLM_KVCACHE_AWARE_ROUTER_HASH_TOKENS_PER_BLOCK: Optional[int]
    TRTLLM_RESPONSES_DEBUG: bool
    TRTLLM_ENABLE_RENDER_ENDPOINTS: bool
    TRTLLM_RENDER_IGNORED_FIELDS: tuple[str, ...]
    TRTLLM_DISALLOW_LOCAL_MEDIA_PATH: str
    TRTLLM_WEB_SEARCH: Optional[str]
    TRTLLM_WEB_SEARCH_MAX_RESULTS: Optional[str]
    TRTLLM_WEB_SEARCH_TIMEOUT_S: Optional[str]
    TRTLLM_WEB_SEARCH_MAX_USES: Optional[str]
    TRTLLM_WEB_SEARCH_RETRIES: Optional[str]
    TRTLLM_WEB_SEARCH_UNAVAILABLE: Optional[str]
    BRAVE_SEARCH_API_KEY: Optional[str]
    TAVILY_API_KEY: Optional[str]
    SEARXNG_URL: Optional[str]
    TRTLLM_TRUST_KV_HINTS: bool
    TRTLLM_DISAGG_SHARED_DRAFT_KV: Optional[str]
    PROMETHEUS_MULTIPROC_DIR: Optional[str]
    # Launcher-internal variables.
    TRTLLM_DISAGG_CONFIG_FILE: str
    TRTLLM_DISAGG_COORDINATOR_URL: str
    TRTLLM_DISAGG_METADATA_CONFIG_FILE: Optional[str]
    TRTLLM_DISAGG_REQUEST_TIMEOUT: int
    TRTLLM_DISAGG_SERVER_START_TIMEOUT: int
    TRTLLM_DISAGG_SCHEDULE_STYLE: Optional[str]
    TRTLLM_DISAGG_LOG_LEVEL: Optional[str]
    TRTLLM_DISAGG_WORKER_PROCESS_ID: str
    TLLM_DISAGG_INSTANCE_IDX: Optional[str]
    TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT: Optional[str]
    TLLM_EXECUTOR_ATTACH_INFO: bool
    TLLM_EXECUTOR_FRONTEND_ID: int
    TLLM_FRONTEND_READY_FD: Optional[str]
    TLLM_FRONTEND_AFFINITY_FD: Optional[str]
    TLLM_SERVE_LAUNCHER_UDS: Optional[str]
    TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR: str
    TRTLLM_DISAGG_ROLE: Optional[str]
    TRTLLM_DISAGG_DEPLOYMENT_ID: Optional[str]
    # External variables.
    OMPI_COMM_WORLD_RANK: Optional[str]
    SLURM_PROCID: Optional[str]

__all__ = ["EnvVar", "consume", "environment_variables"]

Kind = Literal["deploy", "internal", "external"]


@dataclass(frozen=True)
class EnvVar:
    """One declared environment variable."""

    name: str
    kind: Kind
    read: Callable[[], Any]  # reproduces the call site's parsing; reads os.environ on each call
    default: str  # what an unset variable means, in words or as the literal default
    description: str
    secret: bool = False  # never logged and never echoed by any listing
    also_read_by: tuple[str, ...] = ()  # files outside the linted directories reading os.environ


def _flag(name: str, unset: Optional[str]) -> Callable[[], bool]:
    """True when the variable equals "1"; ``unset`` stands in only when the variable is absent."""
    return lambda: os.getenv(name, unset) == "1"


def _raw(name: str, unset: Optional[str] = None) -> Callable[[], Optional[str]]:
    """The value as written, or ``unset`` when the variable is absent."""
    return lambda: os.getenv(name, unset)


def _required(name: str) -> Callable[[], str]:
    """The value; ``KeyError`` when the variable is absent, as ``os.environ[name]`` raises."""
    return lambda: os.environ[name]


def _int(name: str, unset: str) -> Callable[[], int]:
    """``int(value)``; ``ValueError`` on a value that is not an integer, the empty string included."""
    return lambda: int(os.environ.get(name, unset))


def _optional_int(name: str) -> Callable[[], Optional[int]]:
    """``None`` when absent, else ``int(value)`` (``ValueError`` on a non-integer value)."""

    def read() -> Optional[int]:
        value = os.environ.get(name)
        return None if value is None else int(value)

    return read


def _names(name: str) -> Callable[[], tuple[str, ...]]:
    """Comma-separated names, stripped, empty ones dropped."""
    return lambda: tuple(part.strip() for part in os.getenv(name, "").split(",") if part.strip())


_DEPLOY: tuple[EnvVar, ...] = (
    EnvVar(
        "TRTLLM_SERVER_DISABLE_GC",
        "deploy",
        _flag("TRTLLM_SERVER_DISABLE_GC", "0"),
        default="0",
        description=(
            "Set to 1 to disable the Python garbage collector in the trtllm-serve process once "
            "the server is up."
        ),
        also_read_by=("tensorrt_llm/grpc/openengine/server.py",),
    ),
    EnvVar(
        "TRTLLM_DISAGG_SERVER_DISABLE_GC",
        "deploy",
        _flag("TRTLLM_DISAGG_SERVER_DISABLE_GC", "1"),
        default="1 when unset; an empty value counts as not 1 and keeps the collector on",
        description=(
            "Whether the disaggregated server and its fleet workers run with the Python garbage "
            "collector disabled. Only the exact value 1 disables it."
        ),
    ),
    EnvVar(
        "TRTLLM_RESPONSES_API_DISABLE_STORE",
        "deploy",
        lambda: len(os.getenv("TRTLLM_RESPONSES_API_DISABLE_STORE", "")) > 0,
        default="unset: the in-memory Responses API store is enabled",
        description=(
            "Any non-empty value, 0 included, disables the in-memory Responses API store. The "
            "launcher sets it to 1 when more than one frontend serves the same engine, before "
            "the server reads it."
        ),
    ),
    EnvVar(
        "TRTLLM_RUNTIME_CONTROL_API_KEY",
        "deploy",
        _raw("TRTLLM_RUNTIME_CONTROL_API_KEY"),
        default="unset",
        description=(
            "API key the runtime-control (sleep and wake-up) endpoints require; mandatory with "
            "--enable_sleep_mode."
        ),
        secret=True,
    ),
    EnvVar(
        "TRTLLM_MEDIA_STORAGE_PATH",
        "deploy",
        _raw("TRTLLM_MEDIA_STORAGE_PATH"),
        default="unset: the server creates a private media directory",
        description="Directory where generated media is stored; an empty value counts as unset.",
    ),
    EnvVar(
        "DISABLE_HARMONY_ADAPTER",
        "deploy",
        _flag("DISABLE_HARMONY_ADAPTER", "0"),
        default="0",
        description="Set to 1 to serve gpt-oss models without the Harmony adapter.",
    ),
    EnvVar(
        "TRTLLM_KVCACHE_TIME_OUTPUT_PATH",
        "deploy",
        _raw("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", ""),
        default="empty",
        description=(
            "Output path of the KV-cache transfer timing records; when non-empty, requests also "
            "return engine performance metrics."
        ),
        also_read_by=(
            "tensorrt_llm/_torch/disaggregation/native/perf_logger.py",
            "tensorrt_llm/_torch/disaggregation/transceiver.py",
            "cpp/tensorrt_llm/common/envUtils.cpp",
        ),
    ),
    EnvVar(
        "TRTLLM_KIMI_PARAM_POLICY",
        "deploy",
        _flag("TRTLLM_KIMI_PARAM_POLICY", "0"),
        default="0",
        description="Set to 1 to apply the Kimi K3 sampling-parameter policy to chat requests.",
    ),
    EnvVar(
        "TRTLLM_KIMI_K3_STRICT_TOOL_GRAMMAR",
        "deploy",
        _flag("TRTLLM_KIMI_K3_STRICT_TOOL_GRAMMAR", "0"),
        default="0",
        description=(
            "Experimental: set to 1 to constrain Kimi K3 tool calls with the strict "
            "structural-tag grammar."
        ),
    ),
    EnvVar(
        "TRTLLM_DISAGG_NO_RETRY",
        "deploy",
        _flag("TRTLLM_DISAGG_NO_RETRY", "0"),
        default="0",
        description=(
            "Set to 1 to stop the disaggregated client from retrying failed worker requests."
        ),
        also_read_by=(
            "tensorrt_llm/_torch/disaggregation/native/context_first_profile.py",
            "tensorrt_llm/_torch/disaggregation/transceiver.py",
        ),
    ),
    EnvVar(
        "TRTLLM_DISAGG_BENCHMARK_GEN_ONLY",
        "deploy",
        _flag("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", None),
        default="unset",
        description=(
            "Benchmark switch: set to 1 to skip the context phase and serve generation-only "
            "disaggregated requests."
        ),
        also_read_by=(
            "tensorrt_llm/_torch/disaggregation/orchestration/coordinator.py",
            "tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py",
            "cpp/tensorrt_llm/common/envUtils.cpp",
        ),
    ),
    EnvVar(
        "TRTLLM_DISAGG_COORDINATOR_RESERVATION_TIMEOUT",
        "deploy",
        lambda: float(os.environ.get("TRTLLM_DISAGG_COORDINATOR_RESERVATION_TIMEOUT", 180.0)),
        default="180.0",
        description=(
            "Seconds a disaggregated coordinator holds a worker reservation before it expires; "
            "a value that is not a number raises ValueError."
        ),
    ),
    EnvVar(
        "TRTLLM_DISAGG_COORDINATOR_PORT",
        "deploy",
        _optional_int("TRTLLM_DISAGG_COORDINATOR_PORT"),
        default="unset: the launcher uses the public port minus one",
        description="TCP port of the implicit disaggregated coordinator.",
    ),
    EnvVar(
        "TRTLLM_DISAGG_COORDINATOR_UDS",
        "deploy",
        _flag("TRTLLM_DISAGG_COORDINATOR_UDS", "1"),
        default="1",
        description=(
            "Whether the co-located fleet reaches the coordinator over a Unix domain socket (1) "
            "or TCP (any other value)."
        ),
    ),
    EnvVar(
        "TLLM_FRONTEND_READY_TIMEOUT",
        "deploy",
        lambda: float(os.getenv("TLLM_FRONTEND_READY_TIMEOUT", "300")),
        default="300",
        description="Seconds the launcher waits for every attached frontend to report ready.",
    ),
    EnvVar(
        "TRTLLM_ANTHROPIC_BATCH_CONCURRENCY",
        "deploy",
        _raw("TRTLLM_ANTHROPIC_BATCH_CONCURRENCY"),
        default="unset: 4 requests",
        description=(
            "How many Anthropic Message Batches requests run at once; the batch runner parses "
            "it and keeps its default on an invalid value."
        ),
    ),
    EnvVar(
        "TRTLLM_ANTHROPIC_BATCH_MAX_RETAINED",
        "deploy",
        _raw("TRTLLM_ANTHROPIC_BATCH_MAX_RETAINED"),
        default="unset: 100 batches",
        description=(
            "How many finished Anthropic Message Batches the server keeps in memory; the batch "
            "runner parses it and keeps its default on an invalid value."
        ),
    ),
    EnvVar(
        "TRTLLM_KVCACHE_AWARE_ROUTER_HASH_TOKENS_PER_BLOCK",
        "deploy",
        _optional_int("TRTLLM_KVCACHE_AWARE_ROUTER_HASH_TOKENS_PER_BLOCK"),
        default="unset: the router's own block size",
        description=(
            "Tokens per block for the KV-cache-aware router's block hashing; a value that is "
            "not an integer raises ValueError."
        ),
    ),
    EnvVar(
        "TRTLLM_RESPONSES_DEBUG",
        "deploy",
        _flag("TRTLLM_RESPONSES_DEBUG", None),
        default="unset",
        description=(
            "Set to 1 to log Responses API debugging messages; read once when the Responses "
            "module is imported."
        ),
    ),
    EnvVar(
        "TRTLLM_ENABLE_RENDER_ENDPOINTS",
        "deploy",
        _flag("TRTLLM_ENABLE_RENDER_ENDPOINTS", "0"),
        default="0",
        description="Set to 1 to mount the prompt-rendering endpoints on a serving worker.",
    ),
    EnvVar(
        "TRTLLM_RENDER_IGNORED_FIELDS",
        "deploy",
        _names("TRTLLM_RENDER_IGNORED_FIELDS"),
        default="empty: no extra fields",
        description=(
            "Comma-separated request fields the render endpoints drop before rendering, in "
            "addition to the built-in list."
        ),
    ),
    EnvVar(
        "TRTLLM_DISALLOW_LOCAL_MEDIA_PATH",
        "deploy",
        _raw("TRTLLM_DISALLOW_LOCAL_MEDIA_PATH", "0"),
        default="0",
        description=(
            "Set to 1 to refuse local file paths as VisualGen media over HTTP; the caller warns "
            "on any value other than 0 or 1."
        ),
    ),
    EnvVar(
        "TRTLLM_WEB_SEARCH",
        "deploy",
        _raw("TRTLLM_WEB_SEARCH"),
        default="unset: off",
        description=(
            "Web-search provider behind the Responses API web_search tool (brave, tavily, "
            "searxng or off); read per request."
        ),
    ),
    EnvVar(
        "TRTLLM_WEB_SEARCH_MAX_RESULTS",
        "deploy",
        _raw("TRTLLM_WEB_SEARCH_MAX_RESULTS"),
        default="unset: the web-search module's default",
        description=(
            "Maximum results per web search; the web-search module parses it and keeps its "
            "default on a non-integer value."
        ),
    ),
    EnvVar(
        "TRTLLM_WEB_SEARCH_TIMEOUT_S",
        "deploy",
        _raw("TRTLLM_WEB_SEARCH_TIMEOUT_S"),
        default="unset: the web-search module's default",
        description=(
            "Web-search request timeout in seconds; the web-search module parses it and keeps "
            "its default on a non-numeric value."
        ),
    ),
    EnvVar(
        "TRTLLM_WEB_SEARCH_MAX_USES",
        "deploy",
        _raw("TRTLLM_WEB_SEARCH_MAX_USES"),
        default="unset: the web-search module's default",
        description="Maximum web searches per response; parsed by the web-search module.",
    ),
    EnvVar(
        "TRTLLM_WEB_SEARCH_RETRIES",
        "deploy",
        _raw("TRTLLM_WEB_SEARCH_RETRIES"),
        default="unset: the web-search module's default",
        description="Retries per web search; parsed by the web-search module.",
    ),
    EnvVar(
        "TRTLLM_WEB_SEARCH_UNAVAILABLE",
        "deploy",
        _raw("TRTLLM_WEB_SEARCH_UNAVAILABLE"),
        default="unset: ignore",
        description=(
            "What the Responses endpoint does with a web_search tool it cannot honour: ignore "
            "or reject; the caller normalizes the value and warns on an unknown one."
        ),
    ),
    EnvVar(
        "BRAVE_SEARCH_API_KEY",
        "deploy",
        _raw("BRAVE_SEARCH_API_KEY"),
        default="unset",
        description="API key of the brave web-search provider.",
        secret=True,
    ),
    EnvVar(
        "TAVILY_API_KEY",
        "deploy",
        _raw("TAVILY_API_KEY"),
        default="unset",
        description="API key of the tavily web-search provider.",
        secret=True,
    ),
    EnvVar(
        "SEARXNG_URL",
        "deploy",
        _raw("SEARXNG_URL"),
        default="unset",
        description="Endpoint of the searxng web-search provider.",
    ),
    EnvVar(
        "TRTLLM_TRUST_KV_HINTS",
        "deploy",
        _flag("TRTLLM_TRUST_KV_HINTS", "0"),
        default="0",
        description="Set to 1 to trust the KV-cache hints clients send with their requests.",
    ),
    EnvVar(
        "TRTLLM_DISAGG_SHARED_DRAFT_KV",
        "deploy",
        _raw("TRTLLM_DISAGG_SHARED_DRAFT_KV"),
        default="unset: auto",
        description=(
            "Whether disaggregated workers share the drafter's KV cache: 1, 0 or auto. The MPI "
            "worker resolves auto and rejects 1 when the model configuration cannot share it."
        ),
        also_read_by=("tensorrt_llm/_torch/pyexecutor/_util.py",),
    ),
    EnvVar(
        "PROMETHEUS_MULTIPROC_DIR",
        "deploy",
        _raw("PROMETHEUS_MULTIPROC_DIR"),
        default="unset: the launcher creates one when performance metrics are on",
        description=(
            "prometheus_client's multiprocess metrics directory; honoured when already set, "
            "otherwise created by the launcher and shared by every frontend it starts."
        ),
        also_read_by=("tensorrt_llm/_utils.py", "tensorrt_llm/grpc/openengine/metrics.py"),
    ),
)

_INTERNAL: tuple[EnvVar, ...] = (
    EnvVar(
        "TRTLLM_DISAGG_CONFIG_FILE",
        "internal",
        _required("TRTLLM_DISAGG_CONFIG_FILE"),
        default="required: KeyError when missing",
        description="Path of the disaggregated config file, passed by the launcher to each fleet worker.",
    ),
    EnvVar(
        "TRTLLM_DISAGG_COORDINATOR_URL",
        "internal",
        _required("TRTLLM_DISAGG_COORDINATOR_URL"),
        default="required: KeyError when missing",
        description="URL of the coordinator, passed by the launcher to each fleet worker.",
    ),
    EnvVar(
        "TRTLLM_DISAGG_METADATA_CONFIG_FILE",
        "internal",
        _raw("TRTLLM_DISAGG_METADATA_CONFIG_FILE"),
        default="unset",
        description=(
            "Path of the metadata-server config file, passed by the launcher to each fleet "
            "worker when one was given."
        ),
    ),
    EnvVar(
        "TRTLLM_DISAGG_REQUEST_TIMEOUT",
        "internal",
        _int("TRTLLM_DISAGG_REQUEST_TIMEOUT", "180"),
        default="180",
        description="Request timeout in seconds of a fleet worker's disaggregated server.",
    ),
    EnvVar(
        "TRTLLM_DISAGG_SERVER_START_TIMEOUT",
        "internal",
        _int("TRTLLM_DISAGG_SERVER_START_TIMEOUT", "180"),
        default="180",
        description=(
            "Seconds a fleet worker waits for its context and generation servers to come up."
        ),
    ),
    EnvVar(
        "TRTLLM_DISAGG_SCHEDULE_STYLE",
        "internal",
        _raw("TRTLLM_DISAGG_SCHEDULE_STYLE"),
        default="unset",
        description="Scheduling-style override passed by the launcher to each fleet worker.",
    ),
    EnvVar(
        "TRTLLM_DISAGG_LOG_LEVEL",
        "internal",
        _raw("TRTLLM_DISAGG_LOG_LEVEL"),
        default="unset",
        description="The launcher's logger level, restored in each fleet worker process.",
    ),
    EnvVar(
        "TRTLLM_DISAGG_WORKER_PROCESS_ID",
        "internal",
        _raw("TRTLLM_DISAGG_WORKER_PROCESS_ID", "0"),
        default="0",
        description="Index of a fleet worker process (0 to N-1), set per worker by the launcher.",
        also_read_by=("tensorrt_llm/llmapi/disagg_utils.py",),
    ),
    EnvVar(
        "TLLM_DISAGG_INSTANCE_IDX",
        "internal",
        _raw("TLLM_DISAGG_INSTANCE_IDX"),
        default="unset",
        description=(
            "Index of the server configuration an MPI worker serves, set by the MPI leader "
            "before it relaunches the worker."
        ),
    ),
    EnvVar(
        "TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT",
        "internal",
        _raw("TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT"),
        default="unset",
        description="1 when the relaunched MPI worker serves through a remote MPI session client.",
    ),
    EnvVar(
        "TLLM_EXECUTOR_ATTACH_INFO",
        "internal",
        lambda: os.getenv("TLLM_EXECUTOR_ATTACH_INFO") is not None,
        default="unset: this process is the launcher, not an attached frontend",
        description=(
            "Present in every attached frontend the launcher starts. It carries the executor's "
            "attach information, HMAC keys included, which only the executor reads; serving "
            "code reads just whether it is set."
        ),
        secret=True,
        also_read_by=("tensorrt_llm/executor/executor.py", "tensorrt_llm/llmapi/llm.py"),
    ),
    EnvVar(
        "TLLM_EXECUTOR_FRONTEND_ID",
        "internal",
        lambda: int(os.environ["TLLM_EXECUTOR_FRONTEND_ID"]),
        default="required: KeyError when missing",
        description="Index of an attached frontend among the frontends the launcher starts.",
        also_read_by=(
            "tensorrt_llm/executor/executor.py",
            "tensorrt_llm/grpc/openengine/server.py",
        ),
    ),
    EnvVar(
        "TLLM_FRONTEND_READY_FD",
        "internal",
        _raw("TLLM_FRONTEND_READY_FD"),
        default="unset",
        description=(
            "File descriptor an attached frontend writes its READY byte to; read once with "
            "consume() and removed."
        ),
    ),
    EnvVar(
        "TLLM_FRONTEND_AFFINITY_FD",
        "internal",
        _raw("TLLM_FRONTEND_AFFINITY_FD"),
        default="unset",
        description=(
            "File descriptor of the listening socket the launcher bound for an attached "
            "frontend; read once with consume() and removed."
        ),
    ),
    EnvVar(
        "TLLM_SERVE_LAUNCHER_UDS",
        "internal",
        _raw("TLLM_SERVE_LAUNCHER_UDS"),
        default="unset",
        description=(
            "Unix domain socket of the launcher, passed to each attached frontend; read once "
            "with consume() and removed."
        ),
    ),
    EnvVar(
        "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR",
        "internal",
        _required("TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR"),
        default="required: KeyError when missing",
        description="IPC address of the LLM API proxy process, set by the disaggregated MPI leader.",
        also_read_by=("tensorrt_llm/executor/utils.py", "tensorrt_llm/llmapi/_llmapi_mpi_stop.py"),
    ),
    EnvVar(
        "TRTLLM_DISAGG_ROLE",
        "internal",
        _raw("TRTLLM_DISAGG_ROLE"),
        default="unset",
        description=(
            "Role of this process in a disaggregated deployment (context, generation, "
            "coordinator or server_coordinator), set by the launcher for model code and usage "
            "telemetry."
        ),
        also_read_by=(
            "tensorrt_llm/_torch/configs/deepseek_v41.py",
            "tensorrt_llm/_torch/models/modeling_deepseekv41.py",
            "tensorrt_llm/_torch/modules/kimi_kda/cache_manager.py",
            "tensorrt_llm/usage/usage_lib.py",
        ),
    ),
    EnvVar(
        "TRTLLM_DISAGG_DEPLOYMENT_ID",
        "internal",
        _raw("TRTLLM_DISAGG_DEPLOYMENT_ID"),
        default="unset",
        description=(
            "Identifier shared by every worker of one disaggregated deployment, set by the "
            "launcher for usage telemetry; serving code only writes it."
        ),
        also_read_by=("tensorrt_llm/usage/usage_lib.py",),
    ),
)

_EXTERNAL: tuple[EnvVar, ...] = (
    EnvVar(
        "OMPI_COMM_WORLD_RANK",
        "external",
        _raw("OMPI_COMM_WORLD_RANK"),
        default="unset",
        description=(
            "Global rank of this process, set by Open MPI; the disaggregated MPI worker picks "
            "its CUDA device from it."
        ),
        also_read_by=("cpp/tensorrt_llm/common/logger.cpp",),
    ),
    EnvVar(
        "SLURM_PROCID",
        "external",
        _raw("SLURM_PROCID"),
        default="unset",
        description=(
            "Global rank of this process, set by Slurm; the disaggregated MPI worker uses it "
            "when Open MPI's rank is absent."
        ),
        also_read_by=("tensorrt_llm/_torch/visual_gen/executor.py",),
    ),
)

environment_variables: dict[str, EnvVar] = {
    var.name: var for var in (*_DEPLOY, *_INTERNAL, *_EXTERNAL)
}


def __getattr__(name: str) -> Any:
    var = environment_variables.get(name)
    if var is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}. Declare the environment "
            "variable in tensorrt_llm/serve/envs.py before reading it."
        )
    # Outside any try: a required variable's KeyError propagates as KeyError.
    return var.read()


def __dir__() -> list[str]:
    return sorted([*globals(), *environment_variables])


def consume(name: str) -> Optional[str]:
    """Read an internal variable meant for exactly one reader and remove it from the environment."""
    var = environment_variables.get(name)
    if var is None or var.kind != "internal":
        raise KeyError(
            f"{name} is not a declared internal variable; consume() removes a variable a launcher "
            "passed to this process. Declare it in the internal block of tensorrt_llm/serve/envs.py."
        )
    return os.environ.pop(name, None)
