# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import contextlib
import gc
import importlib
import inspect
import json
import os
import signal
import socket
import sys
import tempfile
import warnings
from importlib.util import find_spec
from pathlib import Path
from types import FrameType
from typing import (TYPE_CHECKING, Any, Dict, Iterator, NoReturn, Optional,
                    Sequence, Set, Tuple)

import click
import pydantic
import uvloop
import yaml
from torch.cuda import device_count

from tensorrt_llm import LLM as PyTorchLLM
from tensorrt_llm import MultimodalEncoder
from tensorrt_llm._errors import ConfigInputError
from tensorrt_llm.commands import _telemetry as _command_telemetry
from tensorrt_llm.commands._config_overrides import (ConfigOverride,
                                                     ConfigOverrideError,
                                                     apply_config_overrides,
                                                     parse_config_overrides)
from tensorrt_llm.commands._serve_stability import stability_option
from tensorrt_llm.commands.mooncake import mooncake_master, mooncake_pool_report
from tensorrt_llm.commands.serve_cli._disagg_fleet import DisaggLauncherEnvs
from tensorrt_llm.commands.serve_cli._net import (_bind_listener,
                                                  _diagnose_port_in_use,
                                                  _listener_address_family,
                                                  _publish_bound_address)
from tensorrt_llm.commands.serve_cli._telemetry import \
    _report_observed_child_failure
from tensorrt_llm.commands.serve_cli.disaggregated import (
    disaggregated, disaggregated_mpi_worker)
from tensorrt_llm.commands.utils import (collect_explicit_cli_keys,
                                         get_is_diffusion_only_model)
from tensorrt_llm.executor.utils import MAX_NUM_FRONTENDS
from tensorrt_llm.inputs.multimodal import MultimodalServerConfig
from tensorrt_llm.llmapi.disagg_utils import (
    DisaggClusterConfig, DisaggServerConfig, MetadataServerConfig, ServerRole,
    extract_disagg_cluster_config, parse_metadata_server_config_file,
    validate_config_bool, validate_internal_request_auth_key)
from tensorrt_llm.llmapi.llm_args import (SERVE_CLI_CONFIG_FIELDS,
                                          KvCacheConnectorConfig, SleepConfig,
                                          TorchLlmArgs, build_cli_config)
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_dict
from tensorrt_llm.llmapi.reasoning_parser import (ReasoningParserFactory,
                                                  resolve_auto_reasoning_parser)
from tensorrt_llm.llmapi.startup_checks import no_visible_gpu_error
from tensorrt_llm.logger import logger, severity_map
from tensorrt_llm.mapping import CpType
from tensorrt_llm.serve import OpenAIServer
from tensorrt_llm.serve._frontend_processes import (
    _frontend_affinity_sockets, _init_multi_frontend_mode,
    _signal_frontend_ready, _spawn_attached_frontends,
    _terminate_attached_frontends, _watch_attached_frontends)
from tensorrt_llm.serve.multi_frontend import (LAUNCHER_UDS_ENV,
                                               LAUNCHER_UDS_NAME,
                                               MultiFrontendServing)
from tensorrt_llm.serve.tool_parser import ToolParserFactory
from tensorrt_llm.serve.tool_parser.tool_parser_factory import (
    MODEL_TYPE_TO_TOOL_PARSER, resolve_auto_tool_parser)
from tensorrt_llm.tools.importlib_utils import import_custom_module_from_dir
from tensorrt_llm.usage import apply_usage_session_config
from tensorrt_llm.usage import config as _telemetry_config
from tensorrt_llm.usage.usage_lib import _mark_visual_gen

if TYPE_CHECKING:
    # Type-only: the visual_gen tree is imported lazily inside the VisualGen
    # code paths so plain LLM serving never pays its import cost.
    from tensorrt_llm.visual_gen.args import VisualGenArgs


class _VisualGenStartupTermination(BaseException):

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def _terminate_visual_gen_startup(signum: int,
                                  frame: Optional[FrameType]) -> NoReturn:
    del frame
    raise _VisualGenStartupTermination(signum)


def _apply_effective_telemetry_config(llm_args: dict,
                                      *,
                                      telemetry: bool,
                                      component: str = "server") -> None:
    """Apply a parsed config opt-out before later setup can fail."""
    telemetry_config = llm_args.get("telemetry_config")
    if telemetry_config is not None:
        if not telemetry:
            telemetry_config = telemetry_config.model_copy(
                update={"disabled": True})
            llm_args["telemetry_config"] = telemetry_config
        apply_usage_session_config(
            telemetry_config,
            default_usage_context="cli_serve",
            component=component,
            lifecycle_phase="config_validation",
        )


def _pop_bool_config_option(config: dict[str, Any], key: str) -> bool:
    return validate_config_bool(config.pop(key, False), key)


def _apply_fastapi_middlewares(app, middlewares: Sequence[str]) -> None:
    """Import and register middleware objects on a FastAPI app."""
    for middleware in middlewares:
        try:
            module_path, object_name = middleware.rsplit(".", 1)
        except ValueError as e:
            raise ValueError(f"Invalid middleware import path '{middleware}'. "
                             "Expected format: <module>.<object>.") from e

        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             "Must be a class or an async function.")


def is_non_default_or_required(param_name, value, explicit_cli_keys):
    """
    Check if a parameter should be explicitly included in llm_args.

    Returns True if parameter is either:
    1. Always required (core params that must be present), OR
    2. Set explicitly on the CLI (its name or one of its constructor
       scalars is in `explicit_cli_keys`), OR
    3. Different from its default value in the backend's LlmArgs class
    """
    always_include = {
        "model", "backend", "tokenizer", "custom_tokenizer",
        "post_processor_hook", "postprocess_tokenizer_dir"
    }

    if param_name in always_include:
        return True

    if value is None:
        return False

    if param_name in explicit_cli_keys:
        return True

    # LlmArgs fields built from CLI scalars whose names differ from the field
    # name (e.g. `--free_gpu_memory_fraction` constructs `kv_cache_config`).
    cli_derived_fields = {
        **SERVE_CLI_CONFIG_FIELDS,
    }
    if any(s in explicit_cli_keys
           for s in cli_derived_fields.get(param_name, ())):
        return True

    llm_args_class = TorchLlmArgs

    field_info = llm_args_class.model_fields.get(param_name)
    if not field_info:
        return False

    default = field_info.default
    if callable(default):
        default = default()
    elif field_info.default_factory is not None:
        default = field_info.default_factory()

    return value != default


# CLI/API defaults are sourced from the TorchLlmArgs field defaults so they stay
# in lock-step with the args class and can't drift.
_LLM_ARGS_FIELDS = TorchLlmArgs.model_fields


def _parse_config_overrides(
        assignments: Sequence[str]) -> tuple[ConfigOverride, ...]:
    try:
        return parse_config_overrides(
            assignments,
            allowed_roots=_LLM_ARGS_FIELDS,
        )
    except ConfigOverrideError as error:
        raise click.BadParameter(str(error), param_hint="--set") from error


def _apply_config_overrides(
        config: Dict[str, Any],
        overrides: Sequence[ConfigOverride]) -> Dict[str, Any]:
    try:
        return apply_config_overrides(config, overrides)
    except ConfigOverrideError as error:
        raise click.BadParameter(str(error), param_hint="--set") from error


def get_llm_args(
        model: str,
        tokenizer: Optional[str] = None,
        custom_tokenizer: Optional[str] = None,
        post_processor_hook: Optional[str] = None,
        backend: str = "pytorch",
        generation_config: str = _LLM_ARGS_FIELDS["generation_config"].default,
        max_beam_width: int = _LLM_ARGS_FIELDS["max_beam_width"].default,
        max_batch_size: int = _LLM_ARGS_FIELDS["max_batch_size"].default,
        max_num_tokens: int = _LLM_ARGS_FIELDS["max_num_tokens"].default,
        max_seq_len: int = None,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        context_parallel_size: int = 1,
        cp_config: Optional[dict] = None,
        moe_expert_parallel_size: Optional[int] = None,
        gpus_per_node: Optional[int] = None,
        free_gpu_memory_fraction: float = 0.9,
        kv_cache_dtype: str = "auto",
        num_postprocess_workers: int = 0,
        num_serve_frontends: int = 1,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        reasoning_parser: Optional[str] = None,
        otlp_traces_endpoint: Optional[str] = None,
        enable_chunked_prefill: bool = False,
        enable_attention_dp: bool = False,
        video_pruning_rate: Optional[float] = None,
        telemetry: bool = True,
        agent_percentage: float = 0.0,
        agent_types: Optional[str] = None,
        explicit_cli_keys: Optional[Set[str]] = None,
        enable_locality_domains: bool = False,
        **llm_args_extra_dict: Any):

    explicit_cli_keys = explicit_cli_keys or set()

    if gpus_per_node is None:
        gpus_per_node = device_count()
        if gpus_per_node == 0:
            raise no_visible_gpu_error()

    # TODO: This manual cp_type conversion can be removed once cp_config
    # is refactored to a typed Pydantic model with enum coercion
    if cp_config is not None and "cp_type" in cp_config:
        cp_config = cp_config.copy()
        try:
            cp_config["cp_type"] = CpType[cp_config["cp_type"].upper()]
        except KeyError:
            raise ValueError(f"Invalid cp_type: {cp_config['cp_type']}. " \
                             f"Must be one of: {', '.join([t.name for t in CpType])}")

    cli_maybe_overrides = {
        "model":
        model,
        "backend":
        backend,
        "generation_config":
        generation_config,
        "tokenizer":
        tokenizer,
        "custom_tokenizer":
        custom_tokenizer,
        "post_processor_hook":
        post_processor_hook,
        "postprocess_tokenizer_dir":
        tokenizer or model,
        "kv_cache_config":
        build_cli_config("kv_cache_config",
                         free_gpu_memory_fraction=free_gpu_memory_fraction,
                         kv_cache_dtype=kv_cache_dtype),
        "cp_config":
        cp_config,
        "scheduler_config":
        None,
        "max_batch_size":
        max_batch_size,
        "max_beam_width":
        max_beam_width,
        "tensor_parallel_size":
        tensor_parallel_size,
        "pipeline_parallel_size":
        pipeline_parallel_size,
        "context_parallel_size":
        context_parallel_size,
        "moe_expert_parallel_size":
        moe_expert_parallel_size,
        "gpus_per_node":
        gpus_per_node,
        "trust_remote_code":
        trust_remote_code,
        "max_num_tokens":
        max_num_tokens,
        "max_seq_len":
        max_seq_len,
        "num_postprocess_workers":
        num_postprocess_workers,
        "num_serve_frontends":
        num_serve_frontends,
        "enable_chunked_prefill":
        enable_chunked_prefill,
        "enable_attention_dp":
        enable_attention_dp,
        "enable_locality_domains":
        enable_locality_domains,
        "revision":
        revision,
        "reasoning_parser":
        reasoning_parser,
        "otlp_traces_endpoint":
        otlp_traces_endpoint,
        "multimodal_config":
        build_cli_config("multimodal_config",
                         video_pruning_rate=video_pruning_rate)
        if video_pruning_rate is not None else None,
        "telemetry_config":
        _telemetry_config.TelemetryConfig(
            disabled=not telemetry,
            usage_context=_telemetry_config.UsageContext.CLI_SERVE),
        "agent_percentage":
        agent_percentage,
        "agent_types":
        agent_types,
    }

    llm_args = {
        param: value
        for param, value in cli_maybe_overrides.items()
        if is_non_default_or_required(param, value, explicit_cli_keys)
    }

    return llm_args, llm_args_extra_dict


def _build_llm_args_from_disagg_server_cfg(other_args: Dict) -> Dict:
    """Construct llm_args from a disaggregated server config's `other_args`.

    `other_args` is a single source — there is no separate CLI / YAML
    distinction here. Every key is user-set, so we pass all keys as
    `explicit_cli_keys` to `get_llm_args` to bypass the value-based filter
    that would otherwise drop fields equal to their LlmArgs class defaults
    (e.g. `tensor_parallel_size: 1`).

    Do NOT pass `explicit_cli_keys` to `update_llm_args_with_extra_dict`:
    `llm_args_extra_dict` here is just the catch-all for kwargs that didn't
    match `get_llm_args`'s named signature (e.g. `quant_config`,
    `lora_config`, `pytorch_backend_config`) — it isn't a separate YAML
    being overridden. Passing the explicit set would trigger the merge
    function's "drop YAML keys claimed by explicit CLI" filter and
    silently lose those kwargs.
    """
    disagg_explicit_keys = set(other_args)
    llm_args, llm_args_extra_dict = get_llm_args(
        **other_args, explicit_cli_keys=disagg_explicit_keys)
    return update_llm_args_with_extra_dict(llm_args, llm_args_extra_dict)


def _reserve_multi_frontend_port(host: str, port: int,
                                 family: socket.AddressFamily) -> socket.socket:
    """Refuse a port another server uses; return a lock to hold while serving.

    SO_REUSEPORT would let a second server of the same user join the port. A
    plain bind catches any listener (SO_REUSEADDR ignores TIME_WAIT), and an
    abstract Unix socket named after the port catches a launcher that has not
    started listening yet.
    """
    lock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        lock.bind(f"\0trtllm-serve-{os.geteuid()}-{port}")
    except OSError as e:
        lock.close()
        raise RuntimeError(f"Port {port} is already reserved by another "
                           f"multi-frontend trtllm-serve: {e}") from e
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, port))
    except OSError as e:
        lock.close()
        raise RuntimeError(
            f"Failed to bind socket to {host}:{port}: {e}. "
            f"Port holder(s): {_diagnose_port_in_use(port)}") from e
    return lock


@contextlib.contextmanager
def _provision_kv_cache_pool(llm_args: dict,
                             owns_engine: bool = True) -> Iterator[None]:
    """Bring up the shared cache this server joins, for its lifetime.

    A connector backed by a cluster-wide pool needs that pool reachable before
    any rank opens a handle, and the LLM constructor spawns the ranks, so this
    wraps the construction. A deployment that provisions the pool externally is
    detected and left alone.

    Only the process that owns the engine does this. An attached frontend
    re-execs this command line but shares the launcher's executor, so it would
    otherwise render a second client config over the first.
    """
    if not owns_engine:
        yield
        return

    from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import \
        maybe_provision_pool

    connector_config = llm_args.get("kv_connector_config")
    if isinstance(connector_config, dict):
        # A YAML section arrives unvalidated, and the pool has to be described
        # before the LLM constructor would coerce it. The validated model is
        # handed on so it is not parsed twice.
        connector_config = KvCacheConnectorConfig(**connector_config)
        llm_args["kv_connector_config"] = connector_config

    with maybe_provision_pool(connector_config):
        yield


def _bind_launcher_uds(llm) -> Tuple[socket.socket, str]:
    """Bind the Unix socket attached frontends forward launcher-owned routes to.

    It lives in the multi-frontend ipc dir, which is private to the user and
    removed, socket file included, when the executor proxy shuts down.
    """
    # Only the classic IPC proxy has the attribute.
    ipc_dir = getattr(llm._executor, "_multi_frontend_ipc_dir", None)
    if not ipc_dir:
        raise ValueError(
            "num_serve_frontends > 1 requires the classic IPC executor "
            "proxy in multi-frontend mode")
    path = os.path.join(ipc_dir, LAUNCHER_UDS_NAME)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(path)
    except OSError:
        sock.close()
        raise
    return sock, path


def _share_prometheus_multiproc_dir(llm, ipc_dir: str) -> None:
    """Give every frontend one PROMETHEUS_MULTIPROC_DIR before spawning them.

    Not via set_prometheus_multiproc_dir(): it keeps only its latest
    TemporaryDirectory alive, so OpenAIServer's later call would delete this one.
    """
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        return
    if llm.args.return_perf_metrics or llm.args.perf_metrics_output_dir:
        path = tempfile.mkdtemp(prefix="prometheus_", dir=ipc_dir)
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = path
        logger.info(
            f"Shared PROMETHEUS_MULTIPROC_DIR for all frontends: {path}")


def _resolve_default_num_serve_frontends(llm_args: dict, *, requested: bool,
                                         port: int, grpc: bool,
                                         report_addr: Optional[str],
                                         enable_sleep_mode: bool) -> None:
    """Fall back to one frontend where the default count cannot run.

    An explicit count (CLI, --config YAML or --set) is left alone, so the
    guard that owns the conflict still rejects it.
    """
    num_frontends = llm_args.get("num_serve_frontends", 1)
    if requested or num_frontends <= 1:
        return
    if grpc:
        # OpenEngine supports several frontends, but only when asked to.
        reason = "--grpc"
    elif port == 0 or report_addr:
        reason = "--port 0 / --report_addr"
    elif llm_args.get("orchestrator_type") is not None:
        reason = f"orchestrator_type={llm_args['orchestrator_type']!r}"
    elif llm_args.get("enable_resource_governor"):
        reason = "enable_resource_governor"
    elif enable_sleep_mode:
        reason = "--enable_sleep_mode"
    else:
        return
    logger.info(
        f"num_serve_frontends defaults to {num_frontends} for the OpenAI HTTP "
        f"server only; running 1 frontend because of {reason}. Pass "
        "--num_serve_frontends explicitly to override.")
    llm_args["num_serve_frontends"] = 1


def _bool_setting(value: object) -> bool:
    """A boolean LLM argument as TorchLlmArgs reads it; False if it does not parse.

    A value that does not parse is left to the LLM arguments' own validation.
    """
    try:
        return pydantic.TypeAdapter(bool).validate_python(value)
    except pydantic.ValidationError:
        return False


def _dense_dwdp_setting(dwdp_config: object) -> bool:
    """Whether a ``dwdp_config`` value of the LLM arguments enables dense DWDP.

    The value is a ``DwdpConfig`` once the YAML is merged, or a mapping; a
    value that does not parse is left to the LLM arguments' own validation.
    """
    if isinstance(dwdp_config, dict):
        return _bool_setting(dwdp_config.get("enable_dense"))
    return getattr(dwdp_config, "enable_dense", False) is True


#: The ``TRTLLM_DISAGG_ROLE`` of the servers of a disaggregated MPI launch, by
#: server type.
_DISAGG_SERVER_ROLES = {"ctx": "context", "gen": "generation"}

_DENSE_DWDP_FOR_CONTEXT_ONLY = (
    "dwdp_config.enable_dense is for context servers only: dense DWDP gathers "
    "the dense weights of every covered layer in every forward, decode steps "
    "included.")


def _serving_role(role: Optional[str]) -> Optional[str]:
    """``"context"`` or ``"generation"`` for a ``TRTLLM_DISAGG_ROLE`` value.

    Matched as the usage telemetry matches it: case-insensitive, with the
    compatible legacy values that start with ``ctx`` or ``gen`` (``ctx0``,
    ``gen0``). None for any other value.
    """
    normalized = (role or "").lower()
    if normalized == "context" or normalized.startswith("ctx"):
        return "context"
    if normalized.startswith("gen"):
        return "generation"
    return None


def _check_context_only_settings(llm_args: dict, *, warn: bool) -> None:
    """Refuse dense DWDP on a generation server; warn without the context role.

    Dense DWDP (``dwdp_config.enable_dense``) gathers the dense weights of
    every covered layer in every forward, decode steps included, so it is
    meant for context servers. The role is the one the server's ranks see in
    ``TRTLLM_DISAGG_ROLE`` (read by :func:`_serving_role`): ``launch_server``
    publishes it from ``--server_role``, the disaggregated MPI worker from its
    server's type, and a gRPC server, which takes no ``--server_role``, keeps
    the one its environment holds. A server without the context role gets a
    warning that names the remedy. Runs before the LLM is built, so a refused
    launch fails before the model loads.

    Args:
        llm_args: The merged LLM arguments of the server.
        warn: Whether to log the warning; an attached frontend repeats its
            launcher's check without it.

    Raises:
        click.BadParameter: Dense DWDP is refused for the server's role.
    """
    if not _dense_dwdp_setting(llm_args.get("dwdp_config")):
        return
    role = _serving_role(os.environ.get(DisaggLauncherEnvs.TLLM_DISAGG_ROLE))
    if role == "generation":
        raise click.BadParameter(
            f"{_DENSE_DWDP_FOR_CONTEXT_ONLY} Remove it from this server's "
            "configuration.")
    if warn and role != "context":
        logger.warning(
            "dwdp_config.enable_dense is set on a server without the context "
            "role, where dense DWDP gathers the dense weights of every covered "
            "layer in every forward, decode steps included. It is meant for "
            "context servers and refused on generation servers; launch "
            "context servers with --server_role CONTEXT (a gRPC server with "
            "TRTLLM_DISAGG_ROLE=context) so that the role is checked.")


def _refuse_dense_dwdp_on_generation_servers(
        disagg_cfg: DisaggServerConfig) -> None:
    """Refuse dense DWDP on the generation servers of a disaggregated MPI launch.

    Every rank of the launch runs this on the same configuration before the
    world is split, so a refused launch exits on every rank. The server
    process of an instance's leader would refuse it too, but only once the
    instance's other ranks wait for its MPI tasks, which then never come.

    Raises:
        click.BadParameter: A generation server enables dense DWDP, in its own
            settings or in the top-level ones it inherits.
    """
    for server_cfg in disagg_cfg.server_configs:
        if (_DISAGG_SERVER_ROLES.get(server_cfg.type) == "generation" and
                _dense_dwdp_setting(server_cfg.other_args.get("dwdp_config"))):
            raise click.BadParameter(
                f"{_DENSE_DWDP_FOR_CONTEXT_ONLY} Set it under context_servers "
                "only: the generation servers also inherit the top-level "
                "settings.")


def launch_server(
        host: str,
        port: int,
        llm_args: dict,
        tool_parser: Optional[str] = None,
        middleware: Sequence[str] = (),
        chat_template: Optional[str] = None,
        metadata_server_cfg: Optional[MetadataServerConfig] = None,
        server_role: Optional[ServerRole] = None,
        disagg_cluster_config: Optional[DisaggClusterConfig] = None,
        multimodal_server_config: Optional[MultimodalServerConfig] = None,
        served_model_name: Optional[str] = None,
        allow_request_chat_template: bool = False,
        num_input_processor_workers: int = 8,
        num_media_load_workers: int = 8,
        multi_frontend_enabled: bool = True,
        internal_disagg_auth_key: Optional[str] = None,
        report_addr: Optional[str] = None,
        enable_runtime_control_endpoints: bool = False,
        runtime_control_api_key: Optional[str] = None) -> None:

    # Publish an explicit serving role before constructing/spawning the model.
    # Remote-tail context workers use it to omit generation-only weights.
    if server_role is not None:
        os.environ[
            DisaggLauncherEnvs.TLLM_DISAGG_ROLE] = server_role.name.lower()

    backend = llm_args["backend"]
    model = served_model_name or llm_args["model"]

    multi_frontend = _init_multi_frontend_mode(llm_args, multi_frontend_enabled)
    _check_context_only_settings(llm_args,
                                 warn=not multi_frontend.is_attached_frontend)
    # An attached frontend gets the launcher's socket from its spawn env; the
    # launcher is its parent.
    multi_frontend_serving = None
    if multi_frontend.is_attached_frontend:
        attached_launcher_uds = os.environ.pop(LAUNCHER_UDS_ENV, None)
        if attached_launcher_uds is not None:
            multi_frontend_serving = MultiFrontendServing(
                launcher_uds=attached_launcher_uds,
                is_launcher=False,
                launcher_pid=os.getppid())
    if enable_runtime_control_endpoints and multi_frontend.num_frontends > 1:
        raise click.BadParameter(
            "Runtime control endpoints require num_serve_frontends=1 because "
            "replay protection is local to each frontend.")
    # Same hazard the disaggregated fleet guard covers: _spawn_attached_frontends
    # re-execs this command line verbatim, so with port 0 every frontend binds
    # its own kernel-assigned port instead of sharing one, and every frontend
    # also re-runs _publish_bound_address, leaving the reader with whichever
    # child wrote last.
    if (port == 0 or report_addr) and multi_frontend.num_frontends > 1:
        raise click.BadParameter(
            "port 0 and --report_addr are only supported with a single serving "
            f"frontend, but num_serve_frontends={multi_frontend.num_frontends}."
        )
    if multi_frontend.is_launcher or multi_frontend.is_attached_frontend:
        # The Responses API store is per-process in-memory: with several
        # frontends behind one SO_REUSEPORT port, a follow-up request may
        # land on a sibling that has no record of the previous response.
        # Disable storage group-wide (OpenAIServer.enable_store).
        if not os.getenv("TRTLLM_RESPONSES_API_DISABLE_STORE"):
            logger.warning(
                "num_serve_frontends > 1: stateful Responses API storage "
                "(store/previous_response_id) is disabled; the per-frontend "
                "in-memory store cannot be shared across frontends. Pass "
                "--num_serve_frontends 1 to keep it.")
        os.environ["TRTLLM_RESPONSES_API_DISABLE_STORE"] = "1"

    address_family = _listener_address_family("server", host, port)
    # port == 0 lets the kernel pick the port; the caller then needs a way
    # to learn it, either by service discovery or by report_addr.
    assert port > 0 or disagg_cluster_config is not None or report_addr, (
        "Port must be specified unless disagg cluster config or "
        "--report_addr is provided")
    port_lock = (_reserve_multi_frontend_port(host, port, address_family)
                 if multi_frontend.is_launcher else contextlib.nullcontext())
    # Every frontend process binds its own listening socket on the same port;
    # the kernel load-balances accepts across them.
    reuse_port = multi_frontend.is_launcher or multi_frontend.is_attached_frontend
    with port_lock, _bind_listener("server",
                                   host,
                                   port,
                                   family=address_family,
                                   reuse_port=reuse_port) as s:
        if port == 0:
            port = s.getsockname()[1]

        # Only now is the address final, and the socket stays bound from here
        # until uvicorn takes it over, so no one can steal the port in between.
        _publish_bound_address(report_addr, host, port)

        # The affinity ports are bound before the LLM is created, so the
        # executor's outgoing connections cannot take one of them.
        with (_frontend_affinity_sockets(
                host, port, address_family, multi_frontend,
                llm_args.get("serve_frontend_affinity_port_stride"),
                _diagnose_port_in_use) as
              (affinity_socket, attached_affinity_sockets),
              _provision_kv_cache_pool(
                  llm_args,
                  owns_engine=not multi_frontend.is_attached_frontend)):
            if backend == 'pytorch':
                llm_args.pop("build_config", None)
                llm = PyTorchLLM(**llm_args)
            else:
                raise click.BadParameter(
                    f"{backend} is not a known backend, check help for available options.",
                    param_hint="backend")

            # The finally below is the cleanup boundary for the attached
            # frontends: it must cover everything from their spawn through
            # server construction, middleware registration, and runtime, or a
            # failure in between leaks the child processes.
            frontend_children = []
            stop_watching_frontends = None
            sockets = [s]
            if affinity_socket is not None:
                sockets.append(affinity_socket)
            launcher_uds_sock = None
            try:
                if multi_frontend.is_launcher:
                    launcher_uds_sock, launcher_uds_path = _bind_launcher_uds(
                        llm)
                    _share_prometheus_multiproc_dir(
                        llm, os.path.dirname(launcher_uds_path))
                    frontend_children = _spawn_attached_frontends(
                        llm,
                        multi_frontend.num_frontends,
                        extra_env={LAUNCHER_UDS_ENV: launcher_uds_path},
                        report_failure=_report_observed_child_failure,
                        affinity_sockets=attached_affinity_sockets)
                    stop_watching_frontends = _watch_attached_frontends(
                        frontend_children)
                    multi_frontend_serving = MultiFrontendServing(
                        launcher_uds=launcher_uds_path, is_launcher=True)
                    sockets.append(launcher_uds_sock)

                server = OpenAIServer(
                    generator=llm,
                    model=model,
                    tool_parser=tool_parser,
                    server_role=server_role,
                    metadata_server_cfg=metadata_server_cfg,
                    disagg_cluster_config=disagg_cluster_config,
                    multimodal_server_config=multimodal_server_config,
                    chat_template=chat_template,
                    allow_request_chat_template=allow_request_chat_template,
                    input_processor_workers=num_input_processor_workers,
                    media_load_workers=num_media_load_workers,
                    internal_disagg_auth_key=internal_disagg_auth_key,
                    enable_runtime_control_endpoints=
                    enable_runtime_control_endpoints,
                    runtime_control_api_key=runtime_control_api_key,
                    multi_frontend_serving=multi_frontend_serving,
                    graceful_shutdown_timeout=llm.args.
                    serve_graceful_shutdown_timeout_s)
                _apply_fastapi_middlewares(server.app, middleware)

                # Optionally disable GC (default: not disabled)
                if os.getenv("TRTLLM_SERVER_DISABLE_GC", "0") == "1":
                    gc.disable()

                _signal_frontend_ready(multi_frontend)
                uvloop.run(server(host, port, sockets=sockets))
            finally:
                if stop_watching_frontends is not None:
                    stop_watching_frontends.set()
                if frontend_children:
                    _terminate_attached_frontends(frontend_children)
                if launcher_uds_sock is not None:
                    # Its file goes with the ipc dir (see _bind_launcher_uds).
                    launcher_uds_sock.close()


def launch_mm_encoder_server(
    host: str,
    port: int,
    encoder_args: dict,
    metadata_server_cfg: Optional[MetadataServerConfig] = None,
    allow_request_chat_template: bool = False,
) -> None:
    model = encoder_args["model"]
    encoder_args.pop("build_config", None)
    mm_encoder = MultimodalEncoder(**encoder_args)

    server = OpenAIServer(
        generator=mm_encoder,
        model=model,
        server_role=ServerRole.MM_ENCODER,
        metadata_server_cfg=metadata_server_cfg,
        tool_parser=None,
        allow_request_chat_template=allow_request_chat_template,
        graceful_shutdown_timeout=mm_encoder.args.
        serve_graceful_shutdown_timeout_s)
    uvloop.run(server(host, port))


# HF causal-LM architecture -> TRT-LLM text-embedding architecture. The embeddings
# subcommand declares embedding intent (cf. SGLang --is-embedding); this maps known
# sentence-transformers decoder backbones onto their pooling+normalize wrapper class.
# To support another decoder embedder, add its HF architecture here and a matching
# `*ForTextEmbedding` model class (see Qwen3ForTextEmbedding in modeling_qwen3.py).
# Architectures absent from this map are left as declared (the model self-routes).
_EMBEDDING_ARCH_MAP = {
    "Qwen3ForCausalLM": "Qwen3ForTextEmbedding",
}


def _resolve_embedding_architecture_override(
        model: str,
        trust_remote_code: bool,
        revision: Optional[str] = None) -> Optional[dict]:
    """Return a `model_kwargs` architecture override for a known embedder.

    Returns `None` to leave the model's declared architecture unchanged.
    Qwen3-Embedding ships as `Qwen3ForCausalLM` (a causal decoder) plus a
    sentence-transformers pooling/normalize pipeline. Overriding `architectures`
    flips both the model-class selection (`get_model_architecture`) and the
    encoder-vs-generation routing (`is_generation_model`) with one change.
    """
    try:
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(
            model, trust_remote_code=trust_remote_code, revision=revision)
        architectures = getattr(hf_config, "architectures", None) or []
    except Exception as e:  # noqa: BLE001 - config read is best-effort
        logger.warning(
            f"Could not read model config for embedding routing ({model}): {e}")
        return None

    if not architectures:
        return None
    target = _EMBEDDING_ARCH_MAP.get(architectures[0])
    if target is None:
        return None

    # Best-effort sanity check on a local checkpoint: warn (don't fail) if the
    # sentence-transformers pooling layout is absent. The subcommand is explicit
    # embedding intent, so we still proceed with last-token pooling + L2 norm.
    if os.path.isdir(model):
        pooling_cfg = os.path.join(model, "1_Pooling", "config.json")
        if not os.path.isfile(pooling_cfg):
            logger.warning(
                f"{model} maps to {target} but has no 1_Pooling/config.json; "
                "proceeding with last-token pooling + L2 normalize.")

    logger.info(f"Embedding routing: overriding architecture "
                f"{architectures[0]} -> {target}")
    return {"architectures": [target]}


def _reject_embedding_extra_model_inputs(cuda_graph_config: Any) -> None:
    """Reject encoder CUDA graph extra model inputs for the embeddings server.

    /v1/embeddings has no field for extra model inputs, so the server calls
    llm.encode() without them and every request would miss the declared inputs.
    `cuda_graph_config` is the raw --config mapping or a parsed config object.
    """
    if isinstance(cuda_graph_config, dict):
        specs = cuda_graph_config.get("extra_model_inputs")
    else:
        specs = getattr(cuda_graph_config, "extra_model_inputs", None)
    if specs:
        raise click.BadParameter(
            "cuda_graph_config.extra_model_inputs is not supported by "
            "trtllm-serve embeddings: /v1/embeddings has no field for extra "
            "model inputs. Pass them through the Python llm.encode() API "
            "instead.",
            param_hint="config")


def launch_embedding_server(
    host: str,
    port: int,
    llm_args: dict,
    max_queue_delay: float,
    max_queue_size: int,
    metadata_server_cfg: Optional[MetadataServerConfig] = None,
) -> None:
    model = llm_args["model"]
    llm_args.pop("build_config", None)
    # encode_only is forced on below; drop any value coming from --config to avoid
    # a duplicate keyword argument.
    llm_args.pop("encode_only", None)

    # Sentence-transformers decoder embedders (e.g. Qwen3-Embedding) declare a
    # causal-LM architecture; remap it to the text-embedding model class so the
    # forward returns pooled+normalized sentence embeddings instead of LM logits.
    override = _resolve_embedding_architecture_override(
        model, llm_args.get("trust_remote_code", False),
        llm_args.get("revision"))
    if override is not None:
        model_kwargs = dict(llm_args.get("model_kwargs") or {})
        if "architectures" in model_kwargs:
            # A user-supplied model_kwargs.architectures (e.g. via --config) wins;
            # log so the suppressed auto-remap isn't a silent surprise.
            logger.info(
                "Embedding routing: keeping user-supplied model_kwargs "
                f"architectures={model_kwargs['architectures']} (auto-remap to "
                f"{override['architectures']} suppressed).")
        else:
            model_kwargs["architectures"] = override["architectures"]
        llm_args["model_kwargs"] = model_kwargs

    # Encoder-only (embedding) serving uses the synchronous llm.encode() fast path
    # (no KV cache / sampler / scheduler), coalesced behind the dynamic batcher.
    llm = PyTorchLLM(encode_only=True, **llm_args)

    server = OpenAIServer(
        generator=llm,
        model=model,
        server_role=ServerRole.EMBEDDING,
        metadata_server_cfg=metadata_server_cfg,
        tool_parser=None,
        embedding_max_queue_delay=max_queue_delay,
        embedding_max_queue_size=max_queue_size,
        graceful_shutdown_timeout=llm.args.serve_graceful_shutdown_timeout_s)
    asyncio.run(server(host, port))


def launch_visual_gen_server(
        host: str,
        port: int,
        model: str,
        visual_gen_args: Optional["VisualGenArgs"] = None,
        metadata_server_cfg: Optional[MetadataServerConfig] = None,
        middleware: Sequence[str] = (),
):
    """Launch a VISUAL_GEN model server for image/video generation.

    Args:
        host: Server hostname.
        port: Server port.
        model: Model path or HuggingFace Hub model ID.
        visual_gen_args: Optional validated VisualGenArgs for model configuration.
        metadata_server_cfg: Optional metadata server configuration.
    """
    # Only global rank 0 serves HTTP; in external multi-rank launch ranks > 0
    # must become workers before binding, else every rank on a multi-GPU node
    # races the same port and all but one die EADDRINUSE. VisualGen() on a
    # worker rank never returns (sys.exit in __init__).
    from tensorrt_llm._torch.visual_gen.executor import _detect_external_launch
    from tensorrt_llm.visual_gen import VisualGen
    ext = _detect_external_launch()
    if ext is not None and ext[0] != 0:
        VisualGen(model=model, args=visual_gen_args)
        return

    # Reserve the listening (host, port) by binding the socket *before*
    # constructing the VisualGen pipeline, then hand the bound socket to
    # uvicorn. VisualGen initialization can take many minutes; if we deferred
    # the bind until uvicorn started, anything else on the host could grab the
    # port in that window and trtllm-serve would die at bind() time.
    address_family = _listener_address_family("VisualGen server", host, port)
    with _bind_listener("VisualGen server", host, port,
                        family=address_family) as s:
        logger.info(f"Initializing VisualGen ({model})")

        previous_sigterm_handler = signal.signal(signal.SIGTERM,
                                                 _terminate_visual_gen_startup)
        visual_gen_model = None
        try:
            visual_gen_model = VisualGen(model=model, args=visual_gen_args)

            n_workers = visual_gen_model.args.parallel_config.n_workers
            logger.info(f"World size: {n_workers}")
            logger.info(
                f"CFG size: {visual_gen_model.args.parallel_config.cfg_size}")
            logger.info("Ulysses size: "
                        f"{visual_gen_model.args.parallel_config.ulysses_size}")

            server = OpenAIServer(generator=visual_gen_model,
                                  model=model,
                                  server_role=ServerRole.VISUAL_GEN,
                                  metadata_server_cfg=metadata_server_cfg,
                                  tool_parser=None)
            _apply_fastapi_middlewares(server.app, middleware)
            uvloop.run(server(host, port, sockets=[s]))
        except _VisualGenStartupTermination as e:
            if visual_gen_model is not None:
                visual_gen_model.shutdown()
            raise SystemExit(128 + e.signum) from None
        finally:
            signal.signal(
                signal.SIGTERM,
                signal.SIG_DFL if previous_sigterm_handler is None else
                previous_sigterm_handler,
            )


@click.command("serve")
@click.argument("model", type=str)
@stability_option(
    "--tokenizer",
    type=str,
    default=None,
    help="Path or name of the tokenizer. When using the PyTorch backend, "
    "this replaces the default HuggingFace tokenizer.",
    status="beta")
@stability_option(
    "--custom_tokenizer",
    type=str,
    default=None,
    help=
    "Custom tokenizer type: alias (e.g., 'deepseek_v32') or Python import path "
    "(e.g., 'tensorrt_llm.tokenizer.deepseek_v32.DeepseekV32Tokenizer').",
    status="prototype")
@stability_option(
    "--post_processor_hook",
    type=str,
    default=None,
    help="Python import path of a user post-processing hook applied after "
    "detokenization and before the per-endpoint response formatter (e.g. "
    "'my_pkg.guardrail.MyPostProcessorHook'). The class must be importable "
    "and picklable, take no constructor arguments, and be callable as "
    "'__call__(chunk) -> verdict' (see tensorrt_llm.executor.postprocessor_hook). "
    "It runs once per output, per streaming chunk, and may rewrite, "
    "suppress, or terminate the output; it owns its own per-request state.",
    status="prototype")
@stability_option("--host",
                  type=str,
                  default="localhost",
                  help="Hostname of the server.",
                  status="beta")
@stability_option("--port",
                  type=int,
                  default=8000,
                  help="Port of the server.",
                  status="beta")
@stability_option(
    "--backend",
    type=click.Choice(["pytorch"]),
    default="pytorch",
    help="The backend to use to serve the model. Default is pytorch backend.",
    status="deprecated")
@stability_option(
    "--generation-config",
    type=click.Choice(["auto", "trtllm"]),
    default=_LLM_ARGS_FIELDS["generation_config"].default,
    help="Sampling defaults source. 'auto' loads supported values from the "
    "model's generation_config.json; 'trtllm' uses TRT-LLM defaults.",
    status="prototype")
@stability_option("--custom_module_dirs",
                  type=click.Path(exists=True,
                                  readable=True,
                                  path_type=Path,
                                  resolve_path=True),
                  default=None,
                  multiple=True,
                  help="Paths to custom module directories to import.",
                  status="prototype")
@stability_option('--log_level',
                  type=click.Choice(severity_map.keys()),
                  default='info',
                  help="The logging level.",
                  status="beta")
@stability_option("--max_beam_width",
                  type=int,
                  default=_LLM_ARGS_FIELDS["max_beam_width"].default,
                  help="Maximum number of beams for beam search decoding.",
                  status="beta")
@stability_option(
    "--max_batch_size",
    type=int,
    default=_LLM_ARGS_FIELDS["max_batch_size"].default,
    help="Maximum number of requests that the engine can schedule.",
    status="beta")
@stability_option(
    "--max_num_tokens",
    type=int,
    default=_LLM_ARGS_FIELDS["max_num_tokens"].default,
    help=
    "Maximum number of batched input tokens after padding is removed in each batch.",
    status="beta")
@stability_option(
    "--max_seq_len",
    type=int,
    default=None,
    help="Maximum total length of one request, including prompt and outputs. "
    "If unspecified, the value is deduced from the model config.",
    status="beta")
@stability_option("--tensor_parallel_size",
                  "--tp_size",
                  type=int,
                  default=1,
                  help='Tensor parallelism size.',
                  status='beta')
@stability_option("--pipeline_parallel_size",
                  "--pp_size",
                  type=int,
                  default=1,
                  help='Pipeline parallelism size.',
                  status='beta')
@stability_option("--context_parallel_size",
                  "--cp_size",
                  type=int,
                  default=1,
                  help='Context parallelism size.',
                  status='beta')
@stability_option("--moe_expert_parallel_size",
                  "--ep_size",
                  type=int,
                  default=None,
                  help="expert parallelism size",
                  status="beta")
@stability_option(
    "--moe_cluster_parallel_size",
    "--cluster_size",
    type=int,
    default=None,
    help="[Deprecated] Expert cluster parallelism size. "
    "This option is no longer supported and will be removed in a future release.",
    status="deprecated")
@stability_option(
    "--gpus_per_node",
    type=int,
    default=None,
    help=
    "Number of GPUs per node. Default to None, and it will be detected automatically.",
    status="beta")
@stability_option("--free_gpu_memory_fraction",
                  "--kv_cache_free_gpu_memory_fraction",
                  type=float,
                  default=0.9,
                  help="Free GPU memory fraction reserved for KV Cache, "
                  "after allocating model weights and buffers.",
                  status="beta")
@stability_option(
    "--kv_cache_dtype",
    type=click.Choice(("auto", "fp8", "nvfp4")),
    default="auto",
    help="KV cache quantization dtype for PyTorch backend. "
    "'auto' uses checkpoint/model metadata; explicit values force override.",
    status="prototype")
@stability_option("--num_postprocess_workers",
                  type=int,
                  default=0,
                  help="Number of workers to postprocess raw responses "
                  "to comply with OpenAI protocol.",
                  status="prototype")
@stability_option(
    "--num_serve_frontends",
    type=click.IntRange(min=1, max=MAX_NUM_FRONTENDS),
    # LlmArgs keeps 1: only trtllm-serve spawns frontends.
    default=8,
    help="Number of HTTP or OpenEngine frontend processes serving one "
    "executor; values > 1 share the serving port via "
    "SO_REUSEPORT (classic IPC executor path only). Without an explicit "
    "value, --grpc and configurations that support a single frontend only "
    "use 1 and log why.",
    status="beta")
@stability_option("--num_input_processor_workers",
                  type=click.IntRange(min=1),
                  default=8,
                  help="Size of the dedicated thread pool that runs the HF "
                  "input processor (multimodal preprocess) on the chat "
                  "and completion endpoints.",
                  status="prototype")
@stability_option("--num_media_load_workers",
                  type=click.IntRange(min=1),
                  default=8,
                  help="Size of the dedicated thread pool that decodes media "
                  "payloads (image / video / audio) for multimodal "
                  "requests.",
                  status="prototype")
@stability_option("--trust_remote_code",
                  is_flag=True,
                  default=False,
                  help="Flag for HF transformers.",
                  status="beta")
@stability_option("--hf_revision",
                  "--revision",
                  "revision",
                  type=str,
                  default=None,
                  help="The revision to use for the HuggingFace model "
                  "(branch name, tag name, or commit id). "
                  "Prefer --hf_revision over --revision.",
                  status="beta")
@stability_option(
    "--config",
    "--extra_llm_api_options",
    "extra_llm_api_options",
    type=str,
    default=None,
    help="Path to a YAML configuration file. Explicit CLI flags take precedence "
    "over values in this file, while --set overrides both. Can be specified "
    "as either --config or --extra_llm_api_options.",
    status="prototype")
@stability_option(
    "--set",
    "config_overrides",
    type=str,
    multiple=True,
    help="Override a canonical LlmArgs path as PATH=YAML_VALUE. Repeatable; "
    "applied after --config and dedicated CLI flags. Not supported for "
    "startup, telemetry, environment, server-only, or VisualGen settings.",
    status="prototype")
@stability_option("--reasoning_parser",
                  type=click.Choice(["auto"] +
                                    list(ReasoningParserFactory.keys())),
                  default=None,
                  help="Specify the parser for reasoning models. "
                  "Use 'auto' to automatically select based on the model.",
                  status="prototype")
@stability_option("--tool_parser",
                  type=click.Choice(["auto"] +
                                    list(ToolParserFactory.parsers.keys())),
                  default=None,
                  help="Specify the parser for tool models. "
                  "Use 'auto' to automatically select based on the model.",
                  status="prototype")
@stability_option("--metadata_server_config_file",
                  type=str,
                  default=None,
                  help="Path to metadata server config file",
                  status="prototype")
@stability_option(
    "--server_role",
    type=str,
    default=None,
    help="Server role for disaggregated serving. "
    "CONTEXT=prefill (prompt processing), GENERATION=decode (token generation), "
    "MM_ENCODER=multimodal encoder, VISUAL_GEN=visual generation. "
    "Required when using service registry.",
    status="prototype")
@stability_option("--otlp_traces_endpoint",
                  type=str,
                  default=None,
                  help="Target URL to which OpenTelemetry traces will be sent.",
                  status="prototype")
@stability_option(
    "--telemetry/--no-telemetry",
    default=True,
    help="Enable or disable anonymous usage telemetry collection.",
    status="beta")
@stability_option("--disagg_cluster_uri",
                  type=str,
                  default=None,
                  help="URI of the disaggregated cluster.",
                  status="prototype")
@stability_option("--enable_chunked_prefill",
                  is_flag=True,
                  default=False,
                  help="Enable chunked prefill",
                  status="prototype")
@stability_option("--enable_attention_dp",
                  is_flag=True,
                  default=False,
                  help="Enable attention data parallel.",
                  status="beta")
@stability_option(
    "--enable_locality_domains",
    is_flag=True,
    default=False,
    help="Enable locality domain execution for supported PyTorch backend ops.",
    status="prototype")
@stability_option("--media_io_kwargs",
                  type=str,
                  default=None,
                  help="Keyword arguments for media I/O as a JSON string. "
                  "Keys are modality names (\"video\", \"image\", \"audio\") "
                  "whose values are dicts of keyword arguments forwarded to "
                  "the corresponding loader. "
                  "Example: '{\"video\": {\"extract_audio\": true, "
                  "\"num_frames\": 16}}' to enable audio extraction from "
                  "video files.",
                  status="prototype")
@stability_option("--video_pruning_rate",
                  type=float,
                  default=None,
                  help="Pruning rate for video frames in multimodal models. "
                  "Applied by Efficient Video Sampling (EVS). "
                  "None disables EVS, values in [0, 1) enable pruning.",
                  status="prototype")
@stability_option("--chat_template",
                  type=str,
                  default=None,
                  help="Specify a custom chat template. "
                  "Can be a file path or one-liner template string",
                  status="prototype")
@stability_option(
    "--allow_request_chat_template",
    is_flag=True,
    default=False,
    help="Allow clients to supply per-request chat_template values. "
    "Only enable this for trusted clients.",
    status="prototype")
@stability_option(
    "--middleware",
    multiple=True,
    type=str,
    help="FastAPI middleware import path to add to the server app. "
    "Can be specified multiple times. Each value must point to either "
    "a middleware class or an async HTTP middleware function.",
    status="prototype")
@stability_option(
    "--grpc",
    is_flag=True,
    default=False,
    help="Run the selected gRPC protocol instead of the OpenAI HTTP server.",
    status="prototype")
@stability_option("--grpc-protocol",
                  type=click.Choice(["smg", "openengine"]),
                  default="smg",
                  help="Protocol used when --grpc is enabled.",
                  status="prototype")
@stability_option(
    "--openengine-metrics-port",
    type=click.IntRange(1, 65535),
    default=None,
    help="OpenEngine Prometheus HTTP port (default: 8001). "
    "Enabled by return_perf_metrics or perf_metrics_output_dir in the engine configuration.",
    status="prototype")
@stability_option(
    "--openengine-metrics-host",
    type=str,
    default=None,
    help="Bind address for OpenEngine Prometheus (default: --host).",
    status="prototype")
@stability_option(
    "--openengine-enable-load-metrics",
    is_flag=True,
    default=False,
    help=
    "Expose per-rank KV occupancy for OpenEngine routing, independently of KV events.",
    status="prototype")
@stability_option(
    "--served_model_name",
    type=str,
    default=None,
    help="The model name used in the API. If not specified, the model path is "
    "used as the model name. This is useful when the model path is long or "
    "when you want to expose a custom name to clients.",
    status="prototype")
@stability_option(
    "--enable_visual_gen",
    is_flag=True,
    default=False,
    help="Enable VisualGen runtime for model checkpoints that support both LLM "
    "and Visual Generation. Not required if --visual_gen_args specified "
    "or the model supports Visual Generation only.",
    status="prototype")
@stability_option("--visual_gen_args",
                  type=str,
                  default=None,
                  help="Path to a YAML file with VisualGen engine args.",
                  status="prototype")
@stability_option(
    "--agent_percentage",
    type=float,
    default=0.0,
    help="The percentage of agent requests to schedule. Defaults to 0.0. "
    "Should be between 0.0 and 1.0.",
    status="prototype")
@stability_option(
    "--agent_types",
    type=str,
    default=None,
    help=
    "Types of agents to schedule. Now Only Support Open Deep Research agent.",
    status="prototype")
@stability_option(
    "--report_addr",
    type=str,
    default=None,
    help="Write the host:port this server actually bound to this file, "
    "atomically, once the socket is bound. Lets --port 0 be used and have the "
    "launcher read the kernel-assigned port back instead of reserving one up "
    "front.",
    status="prototype")
@stability_option(
    "--enable_sleep_mode",
    is_flag=True,
    default=False,
    help=
    "Enable runtime-memory checkpointing and authenticated HTTP control endpoints.",
    status="prototype")
def serve(
    model: str,
    tokenizer: Optional[str],
    custom_tokenizer: Optional[str],
    post_processor_hook: Optional[str],
    host: str,
    port: int,
    log_level: str,
    backend: str,
    generation_config: str,
    max_beam_width: int,
    max_batch_size: int,
    max_num_tokens: int,
    max_seq_len: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    context_parallel_size: int,
    moe_expert_parallel_size: Optional[int],
    moe_cluster_parallel_size: Optional[int],
    gpus_per_node: Optional[int],
    free_gpu_memory_fraction: float,
    kv_cache_dtype: str,
    num_postprocess_workers: int,
    num_serve_frontends: int,
    num_input_processor_workers: int,
    num_media_load_workers: int,
    trust_remote_code: bool,
    revision: Optional[str],
    extra_llm_api_options: Optional[str],
    config_overrides: tuple[str, ...],
    reasoning_parser: Optional[str],
    tool_parser: Optional[str],
    metadata_server_config_file: Optional[str],
    server_role: Optional[str],
    otlp_traces_endpoint: Optional[str],
    enable_chunked_prefill: bool,
    enable_attention_dp: bool,
    disagg_cluster_uri: Optional[str],
    media_io_kwargs: Optional[str],
    agent_percentage: float,
    agent_types: Optional[str],
    video_pruning_rate: Optional[float],
    telemetry: bool,
    custom_module_dirs: list[Path],
    chat_template: Optional[str],
    allow_request_chat_template: bool,
    middleware: tuple[str, ...],
    grpc: bool,
    grpc_protocol: str,
    openengine_enable_load_metrics: bool,
    openengine_metrics_port: Optional[int],
    openengine_metrics_host: Optional[str],
    enable_visual_gen: bool,
    served_model_name: Optional[str],
    visual_gen_args: Optional[str],
    report_addr: Optional[str],
    enable_locality_domains: bool,
    enable_sleep_mode: bool,
) -> None:
    """Running an OpenAI API compatible server

    MODEL: model name or Hugging Face checkpoint path
    """
    is_visual_gen = enable_visual_gen or visual_gen_args is not None
    if is_visual_gen:
        # VisualGen reports no startup error category or startup state.
        _mark_visual_gen()
    logger.set_level(log_level)
    if enable_sleep_mode and backend != "pytorch":
        raise click.UsageError(
            "--enable_sleep_mode is only supported by the PyTorch backend.")
    if enable_sleep_mode and grpc:
        raise click.UsageError(
            "--enable_sleep_mode is only supported by the HTTP server.")
    runtime_control_api_key = None
    if enable_sleep_mode:
        runtime_control_api_key = os.getenv("TRTLLM_RUNTIME_CONTROL_API_KEY")
        if not runtime_control_api_key:
            raise click.UsageError(
                "--enable_sleep_mode requires TRTLLM_RUNTIME_CONTROL_API_KEY.")

    if not grpc and grpc_protocol != "smg":
        raise click.UsageError("--grpc-protocol requires --grpc")
    if openengine_enable_load_metrics and (not grpc
                                           or grpc_protocol != "openengine"):
        raise click.UsageError(
            "--openengine-enable-load-metrics requires --grpc --grpc-protocol openengine"
        )
    if openengine_metrics_port is not None and (not grpc or grpc_protocol
                                                != "openengine"):
        raise click.UsageError(
            "--openengine-metrics-port requires --grpc --grpc-protocol openengine"
        )
    if openengine_metrics_host is not None and (not grpc or grpc_protocol
                                                != "openengine"):
        raise click.UsageError(
            "--openengine-metrics-host requires --grpc --grpc-protocol openengine"
        )
    if openengine_metrics_port is not None and openengine_metrics_port == port:
        raise click.UsageError(
            "--openengine-metrics-port must differ from --port")

    if moe_cluster_parallel_size is not None:
        logger.warning(
            "--moe_cluster_parallel_size / --cluster_size is deprecated and "
            "no longer supported. This option will be removed in a future release."
        )

    if tool_parser == "auto":
        resolved = resolve_auto_tool_parser(model)
        if resolved is None:
            supported_model_types = ", ".join(
                sorted(MODEL_TYPE_TO_TOOL_PARSER.keys()))
            raise click.BadParameter(
                f"Cannot auto-detect tool parser for model '{model}'. "
                f"Supported model types for auto-detection: {supported_model_types}. "
                f"Please specify a parser explicitly: "
                f"{list(ToolParserFactory.parsers.keys())}",
                param_hint="--tool_parser")
        logger.info(f"Auto-detected tool parser: {resolved}")
        tool_parser = resolved

    if reasoning_parser == "auto":
        resolved = resolve_auto_reasoning_parser(model)
        if resolved is None:
            raise click.BadParameter(
                f"Cannot auto-detect reasoning parser for model '{model}'. "
                f"Supported model types for auto-detection: qwen3, qwen3_moe, "
                f"qwen3_5, qwen3_5_moe, qwen3_next, deepseek_v3 (R1 only), "
                f"deepseek_v32 (R1 only), deepseek_v4, nemotron_h, gemma4, "
                f"kimi_k2, kimi_k25. "
                f"Please specify a parser explicitly: "
                f"{list(ReasoningParserFactory.keys())}",
                param_hint="--reasoning_parser")
        logger.info(f"Auto-detected reasoning parser: {resolved}")
        reasoning_parser = resolved
    if "--revision" in sys.argv:
        logger.warning("--revision is deprecated, use --hf_revision instead.")

    for custom_module_dir in custom_module_dirs:
        try:
            import_custom_module_from_dir(custom_module_dir)
        except Exception as e:
            logger.error(
                f"Failed to import custom module from {custom_module_dir}: {e}")
            raise e

    explicit_cli_keys = collect_explicit_cli_keys(
        exclude=("extra_llm_api_options", "config", "config_overrides"))

    if not is_visual_gen and get_is_diffusion_only_model(model):
        is_visual_gen = True
        _mark_visual_gen()
    # Honor YAML telemetry opt-out before --set parsing can raise an error.
    raw_llm_args_extra_dict = {}
    if not is_visual_gen and extra_llm_api_options is not None:
        with open(extra_llm_api_options, 'r') as f:
            raw_llm_args_extra_dict = yaml.safe_load(f)
        if raw_llm_args_extra_dict is None:
            raw_llm_args_extra_dict = {}
        elif not isinstance(raw_llm_args_extra_dict, dict):
            raise ConfigInputError("Configuration file root must be a mapping.")
    if not is_visual_gen:
        _command_telemetry.apply_raw_config_telemetry_opt_out(
            raw_llm_args_extra_dict,
            usage_context=_telemetry_config.UsageContext.CLI_SERVE,
            component="server",
            explicit_cli_telemetry="telemetry" in explicit_cli_keys,
        )
    if config_overrides and is_visual_gen:
        raise click.BadParameter(
            "--set configures LlmArgs and is not supported by VisualGen.",
            param_hint="--set")
    parsed_config_overrides = _parse_config_overrides(config_overrides)

    if enable_sleep_mode and is_visual_gen:
        raise click.UsageError(
            "--enable_sleep_mode is only supported by the HTTP "
            "text-generation server.")

    def _serve_llm():
        try:
            from tensorrt_llm.usage.usage_lib import _mark_llm_startup
            _mark_llm_startup()
        except Exception:
            pass
        nonlocal server_role, allow_request_chat_template
        llm_args, _ = get_llm_args(
            model=model,
            tokenizer=tokenizer,
            custom_tokenizer=custom_tokenizer,
            post_processor_hook=post_processor_hook,
            backend=backend,
            generation_config=generation_config,
            max_beam_width=max_beam_width,
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            max_seq_len=max_seq_len,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            context_parallel_size=context_parallel_size,
            moe_expert_parallel_size=moe_expert_parallel_size,
            moe_cluster_parallel_size=moe_cluster_parallel_size,
            gpus_per_node=gpus_per_node,
            free_gpu_memory_fraction=free_gpu_memory_fraction,
            kv_cache_dtype=kv_cache_dtype,
            num_postprocess_workers=num_postprocess_workers,
            num_serve_frontends=num_serve_frontends,
            trust_remote_code=trust_remote_code,
            revision=revision,
            reasoning_parser=reasoning_parser,
            otlp_traces_endpoint=otlp_traces_endpoint,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_attention_dp=enable_attention_dp,
            enable_locality_domains=enable_locality_domains,
            video_pruning_rate=video_pruning_rate,
            telemetry=telemetry,
            agent_percentage=agent_percentage,
            agent_types=agent_types,
            explicit_cli_keys=explicit_cli_keys)

        llm_args_extra_dict = dict(raw_llm_args_extra_dict)
        extra_allow_request_chat_template = _pop_bool_config_option(
            llm_args_extra_dict, "allow_request_chat_template")
        allow_request_chat_template = (allow_request_chat_template
                                       or extra_allow_request_chat_template)
        internal_disagg_auth_key = validate_internal_request_auth_key(
            llm_args_extra_dict.pop("internal_request_auth_key", None))
        # Apply to the raw mapping first so a higher-precedence override can
        # replace an invalid YAML value before nested config construction.
        llm_args_extra_dict = _apply_config_overrides(llm_args_extra_dict,
                                                      parsed_config_overrides)
        set_top_level_fields = {
            override.path[0]
            for override in parsed_config_overrides if len(override.path) == 1
        }
        merge_explicit_cli_keys = explicit_cli_keys - set_top_level_fields
        llm_args = update_llm_args_with_extra_dict(
            llm_args,
            llm_args_extra_dict,
            explicit_cli_keys=merge_explicit_cli_keys)
        # Reapply to the effective mapping so --set remains the final source
        # even when an explicit convenience flag targets the same nested field.
        llm_args = _apply_config_overrides(llm_args, parsed_config_overrides)
        if enable_sleep_mode and llm_args.get("sleep_config") is None:
            llm_args["sleep_config"] = SleepConfig()

        _resolve_default_num_serve_frontends(
            llm_args,
            requested=("num_serve_frontends" in explicit_cli_keys
                       or "num_serve_frontends" in llm_args_extra_dict),
            port=port,
            grpc=grpc,
            report_addr=report_addr,
            enable_sleep_mode=enable_sleep_mode)

        _apply_effective_telemetry_config(llm_args, telemetry=telemetry)
        # Preserve requested settings for failures before entering the LLM constructor.
        try:
            from tensorrt_llm.usage.usage_lib import _capture_startup_context
            _capture_startup_context(requested=llm_args)
        except Exception:
            pass

        metadata_server_cfg = parse_metadata_server_config_file(
            metadata_server_config_file)

        # Specify disagg_cluster_config in config file or through command line "--disagg_cluster_uri",
        # but disagg_cluster_uri takes precedence over cluster uri in config file
        disagg_cluster_config = llm_args.pop("disagg_cluster", None)
        if disagg_cluster_config:
            disagg_cluster_config = extract_disagg_cluster_config(
                disagg_cluster_config, disagg_cluster_uri)
        elif disagg_cluster_uri:
            disagg_cluster_config = DisaggClusterConfig(
                cluster_uri=disagg_cluster_uri)

        if metadata_server_cfg is not None or disagg_cluster_config is not None:
            assert (
                server_role is not None
            ), "server_role is required when metadata_server_cfg or disagg_cluster_config is provided"
        if server_role is not None:
            try:
                server_role = ServerRole[server_role.upper()]
            except KeyError:
                raise ValueError(f"Invalid server role: {server_role}. " \
                                f"Must be one of: {', '.join([role.name for role in ServerRole])}")
        # Parse media_io_kwargs from JSON string to dict if provided
        parsed_media_io_kwargs = None
        if media_io_kwargs is not None:
            try:
                parsed_media_io_kwargs = json.loads(media_io_kwargs)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON for media_io_kwargs: {e}")

        multimodal_server_config = MultimodalServerConfig(
            media_io_kwargs=parsed_media_io_kwargs)

        if grpc:
            effective_num_serve_frontends = llm_args.get(
                "num_serve_frontends", num_serve_frontends)
            if effective_num_serve_frontends != 1 and grpc_protocol != "openengine":
                raise click.UsageError(
                    "Multiple gRPC frontends require --grpc-protocol openengine."
                )

            # gRPC mode: launch gRPC server instead of OpenAI HTTP server
            # Check for unsupported arguments that are silently ignored in gRPC mode
            unsupported_args = {
                "tool_parser":
                tool_parser,
                "middleware":
                middleware if middleware else None,
                "chat_template":
                chat_template,
                "allow_request_chat_template":
                allow_request_chat_template
                if allow_request_chat_template else None,
                "internal_request_auth_key":
                internal_disagg_auth_key
                if grpc_protocol != "openengine" else None,
                "metadata_server_config_file":
                metadata_server_config_file,
                "server_role":
                server_role,
                "disagg_cluster_config":
                disagg_cluster_config,
            }
            for name, value in unsupported_args.items():
                if value is not None:
                    raise ConfigInputError(
                        f"Argument '{name}' is not supported when running in gRPC mode. "
                        f"The gRPC server is designed for use with external routers that handle "
                        f"these features (e.g., tool parsing, chat templates).")
            if grpc_protocol == "smg":
                if find_spec("smg_grpc_proto") is None:
                    raise ConfigInputError(
                        "gRPC serving with the SMG protocol requires the optional "
                        "'smg-grpc-proto' package. Install it with: "
                        'pip install "tensorrt_llm[grpc-smg]"')

                from tensorrt_llm.grpc.smg.server import launch_smg_server

                _check_context_only_settings(llm_args, warn=True)
                launch_smg_server(host,
                                  port,
                                  llm_args,
                                  served_model_name=served_model_name)
            else:
                try:
                    from tensorrt_llm.grpc.openengine.server import \
                        launch_server as launch_grpc_server
                except ImportError as error:
                    raise click.ClickException(
                        f"Failed to import OpenEngine support: {error}. "
                        "Restore the required gRPC runtime with `python -m pip "
                        "install \"grpcio>=1.73.1,<2\" "
                        "\"grpcio-health-checking>=1.73.1,<1.82\"`.") from error

                grpc_multi_frontend = _init_multi_frontend_mode(llm_args,
                                                                enabled=True)
                _check_context_only_settings(
                    llm_args, warn=not grpc_multi_frontend.is_attached_frontend)
                with _provision_kv_cache_pool(
                        llm_args,
                        owns_engine=not grpc_multi_frontend.is_attached_frontend
                ):
                    launch_grpc_server(
                        host,
                        port,
                        llm_args,
                        served_model_name=served_model_name,
                        report_failure=_report_observed_child_failure,
                        enable_load_metrics=openengine_enable_load_metrics,
                        metrics_host=openengine_metrics_host,
                        metrics_port=8001 if openengine_metrics_port is None
                        else openengine_metrics_port,
                        subagent_affinity_auth_key=internal_disagg_auth_key)
        else:
            # Default: launch OpenAI HTTP server
            launch_server(
                host,
                port,
                llm_args,
                tool_parser,
                middleware,
                chat_template,
                metadata_server_cfg,
                server_role,
                disagg_cluster_config,
                multimodal_server_config,
                served_model_name=served_model_name,
                allow_request_chat_template=allow_request_chat_template,
                num_input_processor_workers=num_input_processor_workers,
                num_media_load_workers=num_media_load_workers,
                internal_disagg_auth_key=internal_disagg_auth_key,
                report_addr=report_addr,
                enable_runtime_control_endpoints=enable_sleep_mode,
                runtime_control_api_key=runtime_control_api_key)

    def _serve_visual_gen():
        from tensorrt_llm.visual_gen.args import VisualGenArgs

        parsed_visual_gen_args = (VisualGenArgs.from_yaml(visual_gen_args)
                                  if visual_gen_args is not None else None)

        metadata_server_cfg = parse_metadata_server_config_file(
            metadata_server_config_file)

        launch_visual_gen_server(host, port, model, parsed_visual_gen_args,
                                 metadata_server_cfg, middleware)

    # Only the OpenAI HTTP path publishes the bound address. Fail loudly rather
    # than leaving a launcher waiting forever on a file nobody writes.
    if report_addr and (grpc or is_visual_gen):
        raise click.BadParameter(
            "--report_addr is only supported for the OpenAI HTTP server, not "
            f"the {'gRPC' if grpc else 'VisualGen'} server.")
    if is_visual_gen:
        if grpc:
            raise click.UsageError(
                "--grpc is not supported by the VisualGen server")
        _serve_visual_gen()
    else:
        _serve_llm()


@click.command("mm_embedding_serve")
@click.argument("model", type=str)
@stability_option("--host",
                  type=str,
                  default="localhost",
                  help="Hostname of the server.",
                  status="beta")
@stability_option("--port",
                  type=int,
                  default=8000,
                  help="Port of the server.",
                  status="beta")
@stability_option('--log_level',
                  type=click.Choice(severity_map.keys()),
                  default='info',
                  help="The logging level.",
                  status="beta")
@stability_option(
    "--max_batch_size",
    type=int,
    default=_LLM_ARGS_FIELDS["max_batch_size"].default,
    help="Maximum number of requests that the engine can schedule.",
    status="beta")
@stability_option(
    "--max_num_tokens",
    type=int,
    default=16384,  # set higher default max_num_tokens for multimodal encoder
    help=
    "Maximum number of batched input tokens after padding is removed in each batch.",
    status="beta")
@stability_option(
    "--gpus_per_node",
    type=int,
    default=None,
    help="Number of GPUs per node. Default to None, and it will be "
    "detected automatically.",
    status="beta")
@stability_option("--trust_remote_code",
                  is_flag=True,
                  default=False,
                  help="Flag for HF transformers.",
                  status="beta")
@stability_option(
    "--config",
    "--extra_encoder_options",
    "extra_encoder_options",
    type=str,
    default=None,
    help="Path to a YAML configuration file. Explicit CLI flags take precedence "
    "over values in this file. Prefer --config over --extra_encoder_options.",
    status="prototype")
@stability_option("--hf_revision",
                  "--revision",
                  "revision",
                  type=str,
                  default=None,
                  help="The revision to use for the HuggingFace model "
                  "(branch name, tag name, or commit id).",
                  status="beta")
@stability_option("--free_gpu_memory_fraction",
                  type=float,
                  default=0.9,
                  help="Free GPU memory fraction reserved for KV Cache, "
                  "after allocating model weights and buffers.",
                  status="beta")
@stability_option("--tensor_parallel_size",
                  "--tp_size",
                  type=int,
                  default=1,
                  help="Tensor parallelism size.",
                  status="beta")
@stability_option("--metadata_server_config_file",
                  type=str,
                  default=None,
                  help="Path to metadata server config file",
                  status="prototype")
@stability_option(
    "--allow_request_chat_template",
    is_flag=True,
    default=False,
    help="Allow clients to supply per-request chat_template values. "
    "Only enable this for trusted clients.",
    status="prototype")
@stability_option(
    "--telemetry/--no-telemetry",
    default=True,
    help="Enable or disable anonymous usage telemetry collection.",
    status="beta")
def serve_encoder(
    model: str,
    host: str,
    port: int,
    log_level: str,
    max_batch_size: int,
    max_num_tokens: int,
    gpus_per_node: Optional[int],
    trust_remote_code: bool,
    extra_encoder_options: Optional[str],
    revision: Optional[str],
    free_gpu_memory_fraction: float,
    tensor_parallel_size: int,
    metadata_server_config_file: Optional[str],
    allow_request_chat_template: bool,
    telemetry: bool,
):
    """Running an OpenAI API compatible server

    MODEL: model name or Hugging Face checkpoint path
    """
    logger.set_level(log_level)

    if "--extra_encoder_options" in sys.argv:
        logger.warning(
            "--extra_encoder_options is deprecated, use --config instead.")

    explicit_cli_keys = collect_explicit_cli_keys(
        exclude=("extra_encoder_options", "config"))

    llm_args, _ = get_llm_args(
        model=model,
        max_batch_size=max_batch_size,
        max_num_tokens=max_num_tokens,
        gpus_per_node=gpus_per_node,
        trust_remote_code=trust_remote_code,
        revision=revision,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
        tensor_parallel_size=tensor_parallel_size,
        telemetry=telemetry,
        explicit_cli_keys=explicit_cli_keys)

    encoder_args_extra_dict = {}
    if extra_encoder_options is not None:
        with open(extra_encoder_options, 'r') as f:
            encoder_args_extra_dict = yaml.safe_load(f)
        if encoder_args_extra_dict is None:
            encoder_args_extra_dict = {}
        elif not isinstance(encoder_args_extra_dict, dict):
            raise ConfigInputError("Configuration file root must be a mapping.")
    _command_telemetry.apply_raw_config_telemetry_opt_out(
        encoder_args_extra_dict,
        usage_context=_telemetry_config.UsageContext.CLI_SERVE,
        component="server",
        explicit_cli_telemetry="telemetry" in explicit_cli_keys,
    )
    extra_allow_request_chat_template = _pop_bool_config_option(
        encoder_args_extra_dict, "allow_request_chat_template")
    allow_request_chat_template = (allow_request_chat_template
                                   or extra_allow_request_chat_template)
    encoder_args = update_llm_args_with_extra_dict(
        llm_args, encoder_args_extra_dict, explicit_cli_keys=explicit_cli_keys)

    _apply_effective_telemetry_config(encoder_args, telemetry=telemetry)

    metadata_server_cfg = parse_metadata_server_config_file(
        metadata_server_config_file)

    launch_mm_encoder_server(
        host,
        port,
        encoder_args,
        metadata_server_cfg,
        allow_request_chat_template=allow_request_chat_template)


@click.command("embeddings")
@click.argument("model", type=str)
@click.option("--host",
              type=str,
              default="localhost",
              help="Hostname of the server.")
@click.option("--port", type=int, default=8000, help="Port of the server.")
@click.option('--log_level',
              type=click.Choice(severity_map.keys()),
              default='info',
              help="The logging level.")
@click.option("--max_batch_size",
              type=int,
              default=_LLM_ARGS_FIELDS["max_batch_size"].default,
              help="Maximum batch size coalesced into a single encode() call.")
@click.option(
    "--max_num_tokens",
    type=int,
    default=8192,
    help="Maximum number of batched input tokens in each encode() call.")
@click.option(
    "--max_queue_delay",
    type=click.FloatRange(min=0.0),
    default=0.005,
    help="Dynamic-batching hold window in seconds: how long an incoming request "
    "waits for others to join its batch before being dispatched (mirrors Triton's "
    "max_queue_delay_microseconds).")
@click.option(
    "--max_queue_size",
    type=click.IntRange(min=1),
    default=2048,
    help="Maximum number of in-flight queued requests; further requests are "
    "rejected with HTTP 429 (mirrors Triton's max_queue_size).")
@click.option("--trust_remote_code",
              is_flag=True,
              default=False,
              help="Flag for HF transformers.")
@click.option(
    "--config",
    "--extra_llm_api_options",
    "extra_llm_api_options",
    type=str,
    default=None,
    help="Path to a YAML configuration file. Explicit CLI flags take precedence "
    "over values in this file.")
@click.option("--hf_revision",
              "--revision",
              "revision",
              type=str,
              default=None,
              help="The revision to use for the HuggingFace model "
              "(branch name, tag name, or commit id).")
@click.option("--metadata_server_config_file",
              type=str,
              default=None,
              help="Path to metadata server config file")
@click.option("--telemetry/--no-telemetry",
              default=True,
              help="Enable or disable anonymous usage telemetry collection.")
def serve_embedding(
    model: str,
    host: str,
    port: int,
    log_level: str,
    max_batch_size: int,
    max_num_tokens: int,
    max_queue_delay: float,
    max_queue_size: int,
    trust_remote_code: bool,
    extra_llm_api_options: Optional[str],
    revision: Optional[str],
    metadata_server_config_file: Optional[str],
    telemetry: bool,
):
    """Run an OpenAI-compatible /v1/embeddings server for encoder-only models.

    Coalesces concurrent requests with a dynamic batcher and serves them through
    the synchronous llm.encode() fast path (no KV cache / sampler / scheduler).
    Single-GPU only: the command does not expose tensor/pipeline parallelism.

    MODEL: model name | HF checkpoint path
    """
    logger.set_level(log_level)

    explicit_cli_keys = collect_explicit_cli_keys(exclude=("config", ))

    # Single-GPU, encode-only: tensor_parallel_size is fixed to 1 and not exposed as a
    # flag (the in-process encode path has no multi-GPU worker proxy). gpus_per_node is
    # auto-detected by get_llm_args; free_gpu_memory_fraction is omitted because the
    # encode path allocates no KV cache. All remain settable via --config if needed.
    llm_args, _ = get_llm_args(model=model,
                               max_batch_size=max_batch_size,
                               max_num_tokens=max_num_tokens,
                               trust_remote_code=trust_remote_code,
                               revision=revision,
                               tensor_parallel_size=1,
                               telemetry=telemetry,
                               explicit_cli_keys=explicit_cli_keys)

    extra_dict = {}
    if extra_llm_api_options is not None:
        with open(extra_llm_api_options, 'r') as f:
            extra_dict = yaml.safe_load(f)
        if extra_dict is None:
            extra_dict = {}
        elif not isinstance(extra_dict, dict):
            raise ConfigInputError("Configuration file root must be a mapping.")
    _command_telemetry.apply_raw_config_telemetry_opt_out(
        extra_dict,
        usage_context=_telemetry_config.UsageContext.CLI_SERVE,
        component="server",
        explicit_cli_telemetry="telemetry" in explicit_cli_keys,
    )
    llm_args = update_llm_args_with_extra_dict(
        llm_args, extra_dict, explicit_cli_keys=explicit_cli_keys)

    _apply_effective_telemetry_config(llm_args, telemetry=telemetry)

    # The CLI does not expose TP/PP/CP, but a --config YAML could still set them. Reject
    # that explicitly rather than hang: the in-process encode-only path cannot shard.
    effective_tp = llm_args.get("tensor_parallel_size") or 1
    effective_pp = llm_args.get("pipeline_parallel_size") or 1
    effective_cp = llm_args.get("context_parallel_size") or 1
    if effective_tp > 1 or effective_pp > 1 or effective_cp > 1:
        raise click.BadParameter(
            "The embeddings server is single-GPU only; multi-GPU (TP/PP/CP) is not "
            f"supported yet. Got tensor_parallel_size={effective_tp}, "
            f"pipeline_parallel_size={effective_pp}, "
            f"context_parallel_size={effective_cp} from --config.",
            param_hint="config")
    _reject_embedding_extra_model_inputs(llm_args.get("cuda_graph_config"))

    metadata_server_cfg = parse_metadata_server_config_file(
        metadata_server_config_file)

    launch_embedding_server(host, port, llm_args, max_queue_delay,
                            max_queue_size, metadata_server_cfg)


def _visual_gen_requested(args: Sequence[str]) -> bool:
    """Detect the explicit VisualGen flags of ``serve`` before Click parses them."""
    for arg in args:
        if arg == "--":
            break
        if arg.split("=", 1)[0] in ("--enable_visual_gen", "--visual_gen_args"):
            return True
    return False


class DefaultGroup(_command_telemetry.TelemetryGroup):
    """Custom Click group to allow default command behavior"""

    def resolve_command(self, ctx, args):
        # If the first argument is not a recognized subcommand, assume "serve"
        if args and args[0] not in self.commands:
            result = "serve", self.commands["serve"], args
        else:
            result = super().resolve_command(ctx, args)
        if result[0] == "serve" and _visual_gen_requested(result[2]):
            # Marked before Click parses the serve options, so a VisualGen
            # launch rejected by Click also reports no startup error category
            # or startup state.
            _mark_visual_gen()
        return result


main = DefaultGroup(
    telemetry_usage_context=_telemetry_config.UsageContext.CLI_SERVE,
    telemetry_component="server",
    commands={
        "serve": serve,
        "disaggregated": disaggregated,
        "disaggregated_mpi_worker": disaggregated_mpi_worker,
        "mm_embedding_serve": serve_encoder,
        "embeddings": serve_embedding,
        # The part of a Mooncake pool that outlives and is shared by every
        # server, and the report of what they collectively contributed to it.
        "mooncake_master": mooncake_master,
        "mooncake_pool_report": mooncake_pool_report,
    })

# Old import path of names that moved to
# tensorrt_llm.commands.serve_cli._disagg_fleet,
# tensorrt_llm.commands.serve_cli.disaggregated.
# They are re-exported lazily (PEP 562) so that reading one warns.
# Kept for callers outside the repository until a future release.
_LAZY_NAMES = {
    "DisaggWorkerEnvs": "tensorrt_llm.commands.serve_cli._disagg_fleet",
    "set_cuda_device": "tensorrt_llm.commands.serve_cli.disaggregated",
}


def __getattr__(name):
    module_name = _LAZY_NAMES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"{__name__}.{name} has moved to {module_name} "
        "and will be removed in a future release.",
        FutureWarning,
        stacklevel=2,
    )
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache: later access skips __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_NAMES))


if __name__ == "__main__":
    main()
