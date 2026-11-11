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

import gc
import os
import secrets
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

import click
import torch
import uvloop
from torch.cuda import device_count

from tensorrt_llm._utils import mpi_rank, set_prometheus_multiproc_dir
from tensorrt_llm.commands import _telemetry as _command_telemetry
from tensorrt_llm.commands._serve_stability import stability_option
from tensorrt_llm.commands.serve_cli._disagg_fleet import (
    DisaggLauncherEnvs, _serve_coordinator_and_fleet, _serve_disagg_fleet)
from tensorrt_llm.commands.serve_cli._net import (_bind_listener,
                                                  _listener_address_family,
                                                  _publish_bound_address)
from tensorrt_llm.executor.utils import LlmLauncherEnvs
from tensorrt_llm.llmapi.disagg_utils import (parse_disagg_config_file,
                                              parse_metadata_server_config_file)
from tensorrt_llm.llmapi.mpi_session import find_free_ipc_addr
from tensorrt_llm.logger import logger, severity_map
from tensorrt_llm.serve import OpenAIDisaggServer
from tensorrt_llm.usage import config as _telemetry_config
from tensorrt_llm.usage import set_lifecycle_phase, set_usage_context

# Popen for the disaggregated proxy's process guard.
_child_p_global: Optional[subprocess.Popen] = None


@click.command("disaggregated")
@stability_option(
    "-c",
    "--config",
    "--config_file",
    "config_file",
    type=str,
    default=None,
    help="Path to the disaggregated serving configuration YAML file.",
    status="beta")
@stability_option("-m",
                  "--metadata_server_config_file",
                  type=str,
                  default=None,
                  help="Path to metadata server config file",
                  status="prototype")
@stability_option("-t",
                  "--server_start_timeout",
                  type=int,
                  default=180,
                  help="Server start timeout",
                  status="beta")
@stability_option("-r",
                  "--request_timeout",
                  type=int,
                  default=180,
                  help="Request timeout",
                  status="beta")
@stability_option("-l",
                  '--log_level',
                  type=click.Choice(severity_map.keys()),
                  default='info',
                  help="The logging level.",
                  status="beta")
@stability_option("-s",
                  "--schedule_style",
                  type=click.Choice(["context_first", "generation_first"],
                                    case_sensitive=False),
                  default=None,
                  help="The schedule style for the disaggregated server.",
                  status="beta")
@stability_option(
    "--metrics-log-interval",
    type=int,
    default=0,
    help="[Deprecated] The interval of logging metrics in seconds. "
    "This option is not connected to any functionality and will be removed in a future release.",
    status="deprecated")
@stability_option(
    "--telemetry/--no-telemetry",
    default=True,
    help="Enable or disable anonymous usage telemetry collection.",
    status="beta")
@stability_option(
    "--report_addr",
    type=str,
    default=None,
    help="Write the host:port this server actually bound to this file, "
    "atomically, once the socket is bound. Lets the config set port 0 and "
    "have the launcher read the kernel-assigned port back instead of "
    "reserving one up front.",
    status="prototype")
def disaggregated(
    config_file: Optional[str],
    metadata_server_config_file: Optional[str],
    server_start_timeout: int,
    request_timeout: int,
    log_level: str,
    metrics_log_interval: int,
    schedule_style: str,
    telemetry: bool,
    report_addr: Optional[str],
):
    """Running server in disaggregated mode"""

    set_usage_context(_telemetry_config.UsageContext.DISAGGREGATED.value)
    logger.set_level(log_level)
    set_prometheus_multiproc_dir()

    if metrics_log_interval != 0:
        logger.warning(
            "--metrics-log-interval is deprecated and not connected to any "
            "functionality. This option will be removed in a future release.")

    if "--config_file" in sys.argv:
        logger.warning("--config_file is deprecated, use --config instead.")

    _command_telemetry.apply_disaggregated_telemetry_config(
        config_file,
        telemetry=telemetry,
    )
    disagg_cfg = parse_disagg_config_file(config_file)
    if schedule_style:
        disagg_cfg.schedule_style = schedule_style

    # Generate a shared deployment ID for all workers in this disagg deployment.
    # Inherited by child processes via env var; used for deduplication at query time.
    os.environ[DisaggLauncherEnvs.TLLM_DISAGG_DEPLOYMENT_ID] = uuid.uuid4().hex

    logger.info(f"Reserving disaggregated server address "
                f"{disagg_cfg.hostname}:{disagg_cfg.port} (pid={os.getpid()})")
    metadata_server_cfg = parse_metadata_server_config_file(
        metadata_server_config_file)

    # Topology from explicit config (num_workers + disagg_coordinator_url):
    # (a) url set -> delegate to that external coordinator; (b) url absent,
    # num_workers>1 -> implicit in-process coordinator + delegating fleet;
    # (c) url absent, num_workers==1 -> single self-contained server.
    num_workers = disagg_cfg.num_workers
    coordinator_url = disagg_cfg.disagg_coordinator_url
    os.environ.pop(DisaggLauncherEnvs.TLLM_DISAGG_ROLE, None)

    # Only topology (c) below binds the public socket in this process. The fleet
    # paths hand the port to N SO_REUSEPORT workers, which with port 0 would each
    # get a *different* kernel-assigned port instead of sharing one, so reject
    # the combination rather than publishing an address that serves 1/N requests.
    if (disagg_cfg.port == 0 or report_addr) and (coordinator_url
                                                  or num_workers > 1):
        raise click.BadParameter(
            "port 0 and --report_addr are only supported for a single "
            f"self-contained disaggregated server, but num_workers={num_workers} "
            f"and disagg_coordinator_url={coordinator_url!r} select a fleet.")

    if coordinator_url:
        # (a) External coordinator: fork a fleet of delegating servers (or a
        # single one) pointed at it; never start a coordinator in this process.
        _serve_disagg_fleet(disagg_cfg, config_file,
                            metadata_server_config_file, request_timeout,
                            server_start_timeout, num_workers, coordinator_url)
        return

    if num_workers > 1:
        # (b) Implicit coordinator in this process (on port-1) + a delegating
        # uvicorn fleet (workers=N) on the public port. See below.
        os.environ[DisaggLauncherEnvs.TLLM_DISAGG_ROLE] = "coordinator"
        _serve_coordinator_and_fleet(disagg_cfg, config_file,
                                     metadata_server_config_file,
                                     metadata_server_cfg, request_timeout,
                                     server_start_timeout, num_workers)
        return

    # (c) num_workers==1, no external coordinator: a single disagg server with an
    # in-process (local) coordinator. Pre-bind the socket (validates port), serve.
    os.environ[DisaggLauncherEnvs.TLLM_DISAGG_ROLE] = "server_coordinator"
    address_family = _listener_address_family("disaggregated server",
                                              disagg_cfg.hostname,
                                              disagg_cfg.port,
                                              policy="ipv4")
    with _bind_listener("disaggregated server",
                        disagg_cfg.hostname,
                        disagg_cfg.port,
                        family=address_family) as s:
        if disagg_cfg.port == 0:
            disagg_cfg.port = s.getsockname()[1]

        _publish_bound_address(report_addr, disagg_cfg.hostname,
                               disagg_cfg.port)

        server = OpenAIDisaggServer(
            config=disagg_cfg,
            req_timeout_secs=request_timeout,
            server_start_timeout_secs=server_start_timeout,
            metadata_server_cfg=metadata_server_cfg,
            metrics_interval_secs=metrics_log_interval)

        # Disable GC by default
        #   When concurrency is high, the number of Python objects increases, so
        #   GC runs frequently and takes a long time to process. In this case,
        #   requests are not immediately forwarded to CTX workers and GEN workers,
        #   causing them to run with small batch sizes. Disabling GC can mitigate
        #   this problem.
        #   By testing this feature, we didn't observe significant RSS or VMS
        #   increment, and observed that `count0` (obtained by `gc.get_count()`)
        #   increases by fewer than 1,000 after every 200,000 requests, while the
        #   maximum value of `count0` exceeded 3,000,000 during the test.
        if os.getenv("TRTLLM_DISAGG_SERVER_DISABLE_GC", "1") == "1":
            gc.disable()

        set_lifecycle_phase("serving")
        uvloop.run(server(disagg_cfg.hostname, disagg_cfg.port, sockets=[s]))


def set_cuda_device():
    if (os.getenv("OMPI_COMM_WORLD_RANK")):
        env_global_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    elif (os.getenv("SLURM_PROCID")):
        env_global_rank = int(os.environ["SLURM_PROCID"])
    else:
        raise RuntimeError("Could not determine rank from environment")
    device_id = env_global_rank % device_count()
    print(
        f"env_global_rank: {env_global_rank}, set device_id: {device_id} before importing mpi4py"
    )
    torch.cuda.set_device(device_id)


def _shared_draft_kv_blocker(server_configs) -> Optional[str]:
    """What keeps the context and generation servers from sharing draft KV, or None."""
    with_spec = {
        cfg.type
        for cfg in server_configs if cfg.other_args.get("speculative_config")
    }
    if with_spec != {"ctx", "gen"}:
        return ("speculative_config in both context_servers and "
                "generation_servers")
    if len({(cfg.other_args.get("tensor_parallel_size", 1),
             bool(cfg.other_args.get("enable_attention_dp", False)))
            for cfg in server_configs}) != 1:
        return ("the same tensor_parallel_size and enable_attention_dp in "
                "context_servers and generation_servers")
    for cfg in server_configs:
        spec = cfg.other_args["speculative_config"]
        backend = spec.get("attention_backend")
        if not ((spec.get("decoding_type") == "DFlash" and backend == "TRTLLM")
                or (spec.get("decoding_type") == "DSpark"
                    and backend in (None, "AUTO", "TRTLLM"))):
            return ("speculative_config to be DFlash with attention_backend "
                    "TRTLLM, or DSpark with attention_backend AUTO or TRTLLM, "
                    "in both sections")
        if (spec.get("draft_len_schedule") is not None
                or spec.get("max_concurrency") is not None):
            return "draft_len_schedule and max_concurrency to be unset in both sections"
        if cfg.other_args.get("attn_backend", "TRTLLM") != "TRTLLM":
            return "attn_backend TRTLLM in both sections"
        if (cfg.other_args.get("pipeline_parallel_size", 1) != 1
                or cfg.other_args.get("context_parallel_size", 1) != 1):
            return "pipeline_parallel_size and context_parallel_size to be 1 in both sections"
        if cfg.other_args.get("kv_cache_compression_config") is not None:
            return "kv_cache_compression_config to be unset in both sections"
        kv_cache = cfg.other_args.get("kv_cache_config") or {}
        if str(kv_cache.get("dtype", "auto")).lower().startswith("mixed_"):
            return "a uniform KV cache dtype in both sections"
        if kv_cache.get("tokens_per_block", 32) not in (16, 32, 64):
            return "kv_cache_config.tokens_per_block to be 16, 32, or 64 in both sections"
    return None


@click.command("disaggregated_mpi_worker")
@stability_option(
    "-c",
    "--config",
    "--config_file",
    "config_file",
    type=str,
    default=None,
    help="Path to the disaggregated serving configuration YAML file.",
    status="beta")
@stability_option('--log_level',
                  type=click.Choice(severity_map.keys()),
                  default='info',
                  help="The logging level.",
                  status="beta")
@stability_option(
    "--telemetry/--no-telemetry",
    default=True,
    help="Enable or disable anonymous usage telemetry collection.",
    status="beta")
def disaggregated_mpi_worker(config_file: Optional[str], log_level: str,
                             telemetry: bool):
    """Launching disaggregated MPI worker"""

    # Removed when the remaining serve commands move to commands/serve_cli/.
    from tensorrt_llm.commands.serve import (
        _build_llm_args_from_disagg_server_cfg,
        _refuse_dense_dwdp_on_generation_servers)

    _command_telemetry.apply_disaggregated_telemetry_config(
        config_file,
        telemetry=telemetry,
    )
    disagg_cfg = parse_disagg_config_file(config_file)
    # The servers launched here run a single frontend each, so they bind no
    # affinity ports for the disaggregated server to send conversations to.
    if disagg_cfg.ctx_frontend_affinity is not None:
        raise ValueError(
            "disaggregated_mpi_worker runs one serving frontend per server; "
            "remove num_serve_frontends and serve_frontend_affinity_port_stride "
            "from context_servers, or launch the context servers with trtllm-serve."
        )
    _refuse_dense_dwdp_on_generation_servers(disagg_cfg)

    from tensorrt_llm._utils import mpi_rank
    if os.environ.get(DisaggLauncherEnvs.
                      TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT) != "1":
        set_cuda_device()
    # Importing mpi4py after setting CUDA device. This is needed to work around an issue with mpi4py and CUDA
    from mpi4py.futures import MPICommExecutor

    from tensorrt_llm._utils import global_mpi_rank, set_mpi_comm
    from tensorrt_llm.llmapi.disagg_utils import split_world_comm

    blocker = _shared_draft_kv_blocker(disagg_cfg.server_configs)
    if blocker is not None and os.environ.get(
            "TRTLLM_DISAGG_SHARED_DRAFT_KV") == "1":
        raise click.UsageError(
            f"TRTLLM_DISAGG_SHARED_DRAFT_KV=1 requires {blocker}.")
    if os.environ.get("TRTLLM_DISAGG_SHARED_DRAFT_KV") in (None, "auto"):
        os.environ["TRTLLM_DISAGG_SHARED_DRAFT_KV"] = ("0"
                                                       if blocker else "auto")

    # Run a server with the underlying LLM invokes a RemoteMPISessionClient
    if os.environ.get(DisaggLauncherEnvs.
                      TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT) == "1":
        instance_idx = os.environ.get(
            DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX)
        server_cfg = disagg_cfg.server_configs[int(instance_idx)]

        llm_args = _build_llm_args_from_disagg_server_cfg(server_cfg.other_args)

        # Ignore the non-LLM args
        llm_args.pop("router", None)
        _launch_disaggregated_server(config_file, llm_args)
        return

    is_leader, instance_idx, sub_comm = split_world_comm(
        disagg_cfg.server_configs)

    logger.set_level(log_level)
    set_mpi_comm(sub_comm)
    logger.info(
        f"mpi_session is provided for LLM instance. Global MPI rank: {global_mpi_rank()}, sub-comm MPI rank: {mpi_rank()}"
    )

    # Leader ranks will start the trtllm-server using its own server config
    # and start a RemoteMPISessionServer to accept MPI tasks
    if is_leader:
        os.environ[DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX] = str(
            instance_idx)
        server_cfg = disagg_cfg.server_configs[instance_idx]

        # NOTE: the resulting llm_args is currently unused; _launch_disaggregated_leader
        # does not take it. Keeping the call symmetric with the client branch above
        # so any future use of llm_args here behaves the same way.
        _build_llm_args_from_disagg_server_cfg(server_cfg.other_args)

        _launch_disaggregated_leader(sub_comm, instance_idx, config_file,
                                     log_level)

    else:
        # Common workers
        with MPICommExecutor(sub_comm) as executor:
            if not is_leader and executor is not None:
                raise RuntimeError(
                    f"rank{global_mpi_rank()} should not have executor")


def _launch_disaggregated_server(disagg_config_file: str, llm_args: dict):
    # Removed when the remaining serve commands move to commands/serve_cli/.
    from tensorrt_llm.commands.serve import _DISAGG_SERVER_ROLES, launch_server

    # Launching the server
    instance_idx = os.environ.get(DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX)
    assert instance_idx is not None, f"{DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX} should be set by the launcher"
    disagg_config = parse_disagg_config_file(disagg_config_file)
    server_cfg = disagg_config.server_configs[int(instance_idx)]

    # Set disagg role for telemetry (server_cfg.type is 'ctx' or 'gen')
    os.environ[DisaggLauncherEnvs.TLLM_DISAGG_ROLE] = _DISAGG_SERVER_ROLES.get(
        server_cfg.type, "")

    logger.info(
        f"rank {mpi_rank()} for index {instance_idx} launch the disagg server")

    launch_kwargs = {
        "allow_request_chat_template":
        disagg_config.allow_request_chat_template,
        # Disagg ctx/gen MPI workers must not enter multi-frontend mode.
        "multi_frontend_enabled": False,
    }
    internal_disagg_auth_key = getattr(disagg_config,
                                       "internal_request_auth_key", None)
    if internal_disagg_auth_key is not None:
        launch_kwargs["internal_disagg_auth_key"] = internal_disagg_auth_key

    launch_server(host=server_cfg.hostname,
                  port=server_cfg.port,
                  llm_args=llm_args,
                  **launch_kwargs)


def _launch_disaggregated_leader(sub_comm, instance_idx: int, config_file: str,
                                 log_level: str):
    global _child_p_global  # Declare usage of global variable
    # Assuming logger and mpi_rank are available from module imports or passed in
    from tensorrt_llm._utils import mpi_rank
    from tensorrt_llm.llmapi.mgmn_leader_node import \
        launch_server_main as launch_remote_mpi_session_server
    from tensorrt_llm.llmapi.mpi_session import split_mpi_env

    # This mimics the behavior of trtllm-llmapi-launch
    # TODO: Make the port allocation atomic
    free_ipc_addr = find_free_ipc_addr()
    os.environ[LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS] = "1"
    os.environ[
        LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR.value] = free_ipc_addr
    os.environ[LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY.
               value] = secrets.token_hex(32)
    os.environ[DisaggLauncherEnvs.TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT.
               value] = "1"
    os.environ[DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX] = str(instance_idx)

    logger.debug(
        f"proxy controller address: {os.environ[LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR]}"
    )

    # The MPI-related environment variables will invoke duplicate MPI_Init in
    # the forked process, so we need to remove them before launching the server
    # process.
    non_mpi_env, mpi_env = split_mpi_env()

    assert LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS in non_mpi_env
    assert LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR in non_mpi_env
    assert LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY in non_mpi_env
    assert DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX in non_mpi_env
    assert DisaggLauncherEnvs.TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT in non_mpi_env

    # Two steps:
    # 1. Run the LLM-API Proxy in a separate process for streaming performance.
    #      The Proxy will create a RemoteMpiSessionClient as mpi_session in LLM
    #      class.
    guard_path = Path(
        __file__).resolve().parents[1] / "llmapi" / "_llmapi_process_guard.py"
    command = [
        sys.executable, "-S",
        str(guard_path), "--parent-pid",
        str(os.getpid()), "--term-grace", "5", "--autonomous", "--role",
        "proxy", "--", "python3", sys.argv[0], "disaggregated_mpi_worker", "-c",
        config_file, "--log_level", log_level
    ]
    logger.info(
        f"rank {mpi_rank()} step1: preparing to launch command: {command}")

    # Store original signal handlers
    original_sigterm_handler = signal.getsignal(signal.SIGTERM)
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    # Register new signal handlers
    signal.signal(signal.SIGTERM, _command_telemetry.raise_signal_exit)
    signal.signal(signal.SIGINT, _command_telemetry.raise_signal_exit)

    _child_p_global = None
    failure_pending = False
    try:
        # MPI_Abort and SIGKILL bypass finally. The guard remains the owner of
        # the proxy group and reclaims it when this leader disappears.
        _child_p_global = subprocess.Popen(
            command,
            env=non_mpi_env,
            stdout=sys.stdout,  # Redirect to parent's stdout
            stderr=sys.stderr,  # Redirect to parent's stderr
            start_new_session=True)

        logger.info(
            f"Parent process (PID {os.getpid()}) launched proxy guard (PID {_child_p_global.pid})."
        )

        logger.info(f"rank {mpi_rank()} step2: start the mpi session server")
        # 2. Run the RemoteMpiSessionServer to accept MPI tasks
        assert sub_comm is not None
        assert sub_comm.Get_rank() == 0
        # This is a blocking call
        launch_remote_mpi_session_server(sub_comm)

    except BaseException:
        failure_pending = True
        raise
    finally:
        # Restore original signal handlers
        signal.signal(signal.SIGTERM, original_sigterm_handler)
        signal.signal(signal.SIGINT, original_sigint_handler)

        if _child_p_global:  # If Popen was successful and object exists
            logger.info(
                f"Parent process (PID {os.getpid()}) in finally block. Stopping proxy guard (PID: {_child_p_global.pid})."
            )
            # Check if child is still running
            if _child_p_global.poll() is None:
                _child_p_global.terminate()
                try:
                    _child_p_global.wait(timeout=30)
                except subprocess.TimeoutExpired as error:
                    message = (
                        f"Proxy guard {_child_p_global.pid} cleanup is "
                        "still pending after 30s; retaining its process-group ownership"
                    )
                    logger.error(message)
                    # Killing the guard could strand its payload. Leave it
                    # responsible for cleanup and preserve an existing failure.
                    if not failure_pending:
                        raise RuntimeError(message) from error

    # Check if the process was launched and assert it's terminated
    if _child_p_global and hasattr(_child_p_global,
                                   'pid') and _child_p_global.pid is not None:
        final_status = _child_p_global.poll()
        assert final_status is not None, \
            f"The subprocess (PID {_child_p_global.pid}) should be terminated, but its status is {final_status}"
        logger.info(
            f"Subprocess (PID {_child_p_global.pid}) final status: {final_status}"
        )
    elif _child_p_global is None:
        # This implies Popen might have failed or was not reached.
        # If Popen failed, an exception would likely have occurred earlier.
        logger.info(
            "Child process was not assigned to _child_p_global, skipping final termination assertion."
        )
