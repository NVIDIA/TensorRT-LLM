import asyncio
import os
import signal  # Added import
import subprocess  # nosec B404
import sys
from typing import Any, List, Optional

import click
import torch
import yaml
from strenum import StrEnum
from torch.cuda import device_count

from tensorrt_llm._torch.llm import LLM as PyTorchLLM
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.executor.utils import LlmLauncherEnvs
from tensorrt_llm.llmapi import (LLM, BuildConfig, CapacitySchedulerPolicy,
                                 DynamicBatchConfig, KvCacheConfig,
                                 SchedulerConfig)
from tensorrt_llm.llmapi.disagg_utils import (CtxGenServerConfig,
                                              MetadataServerConfig, ServerRole,
                                              parse_disagg_config_file,
                                              parse_metadata_server_config_file)
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_dict
from tensorrt_llm.llmapi.mpi_session import find_free_port
from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory
from tensorrt_llm.logger import logger, severity_map
from tensorrt_llm.serve import OpenAIDisaggServer, OpenAIServer

# Global variable to store the Popen object of the child process
_child_p_global: Optional[subprocess.Popen] = None


def _signal_handler_cleanup_child(signum, frame):
    """Signal handler to clean up the child process."""
    global _child_p_global
    if _child_p_global and _child_p_global.poll() is None:
        # Using print for safety in signal handlers
        logger.info(
            f"Parent process (PID {os.getpid()}) received signal {signal.Signals(signum).name}. Terminating child process (PID {_child_p_global.pid})."
        )
        _child_p_global.terminate()
        try:
            _child_p_global.wait(
                timeout=10)  # Allow 10 seconds for graceful termination
        except subprocess.TimeoutExpired:
            logger.info(
                f"Child process (PID {_child_p_global.pid}) did not terminate gracefully after signal. Killing."
            )
            _child_p_global.kill()
            try:
                _child_p_global.wait(timeout=10)  # Allow 10 seconds for kill
            except subprocess.TimeoutExpired:
                logger.info(
                    f"Child process (PID {_child_p_global.pid}) failed to die even after kill command from signal handler."
                )

        if _child_p_global.poll() is not None:
            logger.info(
                f"Child process (PID {_child_p_global.pid}) confirmed terminated due to signal {signal.Signals(signum).name}."
            )
        else:
            logger.info(
                f"Child process (PID {_child_p_global.pid}) is still running after cleanup attempt for signal {signal.Signals(signum).name}."
            )

    # Standard exit code for signal termination
    sys.exit(128 + signum)


def get_llm_args(model: str,
                 tokenizer: Optional[str] = None,
                 backend: Optional[str] = None,
                 max_beam_width: int = BuildConfig.max_beam_width,
                 max_batch_size: int = BuildConfig.max_batch_size,
                 max_num_tokens: int = BuildConfig.max_num_tokens,
                 max_seq_len: int = BuildConfig.max_seq_len,
                 tensor_parallel_size: int = 1,
                 pipeline_parallel_size: int = 1,
                 moe_expert_parallel_size: Optional[int] = None,
                 gpus_per_node: Optional[int] = None,
                 free_gpu_memory_fraction: Optional[float] = None,
                 num_postprocess_workers: int = 0,
                 trust_remote_code: bool = False,
                 reasoning_parser: Optional[str] = None,
                 **llm_args_extra_dict: Any):

    if gpus_per_node is None:
        gpus_per_node = device_count()
        if gpus_per_node == 0:
            raise ValueError("No GPU devices found on the node")
    build_config = BuildConfig(max_batch_size=max_batch_size,
                               max_num_tokens=max_num_tokens,
                               max_beam_width=max_beam_width,
                               max_seq_len=max_seq_len)
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=free_gpu_memory_fraction)

    dynamic_batch_config = DynamicBatchConfig(
        enable_batch_size_tuning=True,
        enable_max_num_tokens_tuning=False,
        dynamic_batch_moving_average_window=128)
    scheduler_config = SchedulerConfig(
        capacity_scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        dynamic_batch_config=dynamic_batch_config,
    )

    llm_args = {
        "model": model,
        "scheduler_config": scheduler_config,
        "tokenizer": tokenizer,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "moe_expert_parallel_size": moe_expert_parallel_size,
        "gpus_per_node": gpus_per_node,
        "trust_remote_code": trust_remote_code,
        "build_config": build_config,
        "max_batch_size": max_batch_size,
        "max_num_tokens": max_num_tokens,
        "max_beam_width": max_beam_width,
        "max_seq_len": max_seq_len,
        "kv_cache_config": kv_cache_config,
        "backend": backend if backend == "pytorch" else None,
        "_num_postprocess_workers": num_postprocess_workers,
        "_postprocess_tokenizer_dir": tokenizer or model,
        "_reasoning_parser": reasoning_parser,
    }

    return llm_args, llm_args_extra_dict


def launch_server(host: str,
                  port: int,
                  llm_args: dict,
                  metadata_server_cfg: Optional[MetadataServerConfig] = None,
                  server_role: Optional[ServerRole] = None):

    backend = llm_args["backend"]
    model = llm_args["model"]

    if backend == 'pytorch':
        llm = PyTorchLLM(**llm_args)
    else:
        llm = LLM(**llm_args)

    server = OpenAIServer(llm=llm,
                          model=model,
                          server_role=server_role,
                          metadata_server_cfg=metadata_server_cfg)

    asyncio.run(server(host, port))


@click.command("serve")
@click.argument("model", type=str)
@click.option("--tokenizer",
              type=str,
              default=None,
              help="Path | Name of the tokenizer."
              "Specify this value only if using TensorRT engine as model.")
@click.option("--host",
              type=str,
              default="localhost",
              help="Hostname of the server.")
@click.option("--port", type=int, default=8000, help="Port of the server.")
@click.option("--backend",
              type=click.Choice(["pytorch"]),
              default=None,
              help="Set to 'pytorch' for pytorch path. Default is cpp path.")
@click.option('--log_level',
              type=click.Choice(severity_map.keys()),
              default='info',
              help="The logging level.")
@click.option("--max_beam_width",
              type=int,
              default=BuildConfig.max_beam_width,
              help="Maximum number of beams for beam search decoding.")
@click.option("--max_batch_size",
              type=int,
              default=BuildConfig.max_batch_size,
              help="Maximum number of requests that the engine can schedule.")
@click.option(
    "--max_num_tokens",
    type=int,
    default=BuildConfig.max_num_tokens,
    help=
    "Maximum number of batched input tokens after padding is removed in each batch."
)
@click.option(
    "--max_seq_len",
    type=int,
    default=BuildConfig.max_seq_len,
    help="Maximum total length of one request, including prompt and outputs. "
    "If unspecified, the value is deduced from the model config.")
@click.option("--tp_size", type=int, default=1, help='Tensor parallelism size.')
@click.option("--pp_size",
              type=int,
              default=1,
              help='Pipeline parallelism size.')
@click.option("--ep_size",
              type=int,
              default=None,
              help="expert parallelism size")
@click.option("--cluster_size",
              type=int,
              default=None,
              help="expert cluster parallelism size")
@click.option("--gpus_per_node",
              type=int,
              default=None,
              help="Number of GPUs per node. Default to None, and it will be "
              "detected automatically.")
@click.option("--kv_cache_free_gpu_memory_fraction",
              type=float,
              default=0.9,
              help="Free GPU memory fraction reserved for KV Cache, "
              "after allocating model weights and buffers.")
@click.option(
    "--num_postprocess_workers",
    type=int,
    default=0,
    help="[Experimental] Number of workers to postprocess raw responses "
    "to comply with OpenAI protocol.")
@click.option("--trust_remote_code",
              is_flag=True,
              default=False,
              help="Flag for HF transformers.")
@click.option(
    "--extra_llm_api_options",
    type=str,
    default=None,
    help=
    "Path to a YAML file that overwrites the parameters specified by trtllm-serve."
)
@click.option(
    "--reasoning_parser",
    type=click.Choice(ReasoningParserFactory.parsers.keys()),
    default=None,
    help="[Experimental] Specify the parser for reasoning models.",
)
@click.option("--metadata_server_config_file",
              type=str,
              default=None,
              help="Path to metadata server config file")
@click.option(
    "--server_role",
    type=str,
    default=None,
    help="Server role. Specify this value only if running in disaggregated mode."
)
def serve(model: str, tokenizer: Optional[str], host: str, port: int,
          log_level: str, backend: str, max_beam_width: int,
          max_batch_size: int, max_num_tokens: int, max_seq_len: int,
          tp_size: int, pp_size: int, ep_size: Optional[int],
          cluster_size: Optional[int], gpus_per_node: Optional[int],
          kv_cache_free_gpu_memory_fraction: float,
          num_postprocess_workers: int, trust_remote_code: bool,
          extra_llm_api_options: Optional[str], reasoning_parser: Optional[str],
          metadata_server_config_file: Optional[str],
          server_role: Optional[str]):
    """Running an OpenAI API compatible server

    MODEL: model name | HF checkpoint path | TensorRT engine path
    """
    logger.set_level(log_level)

    llm_args, _ = get_llm_args(
        model=model,
        tokenizer=tokenizer,
        backend=backend,
        max_beam_width=max_beam_width,
        max_batch_size=max_batch_size,
        max_num_tokens=max_num_tokens,
        max_seq_len=max_seq_len,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        moe_expert_parallel_size=ep_size,
        moe_cluster_parallel_size=cluster_size,
        gpus_per_node=gpus_per_node,
        free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
        num_postprocess_workers=num_postprocess_workers,
        trust_remote_code=trust_remote_code,
        reasoning_parser=reasoning_parser)

    llm_args_extra_dict = {}
    if extra_llm_api_options is not None:
        with open(extra_llm_api_options, 'r') as f:
            llm_args_extra_dict = yaml.safe_load(f)
    llm_args = update_llm_args_with_extra_dict(llm_args, llm_args_extra_dict)

    metadata_server_cfg = parse_metadata_server_config_file(
        metadata_server_config_file)

    if metadata_server_cfg is not None:
        assert server_role is not None, "server_role is required when metadata_server_cfg is provided"
        try:
            server_role = ServerRole[server_role.upper()]
        except ValueError:
            raise ValueError(f"Invalid server role: {server_role}. " \
                             f"Must be one of: {', '.join([role.name for role in ServerRole])}")
    launch_server(host, port, llm_args, metadata_server_cfg, server_role)


def get_ctx_gen_server_urls(
        server_configs: List[CtxGenServerConfig]) -> List[str]:
    ctx_server_urls = []
    gen_server_urls = []
    for cfg in server_configs:
        if cfg.type == "ctx":
            ctx_server_urls.append(f"http://{cfg.hostname}:{cfg.port}")
        else:
            gen_server_urls.append(f"http://{cfg.hostname}:{cfg.port}")

    return ctx_server_urls, gen_server_urls


@click.command("disaggregated")
@click.option("-c",
              "--config_file",
              type=str,
              default=None,
              help="Specific option for disaggregated mode.")
@click.option("-m",
              "--metadata_server_config_file",
              type=str,
              default=None,
              help="Path to metadata server config file")
@click.option("-t",
              "--server_start_timeout",
              type=int,
              default=180,
              help="Server start timeout")
@click.option("-r",
              "--request_timeout",
              type=int,
              default=180,
              help="Request timeout")
@click.option("-l",
              '--log_level',
              type=click.Choice(severity_map.keys()),
              default='info',
              help="The logging level.")
def disaggregated(config_file: Optional[str],
                  metadata_server_config_file: Optional[str],
                  server_start_timeout: int, request_timeout: int,
                  log_level: str):
    """Running server in disaggregated mode"""

    logger.set_level(log_level)

    disagg_cfg = parse_disagg_config_file(config_file)

    ctx_server_urls, gen_server_urls = get_ctx_gen_server_urls(
        disagg_cfg.server_configs)

    metadata_server_cfg = parse_metadata_server_config_file(
        metadata_server_config_file)

    server = OpenAIDisaggServer(
        ctx_servers=ctx_server_urls,
        gen_servers=gen_server_urls,
        req_timeout_secs=request_timeout,
        server_start_timeout_secs=server_start_timeout,
        ctx_router_config=disagg_cfg.ctx_router_config,
        gen_router_config=disagg_cfg.gen_router_config,
        conditional_disagg_config=disagg_cfg.conditional_disagg_config,
        metadata_server_cfg=metadata_server_cfg)

    asyncio.run(server(disagg_cfg.hostname, disagg_cfg.port))


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


@click.command("disaggregated_mpi_worker")
@click.option("-c",
              "--config_file",
              type=str,
              default=None,
              help="Specific option for disaggregated mode.")
@click.option('--log_level',
              type=click.Choice(severity_map.keys()),
              default='info',
              help="The logging level.")
def disaggregated_mpi_worker(config_file: Optional[str], log_level: str):
    """Launching disaggregated MPI worker"""

    from tensorrt_llm._utils import mpi_rank
    if os.environ.get(DisaggLauncherEnvs.
                      TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT) != "1":
        set_cuda_device()
    # Importing mpi4py after setting CUDA device. This is needed to war an issue with mpi4py and CUDA
    from mpi4py.futures import MPICommExecutor

    from tensorrt_llm._utils import global_mpi_rank, mpi_rank, set_mpi_comm
    from tensorrt_llm.llmapi.disagg_utils import split_world_comm

    disagg_cfg = parse_disagg_config_file(config_file)

    # Run a server with the underlying LLM invokes a RemoteMPISessionClient
    if os.environ.get(DisaggLauncherEnvs.
                      TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT) == "1":
        instance_idx = os.environ.get(
            DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX)
        server_cfg = disagg_cfg.server_configs[int(instance_idx)]

        llm_args, llm_args_extra_dict = get_llm_args(**server_cfg.other_args)
        llm_args = update_llm_args_with_extra_dict(llm_args,
                                                   llm_args_extra_dict)

        _launch_disaggregated_server(config_file, llm_args)
        return

    is_leader, instance_idx, sub_comm = split_world_comm(
        disagg_cfg.server_configs)

    logger.set_level(log_level)
    os.environ['TRTLLM_USE_MPI_KVCACHE'] = "1"
    set_mpi_comm(sub_comm)
    logger.info(
        f"mpi_session is provided for LLM instance. Global MPI rank: {global_mpi_rank()}, sub-comm MPI rank: {mpi_rank()}"
    )

    # Leader ranks will start the trtllm-server using it's own server config
    # and start a RemoteMPISessionServer to accept MPI tasks
    if is_leader:
        os.environ[DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX] = str(
            instance_idx)
        server_cfg = disagg_cfg.server_configs[instance_idx]

        llm_args, llm_args_extra_dict = get_llm_args(**server_cfg.other_args)
        llm_args = update_llm_args_with_extra_dict(llm_args,
                                                   llm_args_extra_dict)

        _launch_disaggregated_leader(sub_comm, instance_idx, config_file,
                                     log_level)

    else:
        # Common workers
        with MPICommExecutor(sub_comm) as executor:
            if not is_leader and executor is not None:
                raise RuntimeError(
                    f"rank{global_mpi_rank()} should not have executor")


class DisaggLauncherEnvs(StrEnum):
    TLLM_DISAGG_INSTANCE_IDX = "TLLM_DISAGG_INSTANCE_IDX"
    TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT = "TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT"


def _launch_disaggregated_server(disagg_config_file: str, llm_args: dict):
    # Launching the server
    instance_idx = os.environ.get(DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX)
    assert instance_idx is not None, f"{DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX} should be set by the launcher"
    disagg_config = parse_disagg_config_file(disagg_config_file)
    server_cfg = disagg_config.server_configs[int(instance_idx)]

    logger.info(
        f"rank {mpi_rank()} for index {instance_idx} launch the disagg server")

    launch_server(host=server_cfg.hostname,
                  port=server_cfg.port,
                  llm_args=llm_args)


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
    free_port = find_free_port()
    os.environ[LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS] = "1"
    os.environ[LlmLauncherEnvs.TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR.
               value] = f"tcp://127.0.0.1:{free_port}"
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
    assert DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX in non_mpi_env
    assert DisaggLauncherEnvs.TLLM_DISAGG_RUN_REMOTE_MPI_SESSION_CLIENT in non_mpi_env

    # Two steps:
    # 1. Run the LLM-API Proxy in a separate process for streaming performance.
    #      The Proxy will create a RemoteMpiSessionClient as mpi_session in LLM
    #      class.
    command = [
        "python3", sys.argv[0], "disaggregated_mpi_worker", "-c", config_file,
        "--log_level", log_level
    ]
    logger.info(
        f"rank {mpi_rank()} step1: preparing to launch command: {command}")

    # Store original signal handlers
    original_sigterm_handler = signal.getsignal(signal.SIGTERM)
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    # Register new signal handlers
    signal.signal(signal.SIGTERM, _signal_handler_cleanup_child)
    signal.signal(signal.SIGINT, _signal_handler_cleanup_child)

    try:
        _child_p_global = subprocess.Popen(
            command,
            env=non_mpi_env,
            stdout=sys.stdout,  # Redirect to parent's stdout
            stderr=sys.stderr,  # Redirect to parent's stderr
            start_new_session=True)

        logger.info(
            f"Parent process (PID {os.getpid()}) launched child process (PID {_child_p_global.pid})."
        )

        logger.info(f"rank {mpi_rank()} step2: start the mpi session server")
        # 2. Run the RemoteMpiSessionServer to accept MPI tasks
        assert sub_comm is not None
        assert sub_comm.Get_rank() == 0
        # This is a blocking call
        launch_remote_mpi_session_server(sub_comm)

    finally:
        # Restore original signal handlers
        signal.signal(signal.SIGTERM, original_sigterm_handler)
        signal.signal(signal.SIGINT, original_sigint_handler)

        if _child_p_global:  # If Popen was successful and object exists
            logger.info(
                f"Parent process (PID {os.getpid()}) in finally block. Cleaning up child process (PID: {_child_p_global.pid})."
            )
            # Check if child is still running
            if _child_p_global.poll() is None:
                _child_p_global.terminate()
                try:
                    _child_p_global.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"Child process {_child_p_global.pid} timed out on terminate (30s), killing."
                    )
                    _child_p_global.kill()
                    try:
                        _child_p_global.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        logger.error(
                            f"Child process {_child_p_global.pid} failed to be killed even after 30s."
                        )
            assert _child_p_global.poll(
            ) is not None, f"the subprocess should be terminated"

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


class DefaultGroup(click.Group):
    """Custom Click group to allow default command behavior"""

    def resolve_command(self, ctx, args):
        # If the first argument is not a recognized subcommand, assume "serve"
        if args and args[0] not in self.commands:
            return "serve", self.commands["serve"], args
        return super().resolve_command(ctx, args)


main = DefaultGroup(
    commands={
        "serve": serve,
        "disaggregated": disaggregated,
        "disaggregated_mpi_worker": disaggregated_mpi_worker
    })

if __name__ == "__main__":
    main()
