import asyncio
import os
from typing import Any, List, Optional

import click
import torch
import yaml
from torch.cuda import device_count
from transformers import AutoTokenizer

from tensorrt_llm._torch.llm import LLM as PyTorchLLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.bindings.executor import (CapacitySchedulerPolicy,
                                            DynamicBatchConfig, SchedulerConfig)
from tensorrt_llm.llmapi import LLM, BuildConfig, KvCacheConfig
from tensorrt_llm.llmapi.disagg_utils import (CtxGenServerConfig,
                                              parse_disagg_config_file)
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_dict
from tensorrt_llm.logger import logger, severity_map
from tensorrt_llm.serve import OpenAIDisaggServer, OpenAIServer


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
                 **llm_args_dict: Any):

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

    pytorch_backend_config = PyTorchConfig(
        enable_overlap_scheduler=True) if backend == "pytorch" else None
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
        "kv_cache_config": kv_cache_config,
        "backend": backend if backend == "pytorch" else None,
        "pytorch_backend_config": pytorch_backend_config,
        "_num_postprocess_workers": num_postprocess_workers,
        "_postprocess_tokenizer_dir": tokenizer or model,
    }

    llm_args = update_llm_args_with_extra_dict(llm_args, llm_args_dict)

    return llm_args


def launch_server(host: str, port: int, llm_args: dict):

    backend = llm_args["backend"]
    model = llm_args["model"]
    tokenizer = llm_args["tokenizer"]

    if backend == 'pytorch':
        llm = PyTorchLLM(**llm_args)
    else:
        llm = LLM(**llm_args)

    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer or model)

    server = OpenAIServer(llm=llm, model=model, hf_tokenizer=hf_tokenizer)

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
def serve(model: str, tokenizer: Optional[str], host: str, port: int,
          log_level: str, backend: str, max_beam_width: int,
          max_batch_size: int, max_num_tokens: int, max_seq_len: int,
          tp_size: int, pp_size: int, ep_size: Optional[int],
          gpus_per_node: Optional[int],
          kv_cache_free_gpu_memory_fraction: float,
          num_postprocess_workers: int, trust_remote_code: bool,
          extra_llm_api_options: Optional[str]):
    """Running an OpenAI API compatible server

    MODEL: model name | HF checkpoint path | TensorRT engine path
    """
    logger.set_level(log_level)

    llm_args_dict = {}
    if extra_llm_api_options is not None:
        with open(extra_llm_api_options, 'r') as f:
            llm_args_dict = yaml.safe_load(f)

    llm_args = get_llm_args(
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
        gpus_per_node=gpus_per_node,
        free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
        num_postprocess_workers=num_postprocess_workers,
        trust_remote_code=trust_remote_code,
        **llm_args_dict)

    launch_server(host, port, llm_args)


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
def disaggregated(config_file: Optional[str], server_start_timeout: int,
                  request_timeout: int):
    """Running server in disaggregated mode"""

    disagg_cfg = parse_disagg_config_file(config_file)

    ctx_server_urls, gen_server_urls = get_ctx_gen_server_urls(
        disagg_cfg.server_configs)

    server = OpenAIDisaggServer(ctx_servers=ctx_server_urls,
                                gen_servers=gen_server_urls,
                                req_timeout_secs=request_timeout,
                                server_start_timeout_secs=server_start_timeout)

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
def disaggregated_mpi_worker(config_file: Optional[str]):
    """Launching disaggregated MPI worker"""

    set_cuda_device()
    # Importing mpi4py after setting CUDA device. This is needed to war an issue with mpi4py and CUDA
    from mpi4py.futures import MPICommExecutor

    from tensorrt_llm._utils import global_mpi_rank, mpi_rank, set_mpi_comm
    from tensorrt_llm.llmapi import MpiCommSession
    from tensorrt_llm.llmapi.disagg_utils import split_world_comm

    disagg_cfg = parse_disagg_config_file(config_file)

    is_leader, instance_idx, sub_comm = split_world_comm(
        disagg_cfg.server_configs)

    os.environ['TRTLLM_USE_MPI_KVCACHE'] = "1"
    set_mpi_comm(sub_comm)
    logger.info(
        f"mpi_session is provided for LLM instance. Global MPI rank: {global_mpi_rank()}, sub-comm MPI rank: {mpi_rank()}"
    )

    # Leader ranks will start the trtllm-server using it's own server config
    if is_leader:
        server_cfg = disagg_cfg.server_configs[instance_idx]

        llm_args = get_llm_args(**server_cfg.other_args)

        mpi_session = MpiCommSession(
            comm=sub_comm,
            n_workers=sub_comm.Get_size()) if sub_comm is not None else None

        llm_args["_mpi_session"] = mpi_session

        launch_server(host=server_cfg.hostname,
                      port=server_cfg.port,
                      llm_args=llm_args)
    else:
        with MPICommExecutor(sub_comm) as executor:
            if not is_leader and executor is not None:
                raise RuntimeError(
                    f"rank{global_mpi_rank()} should not have executor")


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
