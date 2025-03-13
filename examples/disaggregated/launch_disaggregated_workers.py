import argparse
import asyncio
import logging
import os
from typing import Any, Optional

from transformers import AutoTokenizer

from tensorrt_llm._torch.llm import LLM as PyTorchLLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm._utils import global_mpi_rank, mpi_rank, set_mpi_comm
from tensorrt_llm.bindings.executor import (CapacitySchedulerPolicy,
                                            DynamicBatchConfig, SchedulerConfig)
from tensorrt_llm.llmapi import LLM, BuildConfig, KvCacheConfig, MpiCommSession
from tensorrt_llm.llmapi.disagg_utils import (parse_disagg_config_file,
                                              split_world_comm)
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_dict
from tensorrt_llm.serve import OpenAIServer

logging.basicConfig(level=logging.INFO)

from mpi4py.futures import MPICommExecutor
from mpi4py.MPI import COMM_WORLD, Comm
from torch.cuda import device_count


#This script must be executed by all ranks
def launch_server(model: str,
                  hostname: str,
                  port: int,
                  sub_comm: Comm,
                  max_batch_size: int = BuildConfig.max_batch_size,
                  max_num_tokens: int = BuildConfig.max_num_tokens,
                  max_beam_width: int = BuildConfig.max_beam_width,
                  max_seq_len: int = BuildConfig.max_seq_len,
                  free_gpu_memory_fraction: Optional[float] = None,
                  tensor_parallel_size: int = 1,
                  pipeline_parallel_size: int = 1,
                  moe_expert_parallel_size: Optional[int] = None,
                  enable_attention_dp: bool = False,
                  backend: Optional[str] = None,
                  tokenizer: Optional[str] = None,
                  gpus_per_node: Optional[int] = None,
                  trust_remote_code: bool = False,
                  use_cuda_graph: bool = False,
                  enable_overlap_scheduler: bool = False,
                  **kwargs: Any):

    mpi_session = MpiCommSession(comm=sub_comm, n_workers=sub_comm.Get_size())

    if gpus_per_node is None:
        gpus_per_node = device_count()
        if gpus_per_node == 0:
            raise ValueError("No GPU devices found on the node")

    build_config = BuildConfig(max_batch_size=max_batch_size,
                               max_num_tokens=max_num_tokens,
                               max_beam_width=max_beam_width,
                               max_seq_len=max_seq_len)

    kv_cache_config = None
    if free_gpu_memory_fraction:
        kv_cache_config = KvCacheConfig(
            free_gpu_memory_fraction=free_gpu_memory_fraction)

    pytorch_backend_config = PyTorchConfig(
        enable_overlap_scheduler=enable_overlap_scheduler,
        use_cuda_graph=use_cuda_graph) if backend == "pytorch" else None

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
        "enable_attention_dp": enable_attention_dp,
        "gpus_per_node": gpus_per_node,
        "trust_remote_code": trust_remote_code,
        "build_config": build_config,
        "kv_cache_config": kv_cache_config,
        "backend": backend if backend == "pytorch" else None,
        "pytorch_backend_config": pytorch_backend_config,
        "_mpi_session": mpi_session,
    }

    llm_args = update_llm_args_with_extra_dict(llm_args, kwargs)

    if backend == 'pytorch':
        llm = PyTorchLLM(**llm_args)
    else:
        llm = LLM(**llm_args)

    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer or model)

    server = OpenAIServer(llm=llm, model=model, hf_tokenizer=hf_tokenizer)

    asyncio.run(server(hostname, port))


def main():
    parser = argparse.ArgumentParser(description="Launch disaggregated workers")
    parser.add_argument("-c",
                        "--disagg_config_file",
                        type=str,
                        required=True,
                        help="Path to the YAML configuration file")
    parser.add_argument("-t",
                        "--server_start_timeout",
                        type=int,
                        default=180,
                        help="Server start timeout")
    args = parser.parse_args()

    disagg_config = parse_disagg_config_file(args.disagg_config_file)

    is_leader, instance_idx, sub_comm = split_world_comm(
        disagg_config.server_configs)

    os.environ['TRTLLM_USE_MPI_KVCACHE'] = "1"
    set_mpi_comm(sub_comm)
    logging.info(
        f"mpi_session is provided for LLM instance. Global MPI rank: {global_mpi_rank()}, sub-comm MPI rank: {mpi_rank()}"
    )

    # Leader ranks will start the trtllm-server using it's own server config
    if is_leader:
        server_cfg = disagg_config.server_configs[instance_idx]

        launch_server(hostname=server_cfg.hostname,
                      port=server_cfg.port,
                      sub_comm=sub_comm,
                      **server_cfg.other_args)
    else:
        with MPICommExecutor(sub_comm) as executor:
            if not is_leader and executor is not None:
                raise RuntimeError(f"rank{COMM_WORLD} should not have executor")


if __name__ == "__main__":
    main()
