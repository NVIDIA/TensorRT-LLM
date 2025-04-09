# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional

import click

import tensorrt_llm.profiler as profiler

from .._torch.llm import LLM as PyTorchLLM
from .._torch.pyexecutor.config import PyTorchConfig
from ..evaluate import MMLU, CnnDailymail
from ..llmapi import LLM, BuildConfig, KvCacheConfig
from ..llmapi.llm_utils import update_llm_args_with_extra_options
from ..logger import logger, severity_map


@click.group()
@click.option(
    "--model",
    required=True,
    type=str,
    help="model name | HF checkpoint path | TensorRT engine path",
)
@click.option("--tokenizer",
              type=str,
              default=None,
              help="Path | Name of the tokenizer."
              "Specify this value only if using TensorRT engine as model.")
@click.option("--backend",
              type=click.Choice(["pytorch", "tensorrt"]),
              default="pytorch",
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
@click.option("--trust_remote_code",
              is_flag=True,
              default=False,
              help="Flag for HF transformers.")
@click.option("--extra_llm_api_options",
              type=str,
              default=None,
              help="Path to a YAML file that overwrites the parameters")
@click.pass_context
def main(ctx, model: str, tokenizer: Optional[str], log_level: str,
         backend: str, max_beam_width: int, max_batch_size: int,
         max_num_tokens: int, max_seq_len: int, tp_size: int, pp_size: int,
         ep_size: Optional[int], gpus_per_node: Optional[int],
         kv_cache_free_gpu_memory_fraction: float, trust_remote_code: bool,
         extra_llm_api_options: Optional[str]):
    logger.set_level(log_level)
    build_config = BuildConfig(max_batch_size=max_batch_size,
                               max_num_tokens=max_num_tokens,
                               max_beam_width=max_beam_width,
                               max_seq_len=max_seq_len)

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction)

    if backend == "tensorrt":
        backend = None
    pytorch_backend_config = None
    if backend == "pytorch":
        pytorch_backend_config = PyTorchConfig(enable_overlap_scheduler=True)

    llm_args = {
        "model": model,
        "tokenizer": tokenizer,
        "tensor_parallel_size": tp_size,
        "pipeline_parallel_size": pp_size,
        "moe_expert_parallel_size": ep_size,
        "gpus_per_node": gpus_per_node,
        "trust_remote_code": trust_remote_code,
        "build_config": build_config,
        "kv_cache_config": kv_cache_config,
        "backend": backend,
        "pytorch_backend_config": pytorch_backend_config,
    }

    if extra_llm_api_options is not None:
        llm_args = update_llm_args_with_extra_options(llm_args,
                                                      extra_llm_api_options)

    profiler.start("trtllm init")
    if backend == 'pytorch':
        llm = PyTorchLLM(**llm_args)
    else:
        llm = LLM(**llm_args)
    profiler.stop("trtllm init")
    elapsed_time = profiler.elapsed_time_in_sec("trtllm init")
    logger.info(f"TRTLLM initialization time: {elapsed_time:.3f} seconds.")

    # Pass llm to subcommands
    ctx.obj = llm


main.add_command(CnnDailymail.command)
main.add_command(MMLU.command)

if __name__ == "__main__":
    main()
