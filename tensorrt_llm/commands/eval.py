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
import click

import tensorrt_llm.profiler as profiler

from .. import LLM as PyTorchLLM
from .._tensorrt_engine import LLM
from ..commands.common_llm_options import common_llm_options
from ..evaluate import (GSM8K, MMLU, MMMU, CnnDailymail, GPQADiamond,
                        GPQAExtended, GPQAMain, JsonModeEval)
from ..llmapi import BuildConfig, KvCacheConfig
from ..llmapi.llm_utils import update_llm_args_with_extra_options
from ..logger import logger, severity_map


@click.group()
@click.option(
    "--model",
    required=True,
    type=str,
    help="model name | HF checkpoint path | TensorRT engine path",
)
@click.option('--log_level',
              type=click.Choice(severity_map.keys()),
              default='info',
              help="The logging level.")
@common_llm_options
@click.pass_context
def main(ctx, model: str, log_level: str, **params):
    logger.set_level(log_level)
    # TODO: unify LlmArgs parsing via Pydantic
    build_config = BuildConfig(max_batch_size=params.get("max_batch_size"),
                               max_num_tokens=params.get("max_num_tokens"),
                               max_beam_width=params.get("max_beam_width"),
                               max_seq_len=params.get("max_seq_len"))

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=params.get("kv_cache_free_gpu_memory_fraction",
                                            0.9),
        enable_block_reuse=not params.get("disable_kv_cache_reuse", False))

    llm_args = {
        "model": model,
        "tokenizer": params.get("tokenizer"),
        "tensor_parallel_size": params.get("tensor_parallel_size", 1),
        "pipeline_parallel_size": params.get("pipeline_parallel_size", 1),
        "moe_expert_parallel_size": params.get("moe_expert_parallel_size"),
        "gpus_per_node": params.get("gpus_per_node"),
        "trust_remote_code": params.get("trust_remote_code", False),
        "build_config": build_config,
        "kv_cache_config": kv_cache_config,
    }

    extra_llm_api_options = params.get("extra_llm_api_options")
    if extra_llm_api_options is not None:
        llm_args = update_llm_args_with_extra_options(llm_args,
                                                      extra_llm_api_options)

    profiler.start("trtllm init")
    backend = params.get("backend", "pytorch")
    if backend == 'pytorch':
        llm = PyTorchLLM(**llm_args)
    elif backend == 'tensorrt':
        llm = LLM(**llm_args)
    else:
        raise click.BadParameter(
            f"{backend} is not a known backend, check help for available options.",
            param_hint="backend")
    profiler.stop("trtllm init")
    elapsed_time = profiler.elapsed_time_in_sec("trtllm init")
    logger.info(f"TRTLLM initialization time: {elapsed_time:.3f} seconds.")
    profiler.reset("trtllm init")

    # Pass llm to subcommands
    ctx.obj = llm


main.add_command(CnnDailymail.command)
main.add_command(MMLU.command)
main.add_command(GSM8K.command)
main.add_command(GPQADiamond.command)
main.add_command(GPQAMain.command)
main.add_command(GPQAExtended.command)
main.add_command(JsonModeEval.command)
main.add_command(MMMU.command)

if __name__ == "__main__":
    main()
