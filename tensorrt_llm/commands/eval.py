# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import yaml

import tensorrt_llm.profiler as profiler

from .. import LLM as PyTorchLLM
from ..evaluate import (AALCR, AIME2025, AIME2026, GSM8K, HLE, MMLU, MMMU,
                        ArenaHard, CnnDailymail, CoVoST2, GPQADiamond,
                        GPQAExtended, GPQAMain, GPQANemoSkills, IFBench,
                        JsonModeEval, LongBenchV1, LongBenchV2, MMMUPro,
                        SciCode)
from ..llmapi import KvCacheConfig
from ..llmapi.llm_args import TorchLlmArgs
from ..llmapi.llm_utils import update_llm_args_with_extra_dict
from ..logger import logger, severity_map
from ..usage import apply_usage_session_config
from ..usage import config as _telemetry_config
from ._telemetry import TelemetryGroup, apply_raw_config_telemetry_opt_out
from .utils import collect_explicit_cli_keys

# CLI defaults are sourced from the TorchLlmArgs field defaults so they stay in
# lock-step with the args class and can't drift.
_LLM_ARGS_FIELDS = TorchLlmArgs.model_fields

# Map Click parameter names to the LlmArgs field name (or merge-function CLI
# scalar name) used by `update_llm_args_with_extra_options`.
_CLICK_TO_LLM_ARG = {
    "tp_size": "tensor_parallel_size",
    "pp_size": "pipeline_parallel_size",
    "ep_size": "moe_expert_parallel_size",
    "kv_cache_free_gpu_memory_fraction": "free_gpu_memory_fraction",
    "disable_kv_cache_reuse": "enable_block_reuse",
}


@click.group(
    cls=TelemetryGroup,
    telemetry_usage_context=_telemetry_config.UsageContext.CLI_EVAL,
    telemetry_component="llm",
)
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
@click.option(
    "--custom_tokenizer",
    type=str,
    default=None,
    help=
    "Custom tokenizer type: alias (e.g., 'deepseek_v32') or Python import path "
    "(e.g., 'tensorrt_llm.tokenizer.deepseek_v32.DeepseekV32Tokenizer'). [Experimental]"
)
@click.option(
    "--backend",
    type=click.Choice(["pytorch"]),
    default="pytorch",
    help="The backend to use for evaluation. Default is pytorch backend.")
@click.option('--log_level',
              type=click.Choice(severity_map.keys()),
              default='info',
              help="The logging level.")
@click.option("--max_beam_width",
              type=int,
              default=_LLM_ARGS_FIELDS["max_beam_width"].default,
              help="Maximum number of beams for beam search decoding.")
@click.option("--max_batch_size",
              type=int,
              default=_LLM_ARGS_FIELDS["max_batch_size"].default,
              help="Maximum number of requests that the engine can schedule.")
@click.option(
    "--max_num_tokens",
    type=int,
    default=_LLM_ARGS_FIELDS["max_num_tokens"].default,
    help=
    "Maximum number of batched input tokens after padding is removed in each batch."
)
@click.option(
    "--max_seq_len",
    type=int,
    default=None,
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
@click.option("--revision",
              type=str,
              default=None,
              help="The revision to use for the HuggingFace model "
              "(branch name, tag name, or commit id).")
@click.option("--config",
              "--extra_llm_api_options",
              "extra_llm_api_options",
              type=str,
              default=None,
              help="Path to a YAML configuration file. Explicit CLI flags "
              "take precedence over values in this file. Can be specified "
              "as either --config or --extra_llm_api_options.")
@click.option("--disable_kv_cache_reuse",
              is_flag=True,
              default=False,
              help="Flag for disabling KV cache reuse.")
@click.option("--telemetry/--no-telemetry",
              default=True,
              help="Enable or disable anonymous usage telemetry collection.")
@click.pass_context
def main(ctx, model: str, tokenizer: Optional[str],
         custom_tokenizer: Optional[str], log_level: str, backend: str,
         max_beam_width: int, max_batch_size: int, max_num_tokens: int,
         max_seq_len: int, tp_size: int, pp_size: int, ep_size: Optional[int],
         gpus_per_node: Optional[int], kv_cache_free_gpu_memory_fraction: float,
         trust_remote_code: bool, revision: Optional[str],
         extra_llm_api_options: Optional[str], disable_kv_cache_reuse: bool,
         telemetry: bool):
    logger.set_level(log_level)

    explicit_cli_keys = collect_explicit_cli_keys(
        exclude=("extra_llm_api_options", "config"),
        translate=_CLICK_TO_LLM_ARG)

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
        enable_block_reuse=not disable_kv_cache_reuse)

    llm_args = {
        "model":
        model,
        "tokenizer":
        tokenizer,
        "custom_tokenizer":
        custom_tokenizer,
        "tensor_parallel_size":
        tp_size,
        "pipeline_parallel_size":
        pp_size,
        "moe_expert_parallel_size":
        ep_size,
        "gpus_per_node":
        gpus_per_node,
        "trust_remote_code":
        trust_remote_code,
        "revision":
        revision,
        "kv_cache_config":
        kv_cache_config,
        "telemetry_config":
        _telemetry_config.TelemetryConfig(
            disabled=not telemetry,
            usage_context=_telemetry_config.UsageContext.CLI_EVAL),
    }

    if backend == 'pytorch':
        llm_cls = PyTorchLLM
        llm_args.update(max_batch_size=max_batch_size,
                        max_num_tokens=max_num_tokens,
                        max_beam_width=max_beam_width,
                        max_seq_len=max_seq_len)
    else:
        raise click.BadParameter(
            f"{backend} is not a known backend, check help for available options.",
            param_hint="backend")

    # Pre-check YAML telemetry settings before full config validation.
    if extra_llm_api_options is not None:
        with open(extra_llm_api_options, 'r') as f:
            llm_args_extra_dict = yaml.safe_load(f)
        if llm_args_extra_dict is None:
            llm_args_extra_dict = {}
        elif not isinstance(llm_args_extra_dict, dict):
            raise ValueError("Configuration file root must be a mapping.")
        apply_raw_config_telemetry_opt_out(
            llm_args_extra_dict,
            usage_context=_telemetry_config.UsageContext.CLI_EVAL,
            component="llm",
            explicit_cli_telemetry="telemetry" in explicit_cli_keys,
        )
        llm_args = update_llm_args_with_extra_dict(
            llm_args,
            llm_args_extra_dict,
            extra_llm_api_options,
            explicit_cli_keys=explicit_cli_keys)

    # Update telemetry state or disable the early session.
    apply_usage_session_config(
        llm_args.get("telemetry_config"),
        default_usage_context=_telemetry_config.UsageContext.CLI_EVAL.value,
        component="llm",
        lifecycle_phase="config_validation",
    )

    profiler.start("trtllm init")
    llm = llm_cls(**llm_args)
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
main.add_command(MMMUPro.command)
main.add_command(CoVoST2.command)
main.add_command(LongBenchV1.command)
main.add_command(LongBenchV2.command)
main.add_command(AIME2025.command)
main.add_command(AIME2026.command)
main.add_command(GPQANemoSkills.command)
main.add_command(IFBench.command)
main.add_command(SciCode.command)
main.add_command(HLE.command)
main.add_command(AALCR.command)
main.add_command(ArenaHard.command)

if __name__ == "__main__":
    main()
