# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import argparse
import copy
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.machinery import SourceFileLoader
from multiprocessing import get_context
from typing import Optional, Union

import torch

from tensorrt_llm._utils import (local_mpi_rank, local_mpi_size, mpi_barrier,
                                 mpi_comm, mpi_rank, mpi_world_size)
from tensorrt_llm.bindings import KVCacheType
from tensorrt_llm.builder import BuildConfig, Engine, build
from tensorrt_llm.logger import logger, severity_map
from tensorrt_llm.lora_helper import LoraConfig
from tensorrt_llm.lora_manager import LoraManager
from tensorrt_llm.models import MODEL_MAP, PretrainedConfig
from tensorrt_llm.models.modeling_utils import SpeculativeDecodingMode
from tensorrt_llm.plugin import PluginConfig, add_plugin_argument
from tensorrt_llm.quantization.mode import QuantAlgo


def enum_type(enum_class):

    def parse_enum(value):
        if isinstance(value, enum_class):
            return value

        if isinstance(value, str):
            return enum_class.from_string(value)

        valid_values = [e.name for e in enum_class]
        raise argparse.ArgumentTypeError(
            f"Invalid value '{value}' of type {type(value).__name__}. Expected one of {valid_values}"
        )

    return parse_enum


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=None,
        help="The directory path that contains TensorRT LLM checkpoint.")
    parser.add_argument(
        '--model_config',
        type=str,
        default=None,
        help="The file path that saves TensorRT LLM checkpoint config.")
    parser.add_argument(
        '--build_config',
        type=str,
        default=None,
        help="The file path that saves TensorRT LLM build config.")
    parser.add_argument(
        '--model_cls_file',
        type=str,
        default=None,
        help="The file path that defines customized TensorRT LLM model.")
    parser.add_argument('--model_cls_name',
                        type=str,
                        default=None,
                        help="The customized TensorRT LLM model class name.")
    parser.add_argument(
        '--output_dir',
        type=str,
        default='engine_outputs',
        help=
        "The directory path to save the serialized engine files and engine config file."
    )

    parser.add_argument(
        '--max_batch_size',
        type=int,
        default=BuildConfig.max_batch_size,
        help="Maximum number of requests that the engine can schedule.")
    parser.add_argument('--max_input_len',
                        type=int,
                        default=BuildConfig.max_input_len,
                        help="Maximum input length of one request.")
    parser.add_argument(
        '--max_seq_len',
        '--max_decoder_seq_len',
        dest='max_seq_len',
        type=int,
        default=BuildConfig.max_seq_len,
        help="Maximum total length of one request, including prompt and outputs. "
        "If unspecified, the value is deduced from the model config.")
    parser.add_argument(
        '--max_beam_width',
        type=int,
        default=BuildConfig.max_beam_width,
        help="Maximum number of beams for beam search decoding.")
    parser.add_argument(
        '--max_num_tokens',
        type=int,
        default=BuildConfig.max_num_tokens,
        help=
        "Maximum number of batched input tokens after padding is removed in each batch. "
        "Currently, the input padding is removed by default; "
        "you may explicitly disable it by specifying ``--remove_input_padding disable``."
    )
    parser.add_argument(
        '--opt_num_tokens',
        type=int,
        default=BuildConfig.opt_num_tokens,
        help=
        "Optimal number of batched input tokens after padding is removed in each batch "
        "It equals to ``max_batch_size * max_beam_width`` by default, set this "
        "value as close as possible to the actual number of tokens on your workload. "
        "Note that this argument might be removed in the future.")
    parser.add_argument(
        '--max_encoder_input_len',
        type=int,
        default=BuildConfig.max_encoder_input_len,
        help="Maximum encoder input length for enc-dec models. "
        "Set ``max_input_len`` to 1 to start generation from decoder_start_token_id of length 1."
    )
    parser.add_argument(
        '--max_prompt_embedding_table_size',
        '--max_multimodal_len',
        type=int,
        default=BuildConfig.max_prompt_embedding_table_size,
        help=
        "Maximum prompt embedding table size for prompt tuning, or maximum multimodal input size for multimodal models. "
        "Setting a value > 0 enables prompt tuning or multimodal input.")
    parser.add_argument(
        '--kv_cache_type',
        default=argparse.SUPPRESS,
        type=enum_type(KVCacheType),
        help=
        "Set KV cache type (continuous, paged, or disabled). For disabled case, KV cache is disabled and only context phase is allowed."
    )
    parser.add_argument(
        '--paged_kv_cache',
        type=str,
        default=argparse.SUPPRESS,
        help=
        "Deprecated. Enabling this option is equvilient to ``--kv_cache_type paged`` for transformer based models."
    )

    parser.add_argument(
        '--input_timing_cache',
        type=str,
        default=BuildConfig.input_timing_cache,
        help=
        "The file path to read the timing cache. This option is ignored if the file does not exist."
    )
    parser.add_argument('--output_timing_cache',
                        type=str,
                        default=BuildConfig.output_timing_cache,
                        help="The file path to write the timing cache.")
    parser.add_argument(
        '--profiling_verbosity',
        type=str,
        default=BuildConfig.profiling_verbosity,
        choices=['layer_names_only', 'detailed', 'none'],
        help=
        "The profiling verbosity for the generated TensorRT engine. Setting to detailed allows inspecting tactic choices and kernel parameters."
    )
    parser.add_argument(
        '--strip_plan',
        default=BuildConfig.use_strip_plan,
        action='store_true',
        help=
        "Enable stripping weights from the final TensorRT engine under the assumption that the refit weights are identical to those provided at build time."
    )
    parser.add_argument('--weight_sparsity',
                        default=BuildConfig.weight_sparsity,
                        action='store_true',
                        help="Enable weight sparsity.")
    parser.add_argument(
        '--weight_streaming',
        default=BuildConfig.weight_streaming,
        action='store_true',
        help=
        "Enable offloading weights to CPU and streaming loading at runtime.",
    )
    parser.add_argument(
        '--fast_build',
        default=False,
        action='store_true',
        help=
        "Enable features for faster engine building. This may cause some performance degradation and is currently incompatible with int8/int4 quantization without plugin.",
    )

    parser.add_argument('--workers',
                        type=int,
                        default=1,
                        help="The number of workers for building in parallel.")
    parser.add_argument('--log_level',
                        type=str,
                        default='info',
                        choices=severity_map.keys(),
                        help="The logging level.")
    parser.add_argument('--enable_debug_output',
                        default=BuildConfig.enable_debug_output,
                        action='store_true',
                        help="Enable debug output.")
    parser.add_argument(
        '--visualize_network',
        type=str,
        default=None,
        help=
        "The directory path to export TensorRT Network as ONNX prior to Engine build for debugging."
    )
    parser.add_argument(
        '--dry_run',
        default=BuildConfig.dry_run,
        action='store_true',
        help=
        "Run through the build process except the actual Engine build for debugging."
    )
    parser.add_argument('--monitor_memory',
                        default=False,
                        action='store_true',
                        help="Enable memory monitor during Engine build.")

    logits_parser = parser.add_argument_group("Logits arguments")
    logits_parser.add_argument('--logits_dtype',
                               type=str,
                               default=None,
                               choices=['float16', 'float32'],
                               help="The data type of logits.")
    logits_parser.add_argument('--gather_context_logits',
                               action='store_true',
                               default=False,
                               help="Enable gathering context logits.")
    logits_parser.add_argument('--gather_generation_logits',
                               action='store_true',
                               default=False,
                               help="Enable gathering generation logits.")
    logits_parser.add_argument(
        '--gather_all_token_logits',
        action='store_true',
        default=False,
        help=
        "Enable both ``gather_context_logits`` and ``gather_generation_logits``."
    )

    lora_parser = parser.add_argument_group("LoRA arguments")
    lora_parser.add_argument(
        '--lora_dir',
        type=str,
        default=None,
        nargs="+",
        help="The directory of LoRA weights. "
        "If multiple directories are provided, the first one is used for configuration."
    )
    lora_parser.add_argument('--lora_ckpt_source',
                             type=str,
                             default="hf",
                             choices=["hf", "nemo"],
                             help="The source type of LoRA checkpoint.")
    lora_parser.add_argument(
        '--lora_target_modules',
        nargs='+',
        default=None,
        choices=LoraManager.LORA_MODULE_IDS.keys(),
        help=
        "The target module names that LoRA is applied. Only effective when ``lora_plugin`` is enabled."
    )
    lora_parser.add_argument(
        '--max_lora_rank',
        type=int,
        default=64,
        help="Maximum LoRA rank for different LoRA modules. "
        "It is used to compute the workspace size of LoRA plugin.")

    spec_parser = parser.add_argument_group("Speculative decoding arguments")
    spec_parser.add_argument('--speculative_decoding_mode',
                             default=None,
                             choices=[
                                 "draft_tokens_external", "lookahead_decoding",
                                 "medusa", "explicit_draft_tokens", "eagle"
                             ],
                             help="Mode of speculative decoding.")
    spec_parser.add_argument(
        '--max_draft_len',
        type=int,
        default=0,
        help=
        "Maximum lengths of draft tokens for speculative decoding target model."
    )

    plugin_config_parser = parser.add_argument_group("Plugin config arguments")
    add_plugin_argument(plugin_config_parser)
    return parser


def build_model(
    build_config: BuildConfig,
    rank: int = 0,
    ckpt_dir: str = None,
    model_config: Union[str, PretrainedConfig] = None,
    model_cls=None,
    dry_run:
    bool = False,  # return the modified BuildConfig without actually building the engine
    **kwargs
) -> Union[Engine, BuildConfig]:
    model_config = copy.deepcopy(model_config)

    logits_dtype = kwargs.get('logits_dtype')
    if logits_dtype is not None:
        model_config.logits_dtype = logits_dtype

    architecture = model_config.architecture
    assert not build_config.plugin_config.streamingllm, \
        "StreamingLLM is no longer supported because attention sink cannot work with the non-cyclic kv cache kernel & runtime changes."
    assert not build_config.plugin_config.pp_reduce_scatter or architecture == "MixtralForCausalLM", \
        "PP reduce scatter is only supported in the mixtral model."

    assert rank < model_config.mapping.world_size

    rank_config = copy.deepcopy(model_config)
    rank_config.set_rank(rank)

    if model_cls is None:
        assert architecture in MODEL_MAP, \
            f"Unsupported model architecture: {architecture}"
        model_cls = MODEL_MAP[architecture]
    if ckpt_dir is None:
        model = model_cls(rank_config)
    else:
        model = model_cls.from_checkpoint(ckpt_dir, config=rank_config)
    is_checkpoint_pruned = getattr(rank_config, 'is_pruned', False)

    if build_config.plugin_config.lora_plugin is not None:
        lora_config = LoraConfig(lora_dir=kwargs['lora_dir'] or [],
                                 lora_ckpt_source=kwargs['lora_ckpt_source'],
                                 max_lora_rank=kwargs['max_lora_rank'])
        if kwargs['lora_target_modules'] is not None:
            # command line options is preferred over the modules in the lora dir
            lora_config.lora_target_modules = kwargs['lora_target_modules']
        build_config.lora_config = lora_config

    if is_checkpoint_pruned or kwargs.pop('strip_plan', False):
        build_config.use_strip_plan = True
    build_config.use_refit = kwargs.get('refit', False)

    return build(model, build_config)


def build_and_save(rank, gpu_id, ckpt_dir, build_config, output_dir, log_level,
                   model_config, model_cls, **kwargs):
    torch.cuda.set_device(gpu_id)
    logger.set_level(log_level)
    engine = build_model(build_config,
                         rank,
                         ckpt_dir,
                         model_config,
                         model_cls=model_cls,
                         **kwargs)
    assert engine is not None
    engine.save(output_dir)
    return True


def parallel_build(model_config: PretrainedConfig,
                   ckpt_dir: Optional[str],
                   build_config: BuildConfig,
                   output_dir: str,
                   workers: int = 1,
                   log_level: str = 'info',
                   model_cls=None,
                   **kwargs):

    world_size = model_config.mapping.world_size
    use_mpi = mpi_world_size() > 1

    if not use_mpi and workers == 1:
        for rank in range(world_size):
            passed = build_and_save(rank, rank % workers, ckpt_dir,
                                    build_config, output_dir, log_level,
                                    model_config, model_cls, **kwargs)
            assert passed, "Engine building failed, please check error log."
    elif not use_mpi:
        with ProcessPoolExecutor(mp_context=get_context('spawn'),
                                 max_workers=workers) as p:
            futures = [
                p.submit(build_and_save, rank, rank % workers, ckpt_dir,
                         build_config, output_dir, log_level, model_config,
                         model_cls, **kwargs) for rank in range(world_size)
            ]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    exceptions.append(e)
            assert len(exceptions
                       ) == 0, "Engine building failed, please check error log."
    else:
        mpi_local_rank = local_mpi_rank()
        node_gpu_count = local_mpi_size()
        exceptions = []
        for engine_rank in range(world_size):
            if engine_rank % mpi_world_size() != mpi_rank():
                continue
            try:
                build_and_save(engine_rank, mpi_local_rank % node_gpu_count,
                               ckpt_dir, build_config, output_dir, log_level,
                               model_config, model_cls, **kwargs)
            except Exception as e:
                traceback.print_exc()
                exceptions.append(e)
        mpi_barrier()
        if len(exceptions) != 0:
            print("Engine building failed, please check error log.", flush=True)
            mpi_comm().Abort()


def main():
    parser = parse_arguments()
    args = parser.parse_args()

    if hasattr(args, 'gather_generation_logits'):
        logger.warning(
            'Option --gather_generation_logits is deprecated, a build flag is not required anymore. Use --output_generation_logits at runtime instead.'
        )

    if args.gather_all_token_logits:
        args.gather_context_logits = True
        args.gather_generation_logits = True
    if args.gather_context_logits and args.max_draft_len > 0:
        raise RuntimeError(
            "Gather context logits is not support with draft len > 0. "
            "If want to get the accepted tokens' logits from target model, please just enable gather_generation_logits"
        )

    logger.set_level(args.log_level)
    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    model_cls = None
    if args.model_cls_file is not None:
        assert args.model_cls_name is not None
        loader = SourceFileLoader('models', args.model_cls_file)
        mod = loader.load_module()
        model_cls = getattr(mod, args.model_cls_name)

    workers = min(torch.cuda.device_count(), args.workers)

    if hasattr(args, 'paged_kv_cache'):
        logger.warning(
            'Option --paged_kv_cache is deprecated, use --kv_cache_type=paged/disabled instead.'
        )

    plugin_config = PluginConfig.from_arguments(args)
    plugin_config.validate()
    if args.fast_build:
        plugin_config.manage_weights = True

    kwargs = {
        'logits_dtype': args.logits_dtype,
        'use_fused_mlp': args.use_fused_mlp,
        'lora_dir': args.lora_dir,
        'lora_ckpt_source': args.lora_ckpt_source,
        'max_lora_rank': args.max_lora_rank,
        'lora_target_modules': args.lora_target_modules,
        'strip_plan': args.strip_plan,
        'refit': False,
    }
    speculative_decoding_mode = SpeculativeDecodingMode.from_arguments(args)

    ckpt_dir_or_model_config = args.checkpoint_dir if args.checkpoint_dir is not None else args.model_config
    if ckpt_dir_or_model_config.lower().endswith('.json'):
        config_path = ckpt_dir_or_model_config
        ckpt_dir = None
    else:
        config_path = os.path.join(ckpt_dir_or_model_config, 'config.json')
        ckpt_dir = ckpt_dir_or_model_config
    model_config = PretrainedConfig.from_json_file(config_path)

    # avoid ValueError if not supported quantization is chosen with use_fused_mlp
    quant_algo = model_config.quantization.quant_algo
    if quant_algo and quant_algo not in (QuantAlgo.FP8,
                                         QuantAlgo.MIXED_PRECISION):
        kwargs['use_fused_mlp'] = False

    if args.build_config is None:
        if args.multiple_profiles == "enable" and args.opt_num_tokens is not None:
            raise RuntimeError(
                "multiple_profiles is enabled, while opt_num_tokens is set. "
                "They are not supposed to be working in the same time for now.")

        # This should only be used for debugging.
        # The env var BUILDER_FORCE_NUM_PROFILES should override the number of
        # optimization profiles during TRT build.
        # BUILDER_FORCE_NUM_PROFILES must be less than or equal to the number of
        # optimization profiles set by model's prepare_inputs().
        force_num_profiles_from_env = os.environ.get(
            "BUILDER_FORCE_NUM_PROFILES", None)
        if force_num_profiles_from_env is not None:
            logger.warning(
                f"Overriding # of builder profiles <= {force_num_profiles_from_env}."
            )

        build_config = BuildConfig.from_dict(
            {
                'max_input_len':
                args.max_input_len,
                'max_seq_len':
                args.max_seq_len,
                'max_batch_size':
                args.max_batch_size,
                'max_beam_width':
                args.max_beam_width,
                'max_num_tokens':
                args.max_num_tokens,
                'opt_num_tokens':
                args.opt_num_tokens,
                'max_prompt_embedding_table_size':
                args.max_prompt_embedding_table_size,
                'gather_context_logits':
                args.gather_context_logits,
                'gather_generation_logits':
                args.gather_generation_logits,
                'strongly_typed':
                True,
                'force_num_profiles':
                force_num_profiles_from_env,
                'weight_sparsity':
                args.weight_sparsity,
                'profiling_verbosity':
                args.profiling_verbosity,
                'enable_debug_output':
                args.enable_debug_output,
                'max_draft_len':
                args.max_draft_len,
                'speculative_decoding_mode':
                speculative_decoding_mode,
                'input_timing_cache':
                args.input_timing_cache,
                'output_timing_cache':
                args.output_timing_cache,
                'dry_run':
                args.dry_run,
                'visualize_network':
                args.visualize_network,
                'max_encoder_input_len':
                args.max_encoder_input_len,
                'weight_streaming':
                args.weight_streaming,
                'monitor_memory':
                args.monitor_memory,
                'use_mrope':
                (True if model_config.qwen_type == "qwen2_vl" else False)
                if hasattr(model_config, "qwen_type") else False
            },
            plugin_config=plugin_config)

        if hasattr(args, 'kv_cache_type'):
            build_config.update_from_dict({'kv_cache_type': args.kv_cache_type})
    else:
        build_config = BuildConfig.from_json_file(args.build_config,
                                                  plugin_config=plugin_config)

    parallel_build(model_config, ckpt_dir, build_config, args.output_dir,
                   workers, args.log_level, model_cls, **kwargs)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Total time of building all engines: {t}')


if __name__ == '__main__':
    main()
