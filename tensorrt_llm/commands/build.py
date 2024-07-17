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

from ..auto_parallel import infer_cluster_config
from ..auto_parallel.cluster_info import cluster_infos
from ..builder import BuildConfig, Engine, build
from ..functional import PositionEmbeddingType
from ..logger import logger
from ..lora_manager import LoraConfig, LoraManager
from ..models import MODEL_MAP, PretrainedConfig
from ..models.modeling_utils import SpeculativeDecodingMode
from ..plugin import PluginConfig, add_plugin_argument


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--model_config', type=str, default=None)
    parser.add_argument('--build_config', type=str, default=None)
    parser.add_argument('--model_cls_file', type=str, default=None)
    parser.add_argument('--model_cls_name', type=str, default=None)
    parser.add_argument(
        '--input_timing_cache',
        type=str,
        default=None,
        help=
        'The path to read timing cache, will be ignored if the file does not exist'
    )
    parser.add_argument('--output_timing_cache',
                        type=str,
                        default='model.cache',
                        help='The path to write timing cache')
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument(
        '--profiling_verbosity',
        type=str,
        default='layer_names_only',
        choices=['layer_names_only', 'detailed', 'none'],
        help=
        'The profiling verbosity for the generated TRT engine. Set to detailed can inspect tactic choices and kernel parameters.'
    )
    parser.add_argument('--enable_debug_output',
                        default=False,
                        action='store_true')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='engine_outputs',
        help='The path to save the serialized engine files and model configs')
    parser.add_argument('--workers',
                        type=int,
                        default='1',
                        help='The number of workers for building in parallel')
    parser.add_argument(
        '--max_batch_size',
        type=int,
        default=256,
        help="Max number of requests that the engine can handle.")
    parser.add_argument('--max_input_len',
                        type=int,
                        default=1024,
                        help="Max input length of one request.")
    parser.add_argument('--max_output_len',
                        type=int,
                        default=None,
                        help="Deprecated. Please use `--max_seq_len` instead.")
    parser.add_argument(
        '--max_seq_len',
        '--max_decoder_seq_len',
        dest='max_seq_len',
        type=int,
        default=None,
        help="Max total length of one request, including prompt and outputs. "
        "If unspecified, will try to deduce from the model config.")
    parser.add_argument('--max_beam_width', type=int, default=1)
    parser.add_argument(
        '--max_num_tokens',
        type=int,
        default=8192,
        help="Max number of batched input tokens after padding is removed "
        "(triggered by `--remove_input_padding`) in each batch.")
    parser.add_argument(
        '--opt_num_tokens',
        type=int,
        default=None,
        help='It equals to max_batch_size*max_beam_width by default, set this '
        'value as close as possible to the actual number of tokens on your workload. '
        'Note that this argument might be removed in the future.')
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--pp_size', type=int, default=1)
    parser.add_argument(
        '--max_prompt_embedding_table_size',
        '--max_multimodal_len',
        type=int,
        default=0,
        help=
        'Setting to a value > 0 enables support for prompt tuning or multimodal input.'
    )
    parser.add_argument(
        '--use_fused_mlp',
        default=False,
        action='store_true',
        help=
        'Enable horizontal fusion in GatedMLP, reduces layer input traffic and potentially improves performance. '
        'For FP8 PTQ, the downside is slight reduction of accuracy because one of the quantization scaling factors is discarded. '
        '(An example for reference only: 0.45734 vs 0.45755 for LLaMA-v2 7B using `modelopt/examples/hf/instruct_eval/mmlu.py`).'
    )
    parser.add_argument(
        '--gather_all_token_logits',
        action='store_true',
        default=False,
        help='Enable both gather_context_logits and gather_generation_logits')
    parser.add_argument('--gather_context_logits',
                        action='store_true',
                        default=False,
                        help='Gather context logits')
    parser.add_argument('--gather_generation_logits',
                        action='store_true',
                        default=False,
                        help='Gather generation logits')

    parser.add_argument('--builder_opt', type=int, default=None)
    parser.add_argument('--logits_dtype',
                        type=str,
                        default=None,
                        choices=['float16', 'float32'])
    parser.add_argument('--weight_sparsity', default=False, action='store_true')
    parser.add_argument(
        '--max_draft_len',
        type=int,
        default=0,
        help=
        'Maximum lengths of draft tokens for speculative decoding target model.'
    )
    parser.add_argument(
        '--lora_dir',
        type=str,
        default=None,
        nargs="+",
        help="The directory of LoRA weights. "
        "Use config from the first directory if multiple directories are provided."
    )
    parser.add_argument('--lora_ckpt_source',
                        type=str,
                        default="hf",
                        choices=["hf", "nemo"],
                        help="The source of lora checkpoint.")
    parser.add_argument(
        '--lora_target_modules',
        nargs='+',
        default=None,
        choices=LoraManager.LORA_MODULE_IDS.keys(),
        help=
        "Add lora in which modules. Only be activated when use_lora_plugin is enabled."
    )
    parser.add_argument(
        '--max_lora_rank',
        type=int,
        default=64,
        help='maximum lora rank for different lora modules. '
        'It is used to compute the workspace size of lora plugin.')
    parser.add_argument('--auto_parallel',
                        type=int,
                        default=1,
                        help='MPI world size for auto parallel.')
    parser.add_argument(
        '--gpus_per_node',
        type=int,
        default=8,
        help=
        'Number of GPUs each node has in a multi-node setup. This is a cluster spec and can be greater/smaller than world size'
    )
    parser.add_argument(
        '--cluster_key',
        type=str,
        default=None,
        choices=cluster_infos.keys(),
        help=
        'Unique name for target GPU type. Inferred from current GPU type if not specified.'
    )
    parser.add_argument(
        '--strip_plan',
        default=False,
        action='store_true',
        help=
        'Whether to strip weights from the final TRT engine under the assumption that the refit weights will be identical to those provided at build time.'
    )
    parser.add_argument(
        '--max_encoder_input_len',
        type=int,
        default=1024,
        help=
        'Specify max encoder input length when using enc-dec models. Set max_input_len to 1 to start generation from decoder_start_token_id of length 1.'
    )
    parser.add_argument(
        '--visualize_network',
        default=False,
        action='store_true',
        help=
        'TRT Networks will be exported to ONNX prior to Engine build for debugging. '
    )
    parser.add_argument(
        '--dry_run',
        default=False,
        action='store_true',
        help=
        'Run through the build process except the actual Engine build for debugging. '
    )
    parser.add_argument('--speculative_decoding_mode',
                        default=None,
                        choices=[
                            "draft_tokens_external",
                            "lookahead_decoding",
                            "medusa",
                        ],
                        help='Mode of speculative decoding.')
    parser.add_argument(
        '--weight_streaming',
        default=False,
        action='store_true',
        help=
        'Specify whether offloading weights to CPU and streaming loading at runtime.',
    )

    plugin_config_parser = parser.add_argument_group("plugin_config")
    add_plugin_argument(plugin_config_parser)

    args = parser.parse_args()
    if args.gather_all_token_logits:
        args.gather_context_logits = True
        args.gather_generation_logits = True

    if args.gather_context_logits and args.max_draft_len > 0:
        raise RuntimeError(
            "Gather context logits is not support with draft len > 0. "
            "If want to get the accepted tokens' logits from target model, please just enable gather_generation_logits"
        )
    return args


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
    assert not build_config.plugin_config.streamingllm or architecture == "LlamaForCausalLM", \
        "StreamingLLM is only supported in the llama model."
    real_rank = rank

    if build_config.plugin_config.reduce_fusion and model_config.mapping.tp_size == 1:
        build_config.plugin_config.reduce_fusion = False

    model_config.mapping.gpus_per_node = build_config.auto_parallel_config.gpus_per_node
    if build_config.auto_parallel_config.enabled:
        assert rank < build_config.auto_parallel_config.world_size
        assert model_config.mapping.pp_size == 1 and model_config.mapping.tp_size == 1, \
            "You must convert to full model with TP=1&&PP=1 to use auto parallel planner"
        #TODO: TRTLLM-193 remove this after the new build API for autopp is done
        rank = 0  # This is a WAR to construct a whole model and load all the weights before auto parallel
    else:
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

    build_config.use_fused_mlp = kwargs.get('use_fused_mlp', False)
    # tells the low level build api to only build rank-th shard of the model
    if build_config.auto_parallel_config.enabled:
        model.config.mapping.rank = real_rank

    if is_checkpoint_pruned or kwargs.pop('strip_plan', False):
        build_config.use_strip_plan = True
    build_config.use_refit = kwargs.get('refit', False)

    if dry_run:
        return build_config

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

    if build_config.auto_parallel_config.enabled:
        if model_config.mapping.world_size > 1:
            raise RuntimeError(
                "manually TP and PP are not supported in auto parallel mode.")
        if build_config.auto_parallel_config.debug_mode:
            world_size = 1
        else:
            world_size = build_config.auto_parallel_config.world_size
    else:
        world_size = model_config.mapping.world_size

    if workers == 1:
        for rank in range(world_size):
            passed = build_and_save(rank, rank % workers, ckpt_dir,
                                    build_config, output_dir, log_level,
                                    model_config, model_cls, **kwargs)
            assert passed, "Engine building failed, please check error log."
    else:
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


def main():
    args = parse_arguments()
    logger.set_level(args.log_level)
    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model_cls = None
    if args.model_cls_file is not None:
        assert args.model_cls_name is not None
        loader = SourceFileLoader('models', args.model_cls_file)
        mod = loader.load_module()
        model_cls = getattr(mod, args.model_cls_name)

    workers = min(torch.cuda.device_count(), args.workers)

    plugin_config = PluginConfig.from_arguments(args)

    kwargs = {
        'logits_dtype': args.logits_dtype,
        'use_fused_mlp': args.use_fused_mlp,
        'tp_size': args.tp_size,
        'pp_size': args.pp_size,
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

    if args.build_config is None:
        if args.multiple_profiles == "enable" and args.opt_num_tokens is not None:
            raise RuntimeError(
                "multiple_profiles is enabled, while opt_num_tokens is set. "
                "They are not supposed to be working in the same time for now.")
        if args.cluster_key is not None:
            cluster_config = dict(cluster_key=args.cluster_key)
        else:
            cluster_config = infer_cluster_config()

        if args.max_output_len:
            logger.warning(
                '--max_output_len has been deprecated in favor of --max_seq_len'
            )
            if args.max_input_len:
                if args.max_seq_len:
                    logger.warning(
                        '--max_seq_len has been overwritten due to --max_output_len being specified'
                    )
                args.max_seq_len = args.max_input_len + args.max_output_len
            else:
                raise Exception(
                    f"--max_output_len is specified but not --max_input_len")

            del args.max_output_len

        # Extract rotary scaling which will be used for checks and default value of max_seq_len
        rotary_scaling = getattr(model_config, "rotary_scaling", None)
        if rotary_scaling is not None:
            rotary_type = rotary_scaling['type']
            rotary_factor = rotary_scaling[
                'factor'] if rotary_type != 'su' else 1
        else:
            rotary_factor = 1

        if args.max_seq_len is None:
            # Step 1: Find the upper bound of max_seq_len
            deduced_max_seq_len = 2048
            if model_config.max_position_embeddings is not None:
                deduced_max_seq_len = model_config.max_position_embeddings

            # Step 2: Scale max_seq_len with rotary scaling
            if rotary_factor != 1:
                deduced_max_seq_len *= rotary_factor
                logger.warning(
                    f'max_seq_len is scaled to {deduced_max_seq_len} by rotary scaling {rotary_factor}'
                )

            # Step 3: Assign the new max_seq_len
            args.max_seq_len = deduced_max_seq_len
            logger.info(
                f'max_seq_len is not specified, using value {deduced_max_seq_len}'
            )
        else:
            if not plugin_config.streamingllm and model_config.max_position_embeddings is not None \
                and model_config.position_embedding_type != PositionEmbeddingType.relative:
                if args.max_seq_len > model_config.max_position_embeddings * rotary_factor:
                    logger.warning(
                        f'max_seq_len {args.max_seq_len} is larger than max_position_embeddings {model_config.max_position_embeddings} * rotary scaling {rotary_factor}, '
                        'the model accuracy might be affected')

        if args.max_input_len > args.max_seq_len:
            logger.warning(
                f'max_input_len is {args.max_input_len} is larger than max_seq_len {args.max_seq_len}, clipping it to max_seq_len'
            )
            args.max_input_len = args.max_seq_len

        build_config = BuildConfig.from_dict(
            {
                'max_input_len': args.max_input_len,
                'max_seq_len': args.max_seq_len,
                'max_batch_size': args.max_batch_size,
                'max_beam_width': args.max_beam_width,
                'max_num_tokens': args.max_num_tokens,
                'opt_num_tokens': args.opt_num_tokens,
                'max_prompt_embedding_table_size':
                args.max_prompt_embedding_table_size,
                'gather_context_logits': args.gather_context_logits,
                'gather_generation_logits': args.gather_generation_logits,
                'strongly_typed': True,
                'builder_opt': args.builder_opt,
                'weight_sparsity': args.weight_sparsity,
                'profiling_verbosity': args.profiling_verbosity,
                'enable_debug_output': args.enable_debug_output,
                'max_draft_len': args.max_draft_len,
                'speculative_decoding_mode': speculative_decoding_mode,
                'input_timing_cache': args.input_timing_cache,
                'output_timing_cache': args.output_timing_cache,
                'auto_parallel_config': {
                    'world_size':
                    args.auto_parallel,
                    'gpus_per_node':
                    args.gpus_per_node,
                    'sharded_io_allowlist': [
                        'past_key_value_\\d+',
                        'present_key_value_\\d*',
                    ],
                    'same_buffer_io': {
                        'past_key_value_(\\d+)': 'present_key_value_\\1',
                    },
                    **cluster_config,
                },
                'dry_run': args.dry_run,
                'visualize_network': args.visualize_network,
                'max_encoder_input_len': args.max_encoder_input_len,
                'weight_streaming': args.weight_streaming,
            },
            plugin_config=plugin_config)
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
