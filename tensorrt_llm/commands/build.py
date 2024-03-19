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
from typing import Union

import safetensors
import torch

from .._common import check_max_num_tokens
from ..auto_parallel.config import _cluster_infos, infer_cluster_key
from ..builder import BuildConfig, Engine, build
from ..logger import logger
from ..lora_manager import LoraBuildConfig
from ..models import MODEL_MAP, PretrainedConfig
from ..models.modeling_utils import (WEIGHT_LOADER_MODELS, optimize_model,
                                     preprocess_weights)
from ..plugin import PluginConfig, add_plugin_argument
from ..quantization import QuantMode
from ..quantization.mode import FP8, W4A16, W8A16


def parse_arguments():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--max_batch_size', type=int, default=1)
    parser.add_argument('--max_input_len', type=int, default=1024)
    parser.add_argument('--max_output_len', type=int, default=1024)
    parser.add_argument('--max_beam_width', type=int, default=1)
    parser.add_argument('--max_num_tokens', type=int, default=None)
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
        'For FP8 PTQ, the downside is slight reduction of accuracy because one of the quantization scaling factors are discarded. '
        '(An example for reference only: 0.45734 vs 0.45755 for LLaMA-v2 7B using `ammo/examples/hf/instruct_eval/mmlu.py`).'
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
    parser.add_argument(
        '--strongly_typed',
        action='store_true',
        default=False,
        help=
        'This option is introduced with TensorRT 9.1.0.1+ and will reduce the engine building time. '
        'It\'s not expected to see performance or accuracy regression after enable this flag. '
        'Note that, we may remove this flag in the future, and enable the feature by default.'
    )
    parser.add_argument('--builder_opt', type=int, default=None)
    parser.add_argument('--logits_dtype',
                        type=str,
                        default=None,
                        choices=['float16', 'float32'])
    parser.add_argument('--weight_only_precision',
                        type=str,
                        default=None,
                        choices=['int8', 'int4'])
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
        choices=[
            "attn_qkv",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_dense",
            "mlp_h_to_4h",
            "mlp_gate",
            "mlp_4h_to_h",
        ],
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
        choices=_cluster_infos.keys(),
        help=
        'Unique name for target GPU type. Inferred from current GPU type if not specified.'
    )

    plugin_config_parser = parser.add_argument_group("plugin_config")
    add_plugin_argument(plugin_config_parser)

    args = parser.parse_args()
    if args.gather_all_token_logits:
        args.gather_context_logits = True
        args.gather_generation_logits = True

    return args


def preprocess_model_config(model_config, **kwargs):
    if model_config.architecture in WEIGHT_LOADER_MODELS:
        model_config.mapping.tp_size = kwargs['tp_size']
        model_config.mapping.pp_size = kwargs['pp_size']
        model_config.mapping.world_size = kwargs['tp_size'] * kwargs['pp_size']


def build_model(build_config: BuildConfig,
                rank: int = 0,
                ckpt_dir: str = None,
                model_config: Union[str, PretrainedConfig] = None,
                weights=None,
                model_cls=None,
                **kwargs) -> Engine:
    if ckpt_dir is not None:
        model_config = PretrainedConfig.from_json_file(
            os.path.join(ckpt_dir, 'config.json'))
    else:
        assert model_config is not None
        if isinstance(model_config, PretrainedConfig):
            model_config = model_config
        else:
            model_config = PretrainedConfig.from_json_file(model_config)

    preprocess_model_config(model_config, **kwargs)

    logits_dtype = kwargs.get('logits_dtype')
    if logits_dtype is not None:
        model_config.logits_dtype = logits_dtype

    model_config.use_prompt_tuning = build_config.max_prompt_embedding_table_size > 0
    weight_only_precision = kwargs.get('weight_only_precision', None)
    if model_config.quant_mode == QuantMode(
            0) and weight_only_precision is not None:
        if weight_only_precision == 'int4':
            model_config.quant_mode = QuantMode.use_weight_only(
                use_int4_weights=True)
            model_config.quantization.quant_algo = W4A16
        else:
            model_config.quant_mode = QuantMode.use_weight_only(
                use_int4_weights=False)
            model_config.quantization.quant_algo = W8A16

    architecture = model_config.architecture

    if model_cls is None:
        if architecture not in MODEL_MAP:
            raise RuntimeError(
                f'Unsupported model architecture: {architecture}')
        model_cls = MODEL_MAP[architecture]

    real_rank = rank
    if build_config.auto_parallel_config.enabled:
        assert rank < build_config.auto_parallel_config.world_size
        rank = 0
    else:
        assert rank < model_config.mapping.world_size

    rank_config = copy.deepcopy(model_config)
    rank_config.set_rank(rank)

    model = model_cls.from_config(rank_config)
    if ckpt_dir is not None:
        if model_config.architecture in WEIGHT_LOADER_MODELS:
            model_path = os.path.join(ckpt_dir, 'rank0.safetensors')
        else:
            model_path = os.path.join(ckpt_dir, f'rank{rank}.safetensors')

        if os.path.isfile(model_path):
            weights = {}
            with safetensors.safe_open(model_path, framework='pt',
                                       device='cpu') as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
        else:
            logger.warning(
                f"Cannot find {model_path}. Use dummy model weights.")

    if weights is not None:
        preprocess_weights(weights, rank_config)
        model.load(weights)

    if model.config.quantization.quant_algo == FP8 or model.config.quantization.kv_cache_quant_algo == FP8:
        build_config.strongly_typed = True

    if hasattr(model.config, 'max_medusa_token_len'):
        build_config.max_draft_len = model.config.max_medusa_token_len

    if build_config.plugin_config.lora_plugin is not None:
        lora_config = LoraBuildConfig(
            lora_dir=kwargs['lora_dir'] or [],
            lora_ckpt_source=kwargs['lora_ckpt_source'],
            max_lora_rank=kwargs['max_lora_rank'])
        if kwargs['lora_target_modules'] is not None:
            # command line options is preferred over the modules in the lora dir
            lora_config.lora_target_modules = kwargs['lora_target_modules']
        # TODO(yuxianq): remove this check after TopModelMixin merged into PretrainedModel
        assert hasattr(model, 'use_lora'), "This model does not support LoRA"
        model.use_lora(lora_config)

    use_fused_mlp = kwargs.get('use_fused_mlp', False)
    use_auto_parallel = build_config.auto_parallel_config.enabled
    model = optimize_model(
        model,
        use_fused_mlp=(use_fused_mlp and not use_auto_parallel),
        use_prompt_tuning=(build_config.max_prompt_embedding_table_size > 0))

    if use_auto_parallel:
        model.config.mapping.rank = real_rank

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


def parallel_build(ckpt_dir_or_model_config: str,
                   build_config: BuildConfig,
                   output_dir: str,
                   workers: int = 1,
                   log_level: str = 'info',
                   model_cls=None,
                   **kwargs):
    ckpt_dir = ckpt_dir_or_model_config
    if ckpt_dir_or_model_config.lower().endswith('.json'):
        model_config = PretrainedConfig.from_json_file(ckpt_dir_or_model_config)
        ckpt_dir = None
    else:
        model_config = PretrainedConfig.from_json_file(
            os.path.join(ckpt_dir_or_model_config, 'config.json'))

    preprocess_model_config(model_config, **kwargs)

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
        'weight_only_precision': args.weight_only_precision,
        'tp_size': args.tp_size,
        'pp_size': args.pp_size,
        'lora_dir': args.lora_dir,
        'lora_ckpt_source': args.lora_ckpt_source,
        'max_lora_rank': args.max_lora_rank,
        'lora_target_modules': args.lora_target_modules,
    }
    if args.build_config is None:
        args.max_num_tokens = check_max_num_tokens(
            max_num_tokens=args.max_num_tokens,
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            remove_input_padding=(args.remove_input_padding == "enable"),
            enable_context_fmha=(args.context_fmha == "enable"),
            tokens_per_block=args.tokens_per_block)
        build_config = BuildConfig.from_dict(
            {
                'max_input_len': args.max_input_len,
                'max_output_len': args.max_output_len,
                'max_batch_size': args.max_batch_size,
                'max_beam_width': args.max_beam_width,
                'max_num_tokens': args.max_num_tokens,
                'max_prompt_embedding_table_size':
                args.max_prompt_embedding_table_size,
                'gather_context_logits': args.gather_context_logits,
                'gather_generation_logits': args.gather_generation_logits,
                'strongly_typed': args.strongly_typed,
                'builder_opt': args.builder_opt,
                'profiling_verbosity': args.profiling_verbosity,
                'enable_debug_output': args.enable_debug_output,
                'max_draft_len': args.max_draft_len,
                'input_timing_cache': args.input_timing_cache,
                'output_timing_cache': args.output_timing_cache,
                'auto_parallel_config': {
                    'world_size':
                    args.auto_parallel,
                    'gpus_per_node':
                    args.gpus_per_node,
                    'cluster_key':
                    args.cluster_key or infer_cluster_key(),
                    'sharded_io_allowlist': [
                        'past_key_value_\\d+',
                        'present_key_value_\\d*',
                    ],
                    'same_buffer_io': {
                        'past_key_value_(\\d+)': 'present_key_value_\\1',
                    },
                },
            },
            plugin_config=plugin_config)
    else:
        build_config = BuildConfig.from_json_file(args.build_config,
                                                  plugin_config=plugin_config)

    source = args.checkpoint_dir if args.checkpoint_dir is not None else args.model_config
    parallel_build(source, build_config, args.output_dir, workers,
                   args.log_level, model_cls, **kwargs)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Total time of building all engines: {t}')


if __name__ == '__main__':
    main()
