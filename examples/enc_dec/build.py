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
import configparser
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
from run import get_engine_name

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.network import net_guard

from t5.weight import parse_t5_config, load_from_hf_t5, load_from_binary_t5  # isort:skip
from bart.weight import parse_bart_config, load_from_binary_bart  # isort:skip
from nmt.weight import parse_nmt_config, load_from_binary_nmt  # isort:skip


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        f.write(engine)
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')


def parse_config(ini_file, component, args):
    config = configparser.ConfigParser()
    assert ini_file.exists(), f"Missing config file {ini_file}"
    config.read(ini_file)
    model_type = config.get('structure', 'model_type')
    args.model_type = model_type
    args = globals()[f'parse_{model_type}_config'](config, component, args)
    return args


def parse_arguments(component):
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='MPI world size (must equal TP * PP)')
    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')
    parser.add_argument('--pp_size',
                        type=int,
                        default=1,
                        help='N-way pipeline parallelism size')
    parser.add_argument(
        '--gpus_per_node',
        type=int,
        default=8,
        help=
        'Number of GPUs each node has in a multi-node setup. This is a cluster spec and can be greater/smaller than world size'
    )
    parser.add_argument('--parallel_build', default=False, action='store_true')
    parser.add_argument('--weight_dir',
                        '-i',
                        type=str,
                        default=None,
                        help='Path to the converted weight file')
    parser.add_argument(
        '--output_dir',
        '-o',
        type=Path,
        default='trt_engines',
        help=
        'The path to save the serialized engine files, timing cache file and model configs'
    )
    parser.add_argument(
        '--weight_from_pytorch_ckpt',
        default=False,
        action='store_true',
        help=
        'Load weight from PyTorch checkpoint. model_dir must point to ckpt directory'
    )
    parser.add_argument('--engine_name',
                        '-n',
                        type=str,
                        default='enc_dec',
                        help='TensorRT engine name prefix')
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument(
        '--timing_cache',
        type=str,
        default='model.cache',
        help=
        'The path of to read timing cache from, will be ignored if the file does not exist'
    )
    parser.add_argument(
        '--profiling_verbosity',
        type=str,
        default='layer_names_only',
        choices=['layer_names_only', 'detailed', 'none'],
        help=
        'The profiling verbosity for the generated TRT engine. Set to detailed can inspect tactic choices and kernel parameters.'
    )
    parser.add_argument('--model_type',
                        type=str,
                        choices=['t5', 'bart', 'nmt'],
                        default='t5')
    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        choices=['float16', 'float32', 'bfloat16'],
        help=
        'Target inference dtype. Weights and Computation will be in this dtype, no matter what original dtype the weight checkpoint has.'
    )
    parser.add_argument('--logits_dtype',
                        type=str,
                        default='float32',
                        choices=['float16', 'float32'])

    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--max_batch_size', type=int, default=8)
    parser.add_argument('--max_encoder_input_len', type=int, default=1024)
    parser.add_argument(
        '--max_decoder_input_len',
        type=int,
        default=1,
        help=
        'If you want deocder_forced_input_ids feature, set to value greater than 1. Otherwise, encoder-decoder model start from decoder_start_token_id of length 1'
    )
    parser.add_argument('--max_output_len', type=int, default=200)
    parser.add_argument('--max_beam_width', type=int, default=1)
    parser.add_argument(
        '--use_bert_attention_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates BERT attention plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_gpt_attention_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates attention plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_gemm_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates GEMM plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_layernorm_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates layernorm plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_rmsnorm_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates rmsnorm plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_lookup_plugin',
        nargs='?',
        const=None,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help="Activates the lookup plugin which enables embedding sharding.")
    parser.add_argument('--enable_qk_half_accum',
                        default=False,
                        action='store_true')
    parser.add_argument('--builder_opt', type=int, default=None)
    parser.add_argument('--remove_input_padding',
                        default=False,
                        action='store_true')
    parser.add_argument(
        '--random_seed',
        type=int,
        default=None,
        help=
        'Seed to use when initializing the random number generator for torch.')
    parser.add_argument(
        '--max_prompt_embedding_table_size',
        type=int,
        default=0,
        help='Setting to a value > 0 enables support for prompt tuning.')
    parser.add_argument(
        '--use_parallel_embedding',
        action="store_true",
        default=False,
        help=
        'By default embedding parallelism is disabled. By setting this flag, embedding parallelism is enabled'
    )
    parser.add_argument(
        '--embedding_sharding_dim',
        type=int,
        default=0,
        choices=[0, 1],
        help=
        'By default the embedding lookup table is sharded along vocab dimension (embedding_sharding_dim=0). '
        'To shard it along hidden dimension, set embedding_sharding_dim=1'
        'Note: embedding sharding is only enabled when embedding_sharding_dim = 0'
    )
    parser.add_argument(
        '--use_custom_all_reduce',
        action='store_true',
        help=
        'Activates latency-optimized algorithm for all-reduce instead of NCCL.')
    parser.add_argument(
        '--strongly_typed',
        default=False,
        action="store_true",
        help=
        'This option is introduced with trt 9.1.0.1+ and will reduce the building time significantly for fp8.'
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

    # parse cmdline args
    args = parser.parse_args()
    logger.set_level(args.log_level)

    # parse model config and add to args
    if args.weight_dir is not None:
        logger.info(f"Setting model configuration from {args.weight_dir}.")
        args = parse_config(
            Path(args.weight_dir) / "config.ini", component, args)

    assert args.pp_size * args.tp_size == args.world_size

    plugins_args = [
        'use_bert_attention_plugin', 'use_gpt_attention_plugin',
        'use_gemm_plugin', 'use_layernorm_plugin', 'use_rmsnorm_plugin',
        'use_lookup_plugin'
    ]
    for plugin_arg in plugins_args:
        if getattr(args, plugin_arg) is None:
            logger.info(
                f"{plugin_arg} set, without specifying a value. Using {args.dtype} automatically."
            )
            setattr(args, plugin_arg, args.dtype)

    if args.dtype == 'bfloat16':
        assert args.use_gemm_plugin, "Please use gemm plugin when dtype is bfloat16"

    if args.gather_all_token_logits:
        args.gather_context_logits = True
        args.gather_generation_logits = True

    return args


def build_rank_engine(builder: Builder,
                      builder_config: tensorrt_llm.builder.BuilderConfig,
                      engine_name, rank, args):
    '''
       @brief: Build the engine on the given rank.
       @param rank: The rank to build the engine.
       @param args: The cmd line arguments.
       @return: The built engine.
    '''
    dtype = str_dtype_to_trt(args.dtype)

    mapping = Mapping(world_size=args.world_size,
                      rank=rank,
                      tp_size=args.tp_size,
                      pp_size=args.pp_size)

    assert args.n_layer % args.pp_size == 0, \
        f"num_layers {args.n_layer} must be a multiple of pipeline parallelism size {args.pp_size}"

    # Initialize Module
    if args.component == 'encoder':
        tllm_model = tensorrt_llm.models.EncoderModel(
            num_layers=args.n_layer,
            num_heads=args.n_head,
            num_kv_heads=args.n_head,
            head_size=args.head_size,
            hidden_size=args.hidden_size,
            ffn_hidden_size=args.ffn_hidden_size,
            vocab_size=args.vocab_size,
            max_position_embeddings=args.n_positions,
            has_position_embedding=args.has_position_embedding,
            relative_attention=args.relative_attention,
            max_distance=args.max_distance,
            num_buckets=args.num_buckets,
            has_embedding_layernorm=args.has_embedding_layernorm,
            has_embedding_scale=args.has_embedding_scale,
            q_scaling=args.q_scaling,
            has_attention_qkvo_bias=args.has_attention_qkvo_bias,
            has_mlp_bias=args.has_mlp_bias,
            has_model_final_layernorm=args.has_model_final_layernorm,
            layernorm_eps=args.layernorm_eps,
            layernorm_position=args.layernorm_position,
            layernorm_type=args.layernorm_type,
            hidden_act=args.hidden_act,
            mlp_type=args.mlp_type,
            dtype=dtype,
            use_prompt_tuning=args.max_prompt_embedding_table_size > 0,
            use_parallel_embedding=args.use_parallel_embedding,
            embedding_sharding_dim=args.embedding_sharding_dim,
            mapping=mapping)
    elif args.component == 'decoder':
        tllm_model = tensorrt_llm.models.DecoderModel(
            num_layers=args.n_layer,
            num_heads=args.n_head,
            num_kv_heads=args.n_head,
            head_size=args.head_size,
            hidden_size=args.hidden_size,
            ffn_hidden_size=args.ffn_hidden_size,
            encoder_hidden_size=args.encoder_hidden_size,
            encoder_num_heads=args.encoder_num_heads,
            encoder_head_size=args.encoder_head_size,
            vocab_size=args.vocab_size,
            max_position_embeddings=args.n_positions,
            has_position_embedding=args.has_position_embedding,
            relative_attention=args.relative_attention,
            max_distance=args.max_distance,
            num_buckets=args.num_buckets,
            has_embedding_layernorm=args.has_embedding_layernorm,
            has_embedding_scale=args.has_embedding_scale,
            q_scaling=args.q_scaling,
            has_attention_qkvo_bias=args.has_attention_qkvo_bias,
            has_mlp_bias=args.has_mlp_bias,
            has_model_final_layernorm=args.has_model_final_layernorm,
            layernorm_eps=args.layernorm_eps,
            layernorm_position=args.layernorm_position,
            layernorm_type=args.layernorm_type,
            hidden_act=args.hidden_act,
            mlp_type=args.mlp_type,
            use_parallel_embedding=args.use_parallel_embedding,
            embedding_sharding_dim=args.embedding_sharding_dim,
            mapping=mapping,
            rescale_before_lm_head=args.rescale_before_lm_head,
            dtype=dtype,
            logits_dtype=args.logits_dtype)

    # No support for relative attention bias in plain TRT mode. Please use attention plugin
    # (If to add such support, need to add into
    #   Attention and BertAttention at tensorrt_llm/layers/attention.py)
    if args.relative_attention:
        assert args.use_bert_attention_plugin, "Relative attention bias is only supported when using BertAttention Plugin"
        assert args.use_gpt_attention_plugin, "Relative attention bias is only supported when using GPTAttention Plugin"

    if args.weight_from_pytorch_ckpt:
        assert args.tp_size == 1, "Loading from framework model via memory is for demonstration purpose. For multi-GPU inference, please use loading from binary for better performance."
        globals()[f'load_from_hf_{args.model_type}'](tllm_model,
                                                     args.weight_dir,
                                                     args.component,
                                                     dtype=args.dtype)
    else:
        globals()[f'load_from_binary_{args.model_type}'](tllm_model,
                                                         args.weight_dir,
                                                         args,
                                                         mapping=mapping,
                                                         dtype=args.dtype)

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name
    if args.use_bert_attention_plugin:
        network.plugin_config.set_bert_attention_plugin(
            dtype=args.use_bert_attention_plugin)
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(
            dtype=args.use_gpt_attention_plugin)
    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
    if args.use_layernorm_plugin:
        network.plugin_config.set_layernorm_plugin(
            dtype=args.use_layernorm_plugin)
    if args.use_rmsnorm_plugin:
        network.plugin_config.set_rmsnorm_plugin(dtype=args.use_rmsnorm_plugin)
    if args.enable_qk_half_accum:
        network.plugin_config.enable_qk_half_accum()
    if args.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()
    if args.use_lookup_plugin:
        # Use the plugin for the embedding parallelism and sharding
        network.plugin_config.set_lookup_plugin(dtype=args.dtype)
    if args.world_size > 1:
        network.plugin_config.set_nccl_plugin(args.dtype,
                                              args.use_custom_all_reduce)

    with net_guard(network):
        # Prepare
        network.set_named_parameters(tllm_model.named_parameters())

        # Forward
        if args.component == 'encoder':
            inputs = tllm_model.prepare_inputs(
                args.max_batch_size,
                args.max_encoder_input_len,
                args.max_prompt_embedding_table_size,
            )
        elif args.component == 'decoder':
            inputs = tllm_model.prepare_inputs(
                args.max_batch_size,
                args.max_beam_width,
                args.max_decoder_input_len,
                args.max_output_len,
                args.max_encoder_input_len,
                gather_context_logits=args.gather_context_logits,
                gather_generation_logits=args.gather_generation_logits)

        tllm_model(*inputs)

        # Adding debug outputs into the network --------------------------
        if args.debug_mode:
            for k, v in tllm_model.named_network_outputs():
                network._mark_output(v, k,
                                     tensorrt_llm.str_dtype_to_trt(args.dtype))
        # ----------------------------------------------------------------

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)

    return engine


def build(rank, args):
    torch.cuda.set_device(rank % args.gpus_per_node)
    tensorrt_llm.logger.set_level(args.log_level)
    component_dir = args.output_dir / args.dtype / f"tp{args.tp_size}" / args.component
    component_dir.mkdir(parents=True, exist_ok=True)

    builder = Builder()
    apply_query_key_layer_scaling = False

    cache = None
    for cur_rank in range(args.world_size):
        # skip other ranks if parallel_build is enabled
        if args.parallel_build and cur_rank != rank:
            continue
        builder_config = builder.create_builder_config(
            name=args.engine_name,
            precision=args.dtype,
            timing_cache=component_dir /
            args.timing_cache if cache is None else cache,
            profiling_verbosity=args.profiling_verbosity,
            tensor_parallel=args.tp_size,
            pipeline_parallel=args.pp_size,
            gpus_per_node=args.gpus_per_node,
            parallel_build=args.parallel_build,
            num_layers=args.n_layer,
            num_heads=args.n_head,
            hidden_size=args.hidden_size,
            head_size=args.head_size,
            vocab_size=args.vocab_size,
            hidden_act=args.hidden_act,
            max_position_embeddings=args.n_positions,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            max_batch_size=args.max_batch_size,
            max_beam_width=args.max_beam_width,
            max_decoder_input_len=args.max_decoder_input_len,
            max_output_len=args.max_output_len,
            max_encoder_input_len=args.max_encoder_input_len,
            opt_level=args.builder_opt,
            cross_attention=(args.component == 'decoder'),
            has_position_embedding=args.has_position_embedding,
            has_token_type_embedding=args.has_token_type_embedding,
            strongly_typed=args.strongly_typed,
            gather_context_logits=args.gather_context_logits,
            gather_generation_logits=args.gather_generation_logits,
            max_prompt_embedding_table_size=(
                args.max_prompt_embedding_table_size
                if args.component == 'encoder' else 0))

        engine_name = get_engine_name(args.engine_name, args.dtype,
                                      args.tp_size, args.pp_size, cur_rank)
        engine = build_rank_engine(builder, builder_config, engine_name,
                                   cur_rank, args)
        assert engine is not None, f'Failed to build engine for rank {cur_rank}'

        if cur_rank == 0:
            # save build config
            config_path = component_dir / 'config.json'
            builder.save_config(builder_config, config_path)

            # Use in-memory timing cache for multiple builder passes.
            if not args.parallel_build:
                cache = builder_config.trt_builder_config.get_timing_cache()

        serialize_engine(engine, component_dir / engine_name)

    if rank == 0:
        # save timing cache to speedup future use
        ok = builder.save_timing_cache(builder_config,
                                       component_dir / args.timing_cache)
        assert ok, "Failed to save timing cache."


def run_build(component):
    assert component == 'encoder' or component == 'decoder', 'Unsupported component!'
    args = parse_arguments(component)
    args.component = component

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)

    logger.set_level(args.log_level)
    tik = time.time()

    if args.parallel_build and args.world_size > 1 and \
            torch.cuda.device_count() >= args.world_size:
        logger.warning(
            f'Parallelly build TensorRT engines. Please make sure that all of the {args.world_size} GPUs are totally free.'
        )
        mp.spawn(build, nprocs=args.world_size, args=(args, ))
    else:
        args.parallel_build = False
        logger.info('Serially build TensorRT engines.')
        build(0, args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Total time of building all engines: {t}')


if __name__ == '__main__':
    run_build(component='encoder')
    run_build(component='decoder')
