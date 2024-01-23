# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.medusa.weight import load_medusa_hf
from tensorrt_llm.network import net_guard


def dynamic_import_from_relative_path(module_name, relative_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_file = os.path.join(current_dir, relative_path)

    spec = importlib.util.spec_from_file_location(module_name, path_to_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# TODO: we use dynamic load here as examples are not in package. Maybe better solution?
trtllm_example_llama_build = dynamic_import_from_relative_path(
    "trtllm_example_llama_build", "../llama/build.py")
llama_builder_config = trtllm_example_llama_build.get_builder_config_namespace
get_engine_name = trtllm_example_llama_build.get_engine_name
get_llama_model = trtllm_example_llama_build.get_model_object
llama_parse_args = trtllm_example_llama_build.parse_arguments
serialize_engine = trtllm_example_llama_build.serialize_engine
to_onnx = trtllm_example_llama_build.to_onnx
llama_update_plugin_configs = trtllm_example_llama_build.update_plugin_configs

MODEL_NAME = "medusa"

# 2 routines: get_engine_name, serialize_engine
# are direct copy from gpt example, TODO: put in utils?


def parse_arguments():
    parser = argparse.ArgumentParser()
    # add medusa specific params
    parser.add_argument('--num_medusa_heads', type=int, default=4)
    parser.add_argument(
        '--fixed_num_medusa_heads',
        type=int,
        default=None,
        help="If exist, fix medusa_num_heads from config.json."
        "num_medusa_heads < medusa_num_heads in config.json < fixed_num_medusa_heads"
    )
    parser.add_argument('--num_medusa_layers', type=int, default=1)
    parser.add_argument('--max_medusa_token_len', type=int, default=63)
    parser.add_argument('--medusa_hidden_act', type=str, default="silu")
    parser.add_argument('--base_model_name', type=str, default="llama")
    parser.add_argument('--medusa_model_dir', type=str, default=None)
    args, extra = parser.parse_known_args()
    # now get the base model args
    # TODO: add more elif paths for other base models
    if args.base_model_name == 'llama':
        base_model_args = llama_parse_args(extra)
    args = argparse.Namespace(**vars(args), **vars(base_model_args))
    if args.medusa_model_dir is not None:
        config_file = Path(args.medusa_model_dir) / "config.json"
        with open(config_file) as fp:
            config = json.load(fp)
        args.num_medusa_heads = config.get('medusa_num_heads',
                                           args.num_medusa_heads)
        args.num_medusa_layers = config.get('medusa_num_layers',
                                            args.num_medusa_layers)
        if args.fixed_num_medusa_heads is not None and args.fixed_num_medusa_heads != args.num_medusa_heads:
            logger.info(
                f"fixing num_medusa_heads from {args.num_medusa_heads} to {args.fixed_num_medusa_heads}"
            )
            args.num_medusa_heads = args.fixed_num_medusa_heads
    assert args.max_beam_width == 1, "Medusa only supports max_beam_width = 1 now, if need beam search, please use example/llama/build.py."
    assert args.use_inflight_batching == False, "Medusa doesn't support inflight batching now."
    assert args.max_medusa_token_len > 0, "should have max_medusa_token_len > 0"

    # Explicitly set max_num_tokens now for tests (default is 4096).
    # FIXME: remove this when the max_num_tokens issue is fixed.
    args.max_num_tokens = args.max_beam_width * args.max_batch_size * max(
        args.max_input_len, args.max_medusa_token_len + 1)

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
    base_model = get_llama_model(args, mapping=mapping, trt_dtype=dtype)
    tensorrt_llm_medusa = tensorrt_llm.models.MedusaLM(
        base_model=base_model,
        mapping=mapping,
        num_medusa_heads=args.num_medusa_heads,
        num_medusa_layers=args.num_medusa_layers,
        hidden_act=args.medusa_hidden_act)
    if args.medusa_model_dir is not None and mapping.is_last_pp_rank():
        load_medusa_hf(args.medusa_model_dir,
                       tensorrt_llm_medusa,
                       mapping=mapping,
                       dtype=args.dtype)

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name
    llama_update_plugin_configs(args, network)

    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_medusa.named_parameters())
        # Forward
        inputs = tensorrt_llm_medusa.prepare_inputs(
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_seq_len=args.max_input_len + args.max_output_len,
            use_cache=True,
            max_medusa_tokens_len=args.max_medusa_token_len,
            max_beam_width=args.max_beam_width,
            max_num_tokens=args.max_num_tokens,
            prompt_embedding_table_size=args.max_prompt_embedding_table_size,
        )
        print([
            t.name if isinstance(t, tensorrt_llm.Tensor) else t for t in inputs
        ])
        tensorrt_llm_medusa(*inputs)
        if args.enable_debug_output:
            # mark intermediate nodes' outputs
            for k, v in tensorrt_llm_medusa.named_network_outputs():
                v = v.trt_tensor
                v.name = k
                network.trt_network.mark_output(v)
                v.dtype = dtype
        if args.visualize:
            model_path = os.path.join(args.output_dir, 'test.onnx')
            to_onnx(network.trt_network, model_path)

    tensorrt_llm.graph_rewriting.optimize(network)

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    if rank == 0:
        config_path = os.path.join(args.output_dir, 'config.json')
        builder.save_config(builder_config, config_path)
    return engine


def build(rank, args):
    torch.cuda.set_device(rank % args.gpus_per_node)
    logger.set_level(args.log_level)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # when doing serializing build, all ranks share one engine
    builder = Builder()

    cache = None
    for cur_rank in range(args.world_size):
        # skip other ranks if parallel_build is enabled
        if args.parallel_build and cur_rank != rank:
            continue
        medusa_config_ns = argparse.Namespace(
            num_medusa_heads=args.num_medusa_heads,
            num_medusa_layers=args.num_medusa_layers,
            medusa_hidden_act=args.medusa_hidden_act,
            max_medusa_token_len=args.max_medusa_token_len,
        )
        if args.base_model_name == 'llama':
            config_ns = llama_builder_config(args, cache)
        medusa_config_ns.base_model_name = config_ns.name
        config_ns.name = f"{MODEL_NAME}_{config_ns.name}"  # override_the model name
        config_ns = argparse.Namespace(**vars(medusa_config_ns),
                                       **vars(config_ns))
        builder_config = builder.create_builder_config(**vars(config_ns))

        engine_name = get_engine_name(config_ns.name, args.dtype, args.tp_size,
                                      args.pp_size, cur_rank)

        engine = build_rank_engine(builder, builder_config, engine_name,
                                   cur_rank, args)
        assert engine is not None, f'Failed to build engine for rank {cur_rank}'

        if cur_rank == 0:
            # Use in-memory timing cache for multiple builder passes.
            if not args.parallel_build:
                cache = builder_config.trt_builder_config.get_timing_cache()

        serialize_engine(engine, os.path.join(args.output_dir, engine_name))

    if rank == 0:
        ok = builder.save_timing_cache(
            builder_config, os.path.join(args.output_dir, "model.cache"))
        assert ok, "Failed to save timing cache."


if __name__ == '__main__':
    args = parse_arguments()
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
    logger.info(f'Total time of building all {args.world_size} engines: {t}')
