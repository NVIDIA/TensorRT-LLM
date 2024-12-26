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
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import tensorrt_llm
from tensorrt_llm._utils import release_gc
from tensorrt_llm.layers import MoeConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import DeepseekV2ForCausalLM
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None, required=True)
    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')
    parser.add_argument('--pp_size',
                        type=int,
                        default=1,
                        help='N-way pipeline parallelism size')
    parser.add_argument(
        '--moe_tp_size',
        type=int,
        default=-1,
        help=
        'N-way tensor parallelism size for MoE, default is tp_size, which will do tp-only for MoE'
    )
    parser.add_argument(
        '--moe_ep_size',
        type=int,
        default=-1,
        help=
        'N-way expert parallelism size for MoE, default is 1, which will do tp-only for MoE'
    )
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--load_model_on_cpu',
                        default=False,
                        action="store_true",
                        help='Choose to load HF cpkt into CPU')
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
        'By default the embedding lookup table is sharded along vocab dimension (embedding_sharding_dim=0)'
        'To shard it along hidden dimension, set embedding_sharding_dim=1'
        'Note: embedding sharing is only enabled when embedding_sharding_dim=0')

    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
        'See --weight_only_precision to set the precision')
    parser.add_argument(
        '--weight_only_precision',
        const='int8',
        type=str,
        nargs='?',
        default='int8',
        choices=['int8', 'int4'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )

    parser.add_argument('--output_dir',
                        type=str,
                        default='trtllm_checkpoint',
                        required=True,
                        help='The path to save the TensorRT-LLM checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    parser.add_argument(
        '--moe_num_experts',
        type=int,
        default=0,
        help='Specify the number of experts to use for MOE layers')
    parser.add_argument(
        '--moe_top_k',
        type=int,
        default=0,
        help=
        'Specify the top_k value to use for MOE layers. Default to 1 if --moe_num_experts is set'
    )
    parser.add_argument(
        '--moe_renorm_mode',
        type=int,
        default=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
        help=
        'Controls renormalization after gate logits. Check layers/moe.py for accepted values'
    )
    parser.add_argument(
        '--save_config_only',
        action="store_true",
        default=False,
        help=
        'Only save the model config w/o read and converting weights, be careful, this is for debug only'
    )
    parser.add_argument(
        '--disable_weight_only_quant_plugin',
        default=False,
        action="store_true",
        help=
        'By default, using plugin implementation for weight quantization. Enabling disable_weight_only_quant_plugin flag will use ootb implementation instead of plugin.'
        'You must also use --use_weight_only for that argument to have an impact'
    )
    # Add quantization related feature later
    args = parser.parse_args()

    return args


def precision_to_config(precision, quant_config) -> QuantConfig:
    '''update config dict for weight-only quantization
    '''
    quant_config = QuantConfig()
    precision_to_algo = {'int8': QuantAlgo.W8A16, 'int4': QuantAlgo.W4A16}
    quant_config.quant_algo = precision_to_algo.get(precision)
    return quant_config


def args_to_quant_config(args: argparse.Namespace) -> QuantConfig:
    '''return config dict with quantization info based on the command line args
    '''
    quant_config = QuantConfig()
    if args.use_weight_only:
        quant_config = precision_to_config(args.weight_only_precision,
                                           quant_config)
    return quant_config


def args_to_build_options(args):
    return {
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'load_model_on_cpu': args.load_model_on_cpu
    }


def execute(workers, func, args):
    if workers == 1:
        for rank, f in enumerate(func):
            f(args, rank)
    else:
        with ThreadPoolExecutor(max_workers=workers) as p:
            futures = [p.submit(f, args, rank) for rank, f in enumerate(func)]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    exceptions.append(e)
            assert len(
                exceptions
            ) == 0, "Checkpoint conversion failed, please check error log."


def convert_and_save_hf(args):
    model_dir = args.model_dir
    world_size = args.tp_size * args.pp_size
    # Need to convert the cli args to the kay-value pairs and override them in the generate config dict.
    # Ideally these fields will be moved out of the config and pass them into build API, keep them here for compatibility purpose for now,
    # before the refactor is done.
    override_fields = {}
    override_fields.update(args_to_build_options(args))

    quant_config = args_to_quant_config(args)

    def convert_and_save_rank(args, rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size,
                          moe_tp_size=args.moe_tp_size,
                          moe_ep_size=args.moe_ep_size)

        deepseekv2 = DeepseekV2ForCausalLM.from_hugging_face(
            model_dir,
            args.dtype,
            mapping=mapping,
            quant_config=quant_config,
            **override_fields)
        deepseekv2.save_checkpoint(args.output_dir, save_config=(rank == 0))
        del deepseekv2

    execute(args.workers, [convert_and_save_rank] * world_size, args)
    release_gc()


def main():
    print(tensorrt_llm.__version__)
    args = parse_arguments()

    if (args.moe_tp_size == -1 and args.moe_ep_size == -1):
        # moe default to tp-only
        args.moe_tp_size = args.tp_size
        args.moe_ep_size = 1
    elif (args.moe_tp_size == -1):
        args.moe_tp_size = args.tp_size // args.moe_ep_size
    elif (args.moe_ep_size == -1):
        args.moe_ep_size = args.tp_size // args.moe_tp_size
    assert (args.moe_tp_size * args.moe_ep_size == args.tp_size
            ), "moe_tp_size * moe_ep_size must equal to tp_size"

    tik = time.time()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    assert args.model_dir is not None
    convert_and_save_hf(args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
