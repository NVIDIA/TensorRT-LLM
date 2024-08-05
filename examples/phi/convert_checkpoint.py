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

from transformers import AutoConfig

import tensorrt_llm
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import Phi3ForCausalLM, PhiForCausalLM
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')
    parser.add_argument('--pp_size',
                        type=int,
                        default=1,
                        help='N-way pipeline parallelism size')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
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
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT-LLM checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    args = parser.parse_args()

    return args


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


def args_to_quant_config(args: argparse.Namespace) -> QuantConfig:
    '''return config dict with quantization info based on the command line args
    '''
    quant_config = QuantConfig()
    if args.use_weight_only:
        if args.weight_only_precision == 'int8':
            quant_config.quant_algo = QuantAlgo.W8A16
        elif args.weight_only_precision == 'int4':
            quant_config.quant_algo = QuantAlgo.W4A16

    return quant_config


if __name__ == '__main__':
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    assert args.pp_size == 1, "Pipeline parallelism is not supported."

    tik = time.time()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model_config = AutoConfig.from_pretrained(args.model_dir,
                                              trust_remote_code=True)
    model_type = model_config.architectures[0]
    supported_models = [
        'PhiForCausalLM', 'Phi3ForCausalLM', 'Phi3VForCausalLM',
        'Phi3SmallForCausalLM'
    ]

    if model_type not in supported_models:
        assert False, "Invalid model type"

    phi_model = Phi3ForCausalLM if model_type.find(
        'Phi3') != -1 else PhiForCausalLM

    hf_model = None

    override_fields = {}
    # override_fields.update(args_to_build_options(args))
    quant_config = args_to_quant_config(args)

    def convert_and_save_rank(args, rank):
        mapping = Mapping(world_size=args.tp_size * args.pp_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)

        phi = phi_model.from_hugging_face(
            args.model_dir if hf_model is None else hf_model,
            args.dtype,
            mapping=mapping,
            quant_config=quant_config,
            **override_fields,
        )
        phi.save_checkpoint(args.output_dir, save_config=(rank == 0))
        del phi

    execute(args.workers, [convert_and_save_rank] * args.tp_size * args.pp_size,
            args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')
