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

from transformers import AutoConfig

import tensorrt_llm
from tensorrt_llm.models import Phi3ForCausalLM, PhiForCausalLM


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
    modelForCausalLM = None
    if model_type not in supported_models:
        assert False, "Invalid model type"
    modelForCausalLM = PhiForCausalLM if model_type == 'PhiForCausalLM' else Phi3ForCausalLM

    modelForCausalLM.convert_hf_checkpoint(args.model_dir,
                                           dtype=args.dtype,
                                           output_dir=args.output_dir,
                                           args=args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')
