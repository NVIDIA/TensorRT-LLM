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
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.baichuan.config import BaichuanConfig
from tensorrt_llm.models.baichuan.convert import load_weights_from_gptq
from tensorrt_llm.models.baichuan.model import BaichuanForCausalLM
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--quant_ckpt_path', type=str, default=None)
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--pp_size', type=int, default=1)
    parser.add_argument('--model_version',
                        type=str,
                        default='v1_13b',
                        choices=['v1_7b', 'v1_13b', 'v2_7b', 'v2_13b'])
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'float16', 'bfloat16', 'float32'],
        help=
        "The data type for the model weights and activations if not quantized. "
        "If 'auto', the data type is automatically inferred from the source model; "
        "however, if the source dtype is float32, it is converted to float16.")
    parser.add_argument('--logits_dtype',
                        type=str,
                        default='float32',
                        choices=['float16', 'float32'])
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT LLM checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    parser.add_argument(
        '--calib_dataset',
        type=str,
        default='ccdv/cnn_dailymail',
        help=
        "The huggingface dataset name or the local directory of the dataset for calibration."
    )
    parser.add_argument(
        '--per_channel',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor for the GEMM\'s result. '
        'per_channel instead uses a different static scaling factor for each channel. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        '--per_token',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor to scale activations in the int8 range. '
        'per_token chooses at run time, and for each token, a custom scaling factor. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        "--smoothquant",
        "-sq",
        type=float,
        default=None,
        help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
        " to Smoothquant the model, and output int8 weights."
        " A good first try is 0.5. Must be in [0, 1]")
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
        choices=['int8', 'int4', 'int4_gptq'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    parser.add_argument('--group_size',
                        type=int,
                        default=128,
                        help='Group size used in GPTQ/AWQ quantization.')
    parser.add_argument(
        '--int8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV'
    )
    args = parser.parse_args()
    return args


def args_to_quant_config(args: argparse.Namespace) -> QuantConfig:
    config = QuantConfig(group_size=args.group_size)

    if args.smoothquant:
        config.smoothquant_val = args.smoothquant
        if args.per_token and args.per_channel:
            config.quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN
        elif not args.per_token and not args.per_channel:
            config.quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN
        elif not args.per_token and args.per_channel:
            config.quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN
        elif args.per_token and not args.per_channel:
            config.quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN
    else:
        if args.use_weight_only and args.weight_only_precision == 'int8':
            config.quant_algo = QuantAlgo.W8A16
        elif args.use_weight_only and args.weight_only_precision == 'int4':
            config.quant_algo = QuantAlgo.W4A16
        elif args.use_weight_only and args.weight_only_precision == 'int4_gptq':
            config.quant_algo = QuantAlgo.W4A16_GPTQ

    if args.int8_kv_cache:
        config.kv_cache_quant_algo = QuantAlgo.INT8

    if args.use_weight_only and args.weight_only_precision == 'int4_gptq':
        config.has_zero_point = True

    return config


def convert_and_save_hf(args):
    world_size = args.tp_size * args.pp_size
    quantization_config = args_to_quant_config(args)

    if args.smoothquant is not None or args.int8_kv_cache:
        mapping = Mapping(
            world_size=world_size,
            tp_size=args.tp_size,
            pp_size=args.pp_size,
        )
        BaichuanForCausalLM.quantize(
            args.model_dir,
            args.output_dir,
            dtype=args.dtype,
            mapping=mapping,
            quant_config=quantization_config,
            calib_dataset=args.calib_dataset,
            model_version=args.model_version,
        )
    else:

        def convert_and_save_rank(args, rank):
            mapping = Mapping(
                world_size=world_size,
                rank=rank,
                tp_size=args.tp_size,
                pp_size=args.pp_size,
            )

            model = BaichuanForCausalLM.from_hugging_face(
                args.model_dir,
                dtype=args.dtype,
                mapping=mapping,
                quant_config=quantization_config,
                model_version=args.model_version,
                logits_dtype=args.logits_dtype,
            )
            model.save_checkpoint(args.output_dir, save_config=(rank == 0))
            del model

        execute(args.workers, [convert_and_save_rank] * world_size, args)


def convert_and_save_gptq(args, rank):
    mapping = Mapping(world_size=args.tp_size * args.pp_size,
                      rank=rank,
                      tp_size=args.tp_size,
                      pp_size=args.pp_size)
    config = BaichuanConfig.from_hugging_face(
        args.model_dir,
        dtype=args.dtype,
        mapping=mapping,
        quant_config=args_to_quant_config(args),
        model_version=args.model_version,
        logits_dtype=args.logits_dtype)

    config.vocab_size = int((config.vocab_size + 63) / 64) * 64
    model = BaichuanForCausalLM(config)
    weights = load_weights_from_gptq(config, args.quant_ckpt_path)
    model.load(weights)
    model.save_checkpoint(args.output_dir, save_config=(rank == 0))


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


def main():
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    world_size = args.tp_size * args.pp_size
    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.use_weight_only and args.weight_only_precision == 'int4_gptq':
        assert args.model_dir is not None
        assert args.quant_ckpt_path is not None
        execute(args.workers, [convert_and_save_gptq] * world_size, args)
    else:
        assert args.model_dir is not None
        assert args.quant_ckpt_path is None, "only gptq weights only needs this option"
        convert_and_save_hf(args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
