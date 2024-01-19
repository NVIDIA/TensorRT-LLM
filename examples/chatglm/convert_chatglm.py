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
import json
import os
import platform
from pathlib import Path

import torch
import torch.multiprocessing as multiprocessing
import transformers
from convert import split_and_save_weight, str_to_np_dtype
from datasets import load_dataset
from smoothquant import capture_activation_range, smooth_gemm
from tqdm import tqdm
from transformers import AutoModelForCausalLM  # transformers-4.10.0-py3

from tensorrt_llm.logger import logger


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        '-i',
        type=Path,
        default=None,
        help='Path of model files from HF',
    )
    parser.add_argument(
        '--output_dir',
        '-o',
        type=Path,
        required="smooth",
        help='Path to save smooth quantized checkpoint',
    )
    parser.add_argument(
        '--tp_size',
        '-tp',
        type=int,
        default=1,
        help='Tensor parallelism size',
    )
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=4,
        help='Number of processes to spawn for conversion (default: 4)',
    )
    parser.add_argument(
        "--calibrate_kv_cache",
        "-kv",
        action="store_true",
        help='Calibrate for KV cache into Int8',
    )
    parser.add_argument(
        "--smoothquant",
        "-sq",
        type=float,
        default=None,
        help='Parameter α for smooth quantization, must be in [0, 1]',
    )
    parser.add_argument(
        '--dtype',
        '-t',
        type=str,
        default='float16',
        choices=['float32', 'float16', 'bfloat16'],
        help='Data type of output checkpoint',
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='dataset/',
        help='Directory of dataset cache for calibration',
    )
    parser.add_argument(
        '--calib_size',
        type=int,
        default=64,
        help='Number of samples for calibration.',
    )
    parser.add_argument(
        "--on_cpu",
        action="store_true",
        help='Do calibration on CPU to save small GPU memory but much slower',
    )

    args = parser.parse_args(args)

    logger.set_level("info")

    if args.processes > 1 and platform.system() == "Windows":
        logger.info("Multi-process on Windows is not implemented, using 1")
        args.processes = 1

    logger.info(' Build Arguments '.center(100, '='))
    for k, v in vars(args).items():
        logger.info(f' - {k.ljust(40, ".")}: {v}')
    logger.info('=' * 100)

    return args


@torch.no_grad()
def smooth_chatglm_model(
    model,
    act_range,
    alpha,
    model_smoother,
):
    for name, module in model.named_modules():
        if not module._get_name() == "GLMBlock":
            continue

        # QKV multiplication weight
        layer_name = name + '.self_attention.query_key_value'
        logger.debug(f'Smoothing module: {layer_name}')
        weight = module.self_attention.query_key_value.weight
        smoother = smooth_gemm(
            weight,
            act_range[layer_name]["x"],
            module.input_layernorm.weight,
            None,
            alpha,
        )
        act_range[layer_name]["x"] = act_range[layer_name]["x"] / smoother
        act_range[layer_name]["w"] = weight.abs().max(dim=1)[0]

        # Dense multiplication weight
        layer_name = name + ".self_attention.dense"
        logger.debug(f'Smoothing module: {layer_name}')
        weight = module.self_attention.dense.weight
        smoother = smooth_gemm(
            weight,
            act_range[layer_name]["x"],
            None,
            None,
            alpha,
        )
        model_smoother[layer_name] = smoother.float()
        act_range[layer_name]["x"] = act_range[layer_name]["x"] / smoother
        act_range[layer_name]["w"] = weight.abs().max(dim=1)[0]

        # Multilayer perceptron h -> 4h weight
        layer_name = name + ".mlp.dense_h_to_4h"
        logger.debug(f'Smoothing module: {layer_name}')
        weight = module.mlp.dense_h_to_4h.weight
        smoother = smooth_gemm(
            weight,
            act_range[layer_name]["x"],
            module.post_attention_layernorm.weight,
            None,
            alpha,
        )
        act_range[layer_name]["x"] = act_range[layer_name]["x"] / smoother
        act_range[layer_name]["w"] = weight.abs().max(dim=1)[0]

        # Multilayer perceptron 4h -> h weight
        layer_name = name + ".mlp.dense_4h_to_h"
        logger.debug(f'Smoothing module: {layer_name}')
        weight = module.mlp.dense_4h_to_h.weight
        smoother = smooth_gemm(
            weight,
            act_range[layer_name]["x"],
            None,
            None,
            alpha,
        )
        model_smoother[layer_name] = smoother.float()
        act_range[layer_name]["x"] = act_range[layer_name]["x"] / smoother
        act_range[layer_name]["w"] = weight.abs().max(dim=1)[0]


def get_bin_name(name):
    weights_global = {
        "transformer.embedding.word_embeddings.weight": "embedding.weight",
        "transformer.encoder.final_layernorm.weight": "final_norm.weight",
        "transformer.output_layer.weight": "lm_head.weight",
    }

    if name in weights_global.keys():
        return weights_global[name]

    weights_in_layer = {
        "input_layernorm.weight": "pre_norm.weight",
        "self_attention.query_key_value.weight":
        "attention.query_key_value.weight",
        "self_attention.query_key_value.bias": "attention.query_key_value.bias",
        "self_attention.dense.weight": "attention.dense.weight",
        "post_attention_layernorm.weight": "post_norm.weight",
        "mlp.dense_h_to_4h.weight": "mlp.fc.weight",
        "mlp.dense_4h_to_h.weight": "mlp.proj.weight",
    }

    _, _, _, layer_id, *weight_name = name.split(".")
    weight_name = ".".join(weight_name)

    if weight_name in weights_in_layer.keys():
        return f"layers.{layer_id}.{weights_in_layer[weight_name]}"

    logger.error(f"Error to convert HF name {name}")
    return None


@torch.no_grad()
def hf_chatglm_converter(args):
    multi_query_mode = True
    output_dir = Path(args.output_dir) / f"{args.tp_size}-gpu"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    config = {}
    for k, v in dict(**vars(args), **vars(model.config)).items():
        config[k] = v if type(v) in [int, float, bool, type(None)] else f"{v}"
    config["storage_dtype"] = args.dtype
    with open(output_dir / "config.json", "w") as f:
        f.write(json.dumps(config))

    if args.on_cpu:
        model = model.float()
        model = model.cpu()
        torch.cuda.empty_cache()
    else:
        model = model.cuda()

    act_range = {}
    if args.smoothquant is not None or args.calibrate_kv_cache:
        os.environ["TOKENIZERS_PARALLELISM"] = \
            os.environ.get("TOKENIZERS_PARALLELISM", "false")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_dir,
            trust_remote_code=True,
        )
        dataset = load_dataset(
            "cnn_dailymail",
            '3.0.0',
            split="validation",
            cache_dir=args.cache_dir,
        )

        act_range = capture_activation_range(
            model,
            tokenizer,
            dataset,
            num_samples=args.calib_size,
        )

        model_smoother = {}  # smoother for query_key_value.dense and mlp.proj
        if args.smoothquant is not None:
            smooth_chatglm_model(
                model,
                act_range,
                args.smoothquant,
                model_smoother,
            )

    weights_global = [
        "embedding.weight",
        "final_norm.weight",
        "lm_head.weight",
    ]
    weights_need_transpose = [
        "self_attention.query_key_value.weight",
        "self_attention.dense.weight",
        "mlp.dense_h_to_4h.weight",
        "mlp.dense_4h_to_h.weight",
    ]

    int8_outputs = None
    if args.calibrate_kv_cache:
        int8_outputs = "kv_cache_only"
    if args.smoothquant is not None:
        int8_outputs = "all"
    arg_dict = {
        "int8_outputs": int8_outputs,
        "multi_query_mode": multi_query_mode,
    }

    starmap_args = []
    for name, param in model.named_parameters():
        if "weight" not in name and "bias" not in name:
            continue

        bin_name = get_bin_name(name)

        if name.replace(".weight", "") in model_smoother.keys():
            key = bin_name.replace(".weight", "") + ".smoother"
            smoother = model_smoother[name.replace(".weight", "")]
            smoother = smoother.detach().cpu().numpy()
            task = (0, output_dir, args.tp_size, key, smoother, None, arg_dict)
            starmap_args.append(task)

        if len(param.shape) == 2 and \
            any([k in name for k in weights_need_transpose]):
            param = param.transpose(0, 1)

        param = param.detach().cpu().numpy().astype(str_to_np_dtype(args.dtype))

        if bin_name in weights_global:
            param.tofile(output_dir / f"{bin_name}.bin")
            continue

        act = act_range.get(name.replace(".weight", ""))
        task = (0, output_dir, args.tp_size, bin_name, param, act, arg_dict)
        starmap_args.append(task)

    starmap_args = tqdm(starmap_args, desc="Export")
    if args.processes > 1:
        with multiprocessing.Pool(args.processes) as pool:
            pool.starmap(split_and_save_weight, starmap_args)
    else:  # for debug usage
        for starmap_arg in starmap_args:
            print(starmap_arg[3])
            split_and_save_weight(*starmap_arg)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    args = parse_arguments(None)
    hf_chatglm_converter(args)
