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
import os
import platform
from pathlib import Path

import torch
import torch.multiprocessing as multiprocessing
import transformers
from datasets import load_dataset
from smoothquant import capture_activation_range, smooth_gemm
from tqdm import tqdm
from transformers import AutoModelForCausalLM  # transformers-4.10.0-py3
from utils.convert import split_and_save_weight

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.logger import logger


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        '-m',
        type=str,
        choices=[
            'chatglm_6b', 'chatglm2_6b', 'chatglm2_6b_32k', 'chatglm3_6b',
            'chatglm3_6b_base', 'chatglm3_6b_32k', 'glm_10b'
        ],
        help='Name of model, use "_" rather than "-" to connect the parts',
    )
    parser.add_argument(
        '--out-dir',
        '-o',
        type=str,
        help='file name of output directory',
        required=True,
    )
    parser.add_argument(
        '--in-file',
        '-i',
        type=str,
        help='file name of input checkpoint file',
        required=True,
    )
    parser.add_argument(
        '--tensor-parallelism',
        '-tp',
        type=int,
        help='Requested tensor parallelism for inference',
        default=1,
    )
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        help=
        "How many processes to spawn for conversion (default: 4). Set it to a lower value to reduce RAM usage.",
        default=4,
    )
    parser.add_argument(
        "--calibrate-kv-cache",
        "-kv",
        action="store_true",
        help=
        "Generate scaling factors for KV cache. Used for storing KV cache in int8.",
    )
    parser.add_argument(
        "--smoothquant",
        "-sq",
        type=float,
        default=None,
        help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
        " to Smoothquant the model, and output int8 weights."
        " A good first try is 0.5. Must be in [0, 1]",
    )
    parser.add_argument(
        "--storage-type",
        "-t",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=str,
        default=None,
        help="cache dir to load the hugging face dataset",
    )
    parser.add_argument(
        "--load-model-on-cpu",
        action="store_true",
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
def smooth_chatglm_model(model, scales, alpha):
    for name, module in model.named_modules():
        if not module._get_name() == "GLMBlock":
            continue

        layer_name = name + '.self_attention.query_key_value'
        smoother = smooth_gemm(
            module.self_attention.query_key_value.weight,
            scales[layer_name]["x"],
            module.input_layernorm.weight,
            None,
            alpha,
        )
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name][
            "w"] = module.self_attention.query_key_value.weight.abs().max(
                dim=1)[0]

        # fc1
        layer_name = name + ".mlp.dense_h_to_4h"
        smoother = smooth_gemm(
            module.mlp.dense_h_to_4h.weight,
            scales[layer_name]["x"],
            module.post_attention_layernorm.weight,
            None,
            alpha,
        )
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.dense_h_to_4h.weight.abs().max(
            dim=1)[0]

        # fc2
        layer_name = name + ".mlp.dense_4h_to_h"
        smoother = smooth_gemm(
            module.mlp.dense_4h_to_h.weight,
            scales[layer_name]["x"],
            None,
            None,
            alpha,
        )
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.dense_4h_to_h.weight.abs().max(
            dim=1)[0]


def chatglm_to_bin_name(orig_name):
    global_weights = {
        "transformer.embedding.word_embeddings.weight": "model.wte",
        "transformer.encoder.final_layernorm.weight":
        "model.final_layernorm.weight",
        "transformer.output_layer.weight": "model.lm_head.weight",
    }

    if orig_name in global_weights:
        return global_weights[orig_name]

    _, _, _, layer_id, *weight_name = orig_name.split(".")
    layer_id = int(layer_id)
    weight_name = "transformer." + ".".join(weight_name)

    per_layer_weights = {
        "transformer.input_layernorm.weight": "input_layernorm.weight",
        "transformer.self_attention.query_key_value.bias":
        "attention.query_key_value.bias",
        "transformer.self_attention.query_key_value.weight":
        "attention.query_key_value.weight",
        "transformer.self_attention.dense.bias": "attention.dense.bias",
        "transformer.self_attention.dense.weight": "attention.dense.weight",
        "transformer.post_attention_layernorm.bias":
        "post_attention_layernorm.bias",
        "transformer.post_attention_layernorm.weight":
        "post_attention_layernorm.weight",
        "transformer.mlp.dense_h_to_4h.bias": "mlp.dense_h_to_4h.bias",
        "transformer.mlp.dense_h_to_4h.weight": "mlp.dense_h_to_4h.weight",
        "transformer.mlp.dense_4h_to_h.bias": "mlp.dense_4h_to_h.bias",
        "transformer.mlp.dense_4h_to_h.weight": "mlp.dense_4h_to_h.weight",
    }
    return f"layers.{layer_id}.{per_layer_weights[weight_name]}"


@torch.no_grad()
def hf_chatglm_converter(args):
    infer_tp = args.tensor_parallelism
    multi_query_mode = False
    saved_dir = Path(args.out_dir) / f"{infer_tp}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    # load position_embedding from rank 0
    model = AutoModelForCausalLM.from_pretrained(
        args.in_file,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    if args.load_model_on_cpu:
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
            args.in_file,
            trust_remote_code=True,
        )
        dataset = load_dataset(
            "ccdv/cnn_dailymail",
            '3.0.0',
            split="validation",
            cache_dir=args.dataset_cache_dir,
        )

        act_range = capture_activation_range(
            model,
            tokenizer,
            dataset,
        )
        if args.smoothquant is not None:
            smooth_chatglm_model(model, act_range, args.smoothquant)

    config = configparser.ConfigParser()
    config[args.model_name] = {}
    for key in vars(args):
        config[args.model_name][key] = f"{vars(args)[key]}"
    for k, v in vars(model.config).items():
        config[args.model_name][k] = f"{v}"
    config[args.model_name]["storage_dtype"] = args.storage_type
    config[args.model_name]["multi_query_mode"] = str(multi_query_mode)
    with open(saved_dir / "config.ini", 'w') as configfile:
        config.write(configfile)

    storage_type = str_dtype_to_torch(args.storage_type)

    global_ft_weights = [
        "model.wte",
        "model.final_layernorm.bias",
        "model.final_layernorm.weight",
        "model.lm_head.weight",
    ]

    int8_outputs = None
    if args.calibrate_kv_cache:
        int8_outputs = "kv_cache_only"
    if args.smoothquant is not None:
        int8_outputs = "all"

    starmap_args = []
    for name, param in model.named_parameters():
        if "weight" not in name and "bias" not in name:
            continue
        bin_name = chatglm_to_bin_name(name)

        if any(k in bin_name for k in [
                "query_key_value.weight", "dense.weight",
                "mlp.dense_h_to_4h.weight", "mlp.dense_4h_to_h.weight"
        ]):
            param = param.transpose(0, 1)
        if bin_name in global_ft_weights:
            torch_to_numpy(param.to(storage_type).cpu()).tofile(
                saved_dir / f"{bin_name}.bin")
        else:
            if args.processes == 1:
                split_and_save_weight(
                    0, saved_dir, infer_tp, bin_name, param.to(storage_type),
                    storage_type, act_range.get(name.replace(".weight", "")), {
                        "int8_outputs": int8_outputs,
                        "multi_query_mode": multi_query_mode,
                        "local_dim": None,
                    })
            else:
                starmap_args.append(
                    (0, saved_dir, infer_tp, bin_name, param.to(storage_type),
                     storage_type, act_range.get(name.replace(".weight", "")), {
                         "int8_outputs": int8_outputs,
                         "multi_query_mode": multi_query_mode,
                         "local_dim": None,
                     }))

    starmap_args = tqdm(starmap_args, desc="saving weights")
    if args.processes > 1:
        with multiprocessing.Pool(args.processes) as pool:
            pool.starmap(split_and_save_weight, starmap_args)


if __name__ == "__main__":

    torch.multiprocessing.set_start_method("spawn")

    args = parse_arguments(None)

    hf_chatglm_converter(args)
