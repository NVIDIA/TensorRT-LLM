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
'''
Convert huggingface GPT-J model. Use https://huggingface.co/EleutherAI/gpt-j-6b as demo.
'''
import argparse
import configparser
import dataclasses
import functools
import os
import platform
from collections import defaultdict
from pathlib import Path

import torch
import torch.multiprocessing as multiprocessing
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM  # transformers-4.10.0-py3
from transformers import AutoTokenizer
from transformers.pytorch_utils import Conv1D
from utils.convert import split_and_save_weight

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy


@torch.no_grad()
def capture_activation_range(model,
                             tokenizer,
                             dataset,
                             num_samples=512,
                             seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

    def stat_tensor(name, tensor, act_scales, key):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float()

        if act_scales[name][key] is None:
            act_scales[name][key] = comming_max
        else:
            act_scales[name][key] = torch.max(act_scales[name][key],
                                              comming_max)

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x, act_scales, "x")
        stat_tensor(name, y, act_scales, "y")

        if act_scales[name]["w"] is None:
            act_scales[name]["w"] = m.weight.abs().clip(1e-8,
                                                        None).max(dim=0)[0]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, Conv1D):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples), desc="calibrating model"):
        input_ids = tokenizer(dataset[i]["text"],
                              return_tensors="pt",
                              max_length=seq_len,
                              truncation=True).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales


@dataclasses.dataclass(frozen=True)
class ProgArgs:
    out_dir: str
    in_file: str
    tensor_parallelism: int = 1
    processes: int = 4
    calibrate_kv_cache: bool = False
    model: str = "gpt"
    storage_type: str = "fp32"
    dataset_cache_dir: str = None
    load_model_on_cpu: bool = False
    convert_model_on_cpu: bool = False

    @staticmethod
    def parse(args=None) -> 'ProgArgs':
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--out-dir',
                            '-o',
                            type=str,
                            help='file name of output directory',
                            required=True)
        parser.add_argument('--in-file',
                            '-i',
                            type=str,
                            help='file name of input checkpoint file',
                            required=True)
        parser.add_argument('--tensor-parallelism',
                            '-tp',
                            type=int,
                            help='Requested tensor parallelism for inference',
                            default=1)
        parser.add_argument(
            "--processes",
            "-p",
            type=int,
            help=
            "How many processes to spawn for conversion (default: 4). Set it to a lower value to reduce RAM usage.",
            default=4)
        parser.add_argument(
            "--calibrate-kv-cache",
            "-kv",
            action="store_true",
            help=
            "Generate scaling factors for KV cache. Used for storing KV cache in int8."
        )
        parser.add_argument(
            "--model",
            default="gpt2",
            type=str,
            help="Specify GPT variants to convert checkpoints correctly",
            choices=["gpt2", "santacoder", "starcoder"])
        parser.add_argument("--storage-type",
                            "-t",
                            type=str,
                            default="float32",
                            choices=["float32", "float16", "bfloat16"])
        parser.add_argument("--dataset-cache-dir",
                            type=str,
                            default=None,
                            help="cache dir to load the hugging face dataset")
        parser.add_argument("--load-model-on-cpu", action="store_true")
        parser.add_argument("--convert-model-on-cpu", action="store_true")
        return ProgArgs(**vars(parser.parse_args(args)))


def merge_qkv_scales(q_name, hf_model, scales, gptj_qkv_para):
    layer_name_q = q_name.replace(".weight", "")
    layer_name_k = layer_name_q.replace("q_proj", "k_proj")
    layer_name_v = layer_name_q.replace("q_proj", "v_proj")
    layer_name_qkv = layer_name_q.replace("q_proj", "qkv_proj")

    q = hf_model.state_dict()[layer_name_q + ".weight"]
    k = hf_model.state_dict()[layer_name_k + ".weight"]
    v = hf_model.state_dict()[layer_name_v + ".weight"]

    weight = torch.cat([q, k, v], dim=0)

    scales[layer_name_qkv]["x"] = scales[layer_name_q]["x"]
    scales[layer_name_qkv]["w"] = weight.abs().max(dim=1)[0]

    scales[layer_name_qkv]["y"] = torch.cat([
        scales[layer_name_q]["y"], scales[layer_name_k]["y"],
        scales[layer_name_v]["y"]
    ],
                                            dim=0)

    gptj_qkv_para[layer_name_qkv] = weight.transpose(0, 1)


def gptj_to_trt_llm_name(orig_name):
    global_weights = {
        "transformer.wte.weight": "model.wte",
        "transformer.ln_f.bias": "model.final_layernorm.bias",
        "transformer.ln_f.weight": "model.final_layernorm.weight",
        "lm_head.weight": "model.lm_head.weight",
        "lm_head.bias": "model.lm_head.bias"
    }

    if orig_name in global_weights:
        return global_weights[orig_name]

    _, _, layer_id, *weight_name = orig_name.split(".")
    layer_id = int(layer_id)
    weight_name = "transformer." + ".".join(weight_name)

    per_layer_weights = {
        "transformer.ln_1.bias": "input_layernorm.bias",
        "transformer.ln_1.weight": "input_layernorm.weight",
        "transformer.attn.q_proj.weight": "attention.query.weight",
        "transformer.attn.q_proj.bias": "attention.query.bias",
        "transformer.attn.k_proj.weight": "attention.key.weight",
        "transformer.attn.k_proj.bias": "attention.key.bias",
        "transformer.attn.v_proj.weight": "attention.value.weight",
        "transformer.attn.v_proj.bias": "attention.value.bias",
        "transformer.attn.out_proj.bias": "attention.dense.bias",
        "transformer.attn.out_proj.weight": "attention.dense.weight",
        "transformer.mlp.fc_in.bias": "mlp.dense_h_to_4h.bias",
        "transformer.mlp.fc_in.weight": "mlp.dense_h_to_4h.weight",
        "transformer.mlp.fc_out.bias": "mlp.dense_4h_to_h.bias",
        "transformer.mlp.fc_out.weight": "mlp.dense_4h_to_h.weight",
    }
    return f"layers.{layer_id}.{per_layer_weights[weight_name]}"


# GPT-J uses nn.Linear for these following ops whose weight matrix is transposed compared to gpt2.
# In order to use the preprocess codes of gpt2, we transpose them firstly.
def transpose_weights(hf_name, param):
    weight_to_transpose = ["out_proj", "fc_in", "fc_out"]
    if any([k in hf_name for k in weight_to_transpose]):
        if len(param.shape) == 2:
            param = param.transpose(0, 1)
    return param


@torch.no_grad()
def hf_gptj_converter(args: ProgArgs):
    infer_tp = args.tensor_parallelism
    multi_query_mode = False
    saved_dir = Path(args.out_dir) / f"{infer_tp}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    # load position_embedding from rank 0
    model = AutoModelForCausalLM.from_pretrained(args.in_file,
                                                 torch_dtype="auto",
                                                 device_map="auto",
                                                 trust_remote_code=True)
    if args.load_model_on_cpu:
        model = model.cpu()
        torch.cuda.empty_cache()
    act_range = {}
    gptj_qkv_para = {}

    if args.calibrate_kv_cache:
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false")
        from datasets import load_dataset
        dataset = load_dataset("lambada",
                               split="validation",
                               cache_dir=args.dataset_cache_dir)
        act_range = capture_activation_range(
            model, AutoTokenizer.from_pretrained(args.in_file), dataset)

    config = configparser.ConfigParser()
    config["gpt"] = {}
    for key in vars(args):
        config["gpt"][key] = f"{vars(args)[key]}"
    for k, v in vars(model.config).items():
        config["gpt"][k] = f"{v}"
    config["gpt"]["storage_dtype"] = args.storage_type
    config["gpt"]["multi_query_mode"] = str(multi_query_mode)
    with open(saved_dir / "config.ini", 'w') as configfile:
        config.write(configfile)

    storage_type = str_dtype_to_torch(args.storage_type)

    global_ft_weights = [
        "model.wte", "model.final_layernorm.bias",
        "model.final_layernorm.weight", "model.lm_head.weight",
        "model.lm_head.bias"
    ]

    int8_outputs = None
    if args.calibrate_kv_cache:
        int8_outputs = "kv_cache_only"

    starmap_args = []
    for name, param in model.named_parameters():
        if "weight" not in name and "bias" not in name:
            continue
        trt_llm_name = gptj_to_trt_llm_name(name)

        param = transpose_weights(name, param)

        if args.convert_model_on_cpu:
            param = param.cpu()
        if trt_llm_name in global_ft_weights:
            torch_to_numpy(param.to(storage_type).cpu()).tofile(
                saved_dir / f"{trt_llm_name}.bin")
        elif 'q_proj' in name:
            trt_llm_name = trt_llm_name.replace("query", "query_key_value")
            # Needed by QKV projection weight split. With multi_query_mode one does not simply take
            # out_dim and divide it by 3 to get local_dim because out_dim = local_dim + 2 * head_size
            local_dim = model.transformer.h[
                0].attn.embed_dim if multi_query_mode else None
            merge_qkv_scales(name, model, act_range, gptj_qkv_para)
            qkv = (0, saved_dir, infer_tp, trt_llm_name,
                   gptj_qkv_para.get(
                       name.replace(".weight",
                                    "").replace(".q_proj",
                                                ".qkv_proj")).to(storage_type),
                   storage_type,
                   act_range.get(
                       name.replace(".weight",
                                    "").replace(".q_proj", ".qkv_proj")), {
                                        "int8_outputs": int8_outputs,
                                        "multi_query_mode": multi_query_mode,
                                        "local_dim": local_dim
                                    })
            starmap_args.append(qkv)
        elif 'k_proj' in name or 'v_proj' in name:
            continue
        else:
            starmap_args.append(
                (0, saved_dir, infer_tp, trt_llm_name, param.to(storage_type),
                 storage_type, act_range.get(name.replace(".weight", "")), {
                     "int8_outputs": int8_outputs,
                     "multi_query_mode": multi_query_mode,
                     "local_dim": None
                 }))

    starmap_args = tqdm(starmap_args, desc="saving weights")
    if args.processes > 1:
        with multiprocessing.Pool(args.processes) as pool:
            pool.starmap(split_and_save_weight, starmap_args)
    else:
        # simpler for debug situations
        for starmap_arg in starmap_args:
            split_and_save_weight(*starmap_arg)


def run_conversion(args: ProgArgs):
    if args.processes > 1 and platform.system() == "Windows":
        print(
            "Resetting processes to 1 because multi-process on Windows is not implemented."
        )
        args = dataclasses.replace(args, processes=1)

    print("\n=============== Arguments ===============")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("========================================")

    assert (args.calibrate_kv_cache), \
        "INT8 kv cache must be enabled for this script. Otherwise you can directly build engines from HuggingFace checkpoints, no need to do this FT-format conversion. "
    hf_gptj_converter(args)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    run_conversion(ProgArgs.parse())
