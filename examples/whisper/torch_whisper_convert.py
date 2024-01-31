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
from convert import split_and_save_weight, str_to_np_dtype
from smoothquant import (capture_activation_range, smooth_gemm,
                         smooth_gemm_fc1_gate)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import whisper

def merge_qkv_scales(q_name, hf_model, scales, whisper_qkv_para):
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
    print(scales[layer_name_q])
    scales[layer_name_qkv]["y"] = torch.cat([
        scales[layer_name_q]["y"], scales[layer_name_k]["y"],
        scales[layer_name_v]["y"]
    ],
                                            dim=0)

    whisper_qkv_para[layer_name_qkv] = weight.transpose(0, 1)


def merge_qkv_bias(q_name, hf_model, whisper_qkv_para={}):
    layer_name_q = q_name.replace(".bias", "")
    layer_name_k = layer_name_q.replace("q_proj", "k_proj")
    layer_name_v = layer_name_q.replace("q_proj", "v_proj")
    # layer_name_qkv = layer_name_q.replace("q_proj", "qkv_proj")

    q = hf_model.state_dict()[layer_name_q + ".bias"]
    k = hf_model.state_dict()[layer_name_k + ".bias"]
    v = hf_model.state_dict()[layer_name_v + ".bias"]

    bias = torch.cat([q, k, v], dim=0)

    return bias


@torch.no_grad()
def smooth_whisper_model(model, scales, alpha, whisper_qkv_para,
                          whisper_smoother):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        if not module.__class__.__name__ == "whisperDecoderLayer":
            continue
        # qkv_proj
        layer_name_q = name + ".self_attn.q_proj"
        layer_name_k = name + ".self_attn.k_proj"
        layer_name_v = name + ".self_attn.v_proj"
        layer_name_qkv = name + ".self_attn.qkv_proj"

        weight = torch.cat([
            module.self_attn.q_proj.weight, module.self_attn.k_proj.weight,
            module.self_attn.v_proj.weight
        ],
                           dim=0)

        smoother = smooth_gemm(weight, scales[layer_name_q]["x"],
                               module.input_layernorm.weight, None, alpha)

        scales[layer_name_qkv]["x"] = scales[layer_name_q]["x"] / smoother
        scales[layer_name_qkv]["w"] = weight.abs().max(dim=1)[0]
        scales[layer_name_qkv]["y"] = torch.cat([
            scales[layer_name_q]["y"], scales[layer_name_k]["y"],
            scales[layer_name_v]["y"]
        ],
                                                dim=0)

        # see transpose_weights function
        whisper_qkv_para[layer_name_qkv] = weight.transpose(0, 1)

        # =================================================================
        layer_name = name + ".self_attn.o_proj"
        smoother = smooth_gemm(module.self_attn.o_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        whisper_smoother[layer_name] = smoother.float()

        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.self_attn.o_proj.weight.abs().max(
            dim=1)[0]

        # ==================================================================
        fc1_layer_name = name + ".mlp.gate_proj"
        gate_layer_name = name + ".mlp.up_proj"

        smoother = smooth_gemm_fc1_gate(module.mlp.gate_proj.weight,
                                        module.mlp.up_proj.weight,
                                        scales[fc1_layer_name]["x"],
                                        module.post_attention_layernorm.weight,
                                        None, alpha)

        scales[fc1_layer_name]["x"] = scales[fc1_layer_name]["x"] / smoother
        scales[fc1_layer_name]["w"] = module.mlp.gate_proj.weight.abs().max(
            dim=1)[0]

        scales[gate_layer_name]["x"] = scales[gate_layer_name]["x"] / smoother
        scales[gate_layer_name]["w"] = module.mlp.up_proj.weight.abs().max(
            dim=1)[0]

        # ==================================================================
        layer_name = name + ".mlp.down_proj"
        smoother = smooth_gemm(module.mlp.down_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        whisper_smoother[layer_name] = smoother.float()
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.down_proj.weight.abs().max(
            dim=1)[0]


# LLaMA uses nn.Linear for these following ops whose weight matrix is transposed compared to gpt2.
# In order to use the preprocess codes of gpt2, we transpose them firstly.
def transpose_weights(hf_name, param):
    weight_to_transpose = ["o_proj", "gate_proj", "down_proj", "up_proj"]
    if any([k in hf_name for k in weight_to_transpose]):
        if len(param.shape) == 2:
            param = param.transpose(0, 1)
    return param


def concat_qkv_weight_bias(q, hf_key, hf_model):
    bias_shape = q.shape
    bias_dtype = q.dtype
    if hf_key.replace("attn.query", "attn.key") not in hf_model.state_dict().keys() and 'key.bias' in hf_key.replace("attn.query", "attn.key"):
        k = torch.zeros([*bias_shape], dtype=bias_dtype).to('cuda')
    else:
        k = hf_model.state_dict()[hf_key.replace("attn.query", "attn.key")]
    v = hf_model.state_dict()[hf_key.replace("attn.query", "attn.value")]
    return torch.cat([q, k, v], dim=0)

def torch_whisper_converter(args):
    infer_tp = args.tensor_parallelism
    saved_dir = Path(args.out_dir) / f"{infer_tp}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    model = whisper.load_model(args.in_file, download_root="assets")
    model = model.to("cuda")

    act_range = {}
    whisper_qkv_para = {}
    # smoother for inputs of self_attn.o_proj and mlp.down_proj
    whisper_smoother = {}

    if args.smoothquant is not None or args.calibrate_kv_cache:
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false")
        act_range = capture_activation_range(
            model, "hf-internal-testing/librispeech_asr_dummy")
        if args.smoothquant is not None:
            smooth_whisper_model(model, act_range, args.smoothquant,
                                  whisper_qkv_para, whisper_smoother)

    pop_list = []
    add_list = []
    for key in act_range.keys():
        if 'attn.query' in key or 'cross_attn.query' in key:
            ft_key = key.replace("query", "query_key_value")
            q_key = key
            k_key = key.replace("query", "key")
            v_key = key.replace("query", "value")
            q_act_range = act_range.get(q_key)
            k_act_range = act_range.get(k_key)
            v_act_range = act_range.get(v_key)
            qkv_act_range = {'x': torch.cat((q_act_range['x'], k_act_range['x'], v_act_range['x']), dim=0),
                             'y': torch.cat((q_act_range['y'], k_act_range['y'], v_act_range['y']), dim=0),
                             'w': torch.cat((q_act_range['w'], k_act_range['w'], v_act_range['w']), dim=0)
                             }
            add_list.append({ft_key : qkv_act_range})
            pop_list.append(q_key)
            pop_list.append(k_key)
            pop_list.append(v_key)
    for add_item in add_list:
        act_range.update(add_item)
    for pop_name in pop_list:
        act_range.pop(pop_name)

    config = configparser.ConfigParser()
    config["whisper"] = {}
    for key in vars(args):
        config["whisper"][key] = f"{vars(args)[key]}"
    # for k, v in vars(model.config).items():
    #     config["whisper"][k] = f"{v}"
    config["whisper"]["weight_data_type"] = args.storage_type
    config["whisper"]["multi_query_mode"] = str(args.multi_query_mode)
    with open(saved_dir / "config.ini", 'w') as configfile:
        config.write(configfile)

    storage_type = str_to_np_dtype(args.storage_type)

    # global_ft_weights = [
    #     'decoder.token_embedding.weight', 'decoder.token_embedding.weight', 'lm_head.weight'
    # ]

    int8_outputs = None
    if args.calibrate_kv_cache:
        int8_outputs = "kv_cache_only"
    if args.smoothquant is not None:
        int8_outputs = "all"

    starmap_args = []
    for name, param in model.named_parameters():
        if "weight" not in name and "bias" not in name:
            continue
        ft_name = name

        if name.replace(".weight", "") in whisper_smoother.keys():
            smoother = whisper_smoother[name.replace(".weight", "")]
            smoother = smoother.detach().cpu().numpy()
            starmap_args.append(
                (0, saved_dir, infer_tp,
                 f"{ft_name}.smoother".replace(".weight", ""), smoother, None, {
                     "int8_outputs": int8_outputs,
                     "multi_query_mode": args.multi_query_mode,
                     "local_dim": None,
                 }))

        # param = transpose_weights(name, param)

        if "weight" not in name and "bias" not in name:
            continue
        ft_name = name

        if 'attn.query' in name or 'cross_attn.query' in name:
            param = concat_qkv_weight_bias(param, name, model)
            ft_name = ft_name.replace("query", "query_key_value")
            
        param = param.detach().cpu().numpy().astype(storage_type)
        
        starmap_args.append(
                (0, saved_dir, infer_tp, ft_name, param,
                 act_range.get(ft_name.replace(".weight", "")), {
                     "int8_outputs": int8_outputs,
                     "multi_query_mode": args.multi_query_mode,
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


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

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
        help="How many processes to spawn for conversion (default: 8)",
        default=8)
    parser.add_argument(
        "--calibrate-kv-cache",
        "-kv",
        action="store_true",
        help=
        "Generate scaling factors for KV cache. Used for storing KV cache in int8."
    )
    parser.add_argument(
        "--smoothquant",
        "-sq",
        type=float,
        default=None,
        help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
        " to Smoothquant the model, and output int8 weights."
        " A good first try is 0.5. Must be in [0, 1]")
    parser.add_argument("--storage-type",
                        "-t",
                        type=str,
                        default="fp32",
                        choices=["fp32", "fp16"])
    parser.add_argument("--multi-query-mode",
                        action="store_true",
                        help="Use multi-query-attention.")

    args = parser.parse_args()
    if args.processes > 1 and platform.system() == "Windows":
        print(
            "Resetting processes to 1 because multi-process on Windows is not implemented."
        )
        args.processes = 1

    print("\n=============== Argument ===============")
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))
    print("========================================")

    assert (args.calibrate_kv_cache or args.smoothquant), \
        "Either INT8 kv cache or SmoothQuant must be enabled for this script. Otherwise you can directly build engines from HuggingFace checkpoints, no need to do this FT-format conversion. "

    torch_whisper_converter(args)
