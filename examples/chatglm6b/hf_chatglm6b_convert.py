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
'''
Convert huggingface ChatGLM-6b model. Use https://huggingface.co/THUDM/chatglm-6b as demo.
'''
import argparse
import configparser
import os
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as multiprocessing
from convert import split_and_save_weight, str_to_np_dtype
from smoothquant import capture_activation_range, smooth_gemm
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block


@torch.no_grad()
def smooth_gpt_model(model, scales, alpha):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        if not isinstance(module, GPT2Block):
            continue

        # qkv_proj
        layer_name = name + ".attn.c_attn"
        smoother = smooth_gemm(module.attn.c_attn.weight.T,
                               scales[layer_name]["x"], module.ln_1.weight,
                               module.ln_1.bias, alpha)
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.attn.c_attn.weight.abs().max(dim=0)[0]

        # fc1
        layer_name = name + ".mlp.c_fc"
        smoother = smooth_gemm(module.mlp.c_fc.weight.T,
                               scales[layer_name]["x"], module.ln_2.weight,
                               module.ln_2.bias, alpha)
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.c_fc.weight.abs().max(dim=0)[0]


def gpt_to_ft_name(orig_name):
    global_weights = { \
                        "transformer.final_layernorm.bias": "model.final_layernorm.bias", \
                        "transformer.final_layernorm.weight": "model.final_layernorm.weight", \
                        }

    if orig_name in global_weights:
        return global_weights[orig_name]

    return ".".join(orig_name.split(".")[1:])


@torch.no_grad()
def hf_chatglm6b_converter(args):
    infer_tp = args.tensor_parallelism
    saved_dir = Path(args.out_dir) / f"{infer_tp}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    # load position_embedding from rank 0
    model = AutoModel.from_pretrained(args.in_file, trust_remote_code=True)

    act_range = {}
    if args.smoothquant is not None or args.calibrate_kv_cache:
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false")
        act_range = capture_activation_range(
            model, AutoTokenizer.from_pretrained(args.in_file))
        if args.smoothquant is not None:
            smooth_gpt_model(model, act_range, args.smoothquant)

    config = configparser.ConfigParser()
    config["chatglm6b"] = {}
    for key in vars(args):
        config["chatglm6b"][key] = f"{vars(args)[key]}"
    for k, v in vars(model.config).items():
        config["chatglm6b"][k] = f"{v}"
    config["chatglm6b"]["weight_data_type"] = args.storage_type
    with open(saved_dir / "config.ini", 'w') as configfile:
        config.write(configfile)

    storage_type = str_to_np_dtype(args.storage_type)

    if args.calibrate_kv_cache:
        pass
    if args.smoothquant is not None:
        pass
    '''
    # list all named parameters
    for name, param in model.named_parameters():
        print(name,param.shape)
    '''
    # add weight of LM
    data = np.load("lm.npy")
    data.astype(storage_type).tofile(saved_dir / "model.lm.weight.bin")
    print("Save model.lm.weight.bin")
    # add weight of position embedding
    nMaxSL = 2048
    inv_freq = 10**(-1 / 16 * np.arange(0, 64, 2, dtype=np.float32))
    valueTable = np.matmul(
        np.arange(nMaxSL, dtype=np.float32).reshape(-1, 1),
        np.concatenate([inv_freq, inv_freq],
                       axis=0).reshape(1, -1)).reshape(nMaxSL,
                                                       len(inv_freq) * 2)
    np.cos(valueTable).astype(storage_type).tofile(saved_dir /
                                                   "model.cosTable.weight.bin")
    np.sin(valueTable).astype(storage_type).tofile(saved_dir /
                                                   "model.sinTable.weight.bin")
    print("Save model.cosTable.weight.bin")
    print("Save model.sinTable.weight.bin")

    starmap_args = []
    for name, param in model.named_parameters():
        if "weight" not in name and "bias" not in name:
            print("Skip %s" % name)
            continue
        elif name in [
                "transformer.word_embeddings.weight",
                "transformer.final_layernorm.weight",
                "transformer.final_layernorm.bias"
        ]:
            param.detach().cpu().numpy().astype(storage_type).tofile(
                saved_dir / (name.replace("transformer", "model") + ".bin"))
            print("Save %s" % name)
            continue

        ft_name = gpt_to_ft_name(name)

        param = param.detach().cpu().numpy().astype(storage_type)
        starmap_args.append((0, saved_dir, infer_tp, ft_name, args, param,
                             act_range.get(name.replace(".weight", ""))))

    starmap_args = tqdm(starmap_args, desc="saving weights")
    if args.processes > 1:
        with multiprocessing.Pool(args.processes) as pool:
            pool.starmap(split_and_save_weight, starmap_args)
    else:
        # simpler for debug situations
        for starmap_arg in starmap_args:
            split_and_save_weight(*starmap_arg)
            print("Save %s" % starmap_arg[3])


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
        help="How many processes to spawn for conversion (default: 4)",
        default=4)
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

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))
    print("========================================")

    hf_chatglm6b_converter(args)
