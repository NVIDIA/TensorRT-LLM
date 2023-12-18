# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2022, MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert MPT model checkpoint to FT format.

It's a modified version of
https://github.com/NVIDIA/FasterTransformer/blob/main/examples/pytorch/gpt/utils/huggingface_gpt_convert.py
"""

import argparse
import configparser
import dataclasses
import multiprocessing
import os
import platform
from pathlib import Path

import numpy as np
import torch
import transformers
from convert import split_and_save_weight, write_zero_bias
from datasets import load_dataset
from smoothquant import capture_activation_range, smooth_gemm
from tqdm import tqdm

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy

# numpy doesn't know bfloat16, define abstract binary type instead
np_bfloat16 = np.dtype('V2', metadata={"dtype": "bfloat16"})


@dataclasses.dataclass(frozen=True)
class ProgArgs:
    out_dir: str
    in_file: str
    tensor_parallelism: int = 1
    processes: int = 4
    calibrate_kv_cache: bool = False
    smoothquant: float = None
    model: str = "mpt"
    storage_type: str = "float16"
    dataset_cache_dir: str = None
    load_model_on_cpu: bool = False
    convert_model_on_cpu: bool = False
    force: bool = False

    @staticmethod
    def parse(args=None) -> 'ProgArgs':
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--out_dir',
                            '-o',
                            type=str,
                            help='file name of output directory',
                            required=True)
        parser.add_argument('--in_file',
                            '-i',
                            type=str,
                            help='file name of input checkpoint file',
                            required=True)
        parser.add_argument('--tensor_parallelism',
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
            "--calibrate_kv_cache",
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
        parser.add_argument(
            "--model",
            default="mpt",
            type=str,
            help="Specify MPT variants to convert checkpoints correctly",
            choices=["mpt"])
        parser.add_argument("--storage_type",
                            "-t",
                            type=str,
                            default="float16",
                            choices=["float32", "float16", "bfloat16"])
        parser.add_argument("--dataset_cache_dir",
                            type=str,
                            default=None,
                            help="cache dir to load the hugging face dataset")
        parser.add_argument("--load_model_on_cpu", action="store_true")
        parser.add_argument("--convert_model_on_cpu", action="store_true")
        parser.add_argument(
            '--force',
            action='store_true',
            help=
            'Force conversion to FT even if some features may not work as expected in FT'
        )

        return ProgArgs(**vars(parser.parse_args(args)))


def transpose_weights(hf_name, param):
    weight_to_transpose = [
        "attention.query_key_value", "attention.dense", "mlp.dense_h_to_4h",
        "mlp.dense_4h_to_h"
    ]
    if any([k in hf_name for k in weight_to_transpose]):
        if len(param.shape) == 2:
            param = param.transpose(0, 1)
    return param


def mpt_to_trt_llm_name(orig_name):
    global_weights = {
        "transformer.wpe.weight": "model.wpe",
        "transformer.wte.weight": "model.wte",
        "transformer.norm_f.bias": "model.final_layernorm.bias",
        "transformer.norm_f.weight": "model.final_layernorm.weight",
        "transformer.lm_head.weight": "model.lm_head.weight",
    }

    if orig_name in global_weights:
        return global_weights[orig_name]

    _, _, layer_id, *weight_name = orig_name.split(".")
    layer_id = int(layer_id)
    weight_name = "transformer." + ".".join(weight_name)

    per_layer_weights = {
        'transformer.norm_1.bias': 'input_layernorm.bias',
        'transformer.norm_1.weight': 'input_layernorm.weight',
        'transformer.attn.Wqkv.bias': 'attention.query_key_value.bias',
        'transformer.attn.Wqkv.weight': 'attention.query_key_value.weight',
        'transformer.attn.out_proj.bias': 'attention.dense.bias',
        'transformer.attn.out_proj.weight': 'attention.dense.weight',
        'transformer.norm_2.bias': 'post_attention_layernorm.bias',
        'transformer.norm_2.weight': 'post_attention_layernorm.weight',
        'transformer.ffn.up_proj.bias': 'mlp.dense_h_to_4h.bias',
        'transformer.ffn.up_proj.weight': 'mlp.dense_h_to_4h.weight',
        'transformer.ffn.down_proj.bias': 'mlp.dense_4h_to_h.bias',
        'transformer.ffn.down_proj.weight': 'mlp.dense_4h_to_h.weight',
    }
    return f"layers.{layer_id}.{per_layer_weights[weight_name]}"


@torch.no_grad()
def smooth_mpt_model(model, scales, alpha, mpt_qkv_para, mpt_smoother):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        if not isinstance(module, type(model.transformer.blocks[0])):
            continue
        # qkv_proj
        layer_name_qkv = name + ".attn.Wqkv"
        weight = module.attn.Wqkv.weight
        smoother = smooth_gemm(weight, scales[layer_name_qkv]["x"],
                               module.norm_1.weight, module.norm_1.bias, alpha)
        scales[layer_name_qkv]["x"] = scales[layer_name_qkv]["x"] / smoother
        scales[layer_name_qkv]["w"] = weight.abs().max(dim=1)[0]
        # see transpose_weights function
        mpt_qkv_para[layer_name_qkv] = weight.transpose(0, 1)

        # =================================================================
        layer_name = name + ".attn.out_proj"
        smoother = smooth_gemm(module.attn.out_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        mpt_smoother[layer_name] = smoother.float()

        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.attn.out_proj.weight.abs().max(
            dim=1)[0]

        # ==================================================================
        fc1_layer_name = name + ".ffn.up_proj"

        smoother = smooth_gemm(module.ffn.up_proj.weight,
                               scales[fc1_layer_name]["x"],
                               module.norm_2.weight, module.norm_2.bias, alpha)

        scales[fc1_layer_name]["x"] = scales[fc1_layer_name]["x"] / smoother
        scales[fc1_layer_name]["w"] = module.ffn.up_proj.weight.abs().max(
            dim=1)[0]

        # ==================================================================
        layer_name = name + ".ffn.down_proj"
        smoother = smooth_gemm(module.ffn.down_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        mpt_smoother[layer_name] = smoother.float()
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.ffn.down_proj.weight.abs().max(
            dim=1)[0]


def hf_mpt_converter(args: ProgArgs) -> None:
    """Convert an MPT checkpoint to a FasterTransformer compatible format.
    """
    infer_tp = args.tensor_parallelism
    saved_dir = Path(args.out_dir) / f"{infer_tp}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    torch_data_type = str_dtype_to_torch(args.storage_type)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.in_file,
        trust_remote_code=True,
        torch_dtype=torch_data_type,
        device_map="auto")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.in_file, trust_remote_code=True)

    if args.load_model_on_cpu:
        model = model.cpu()
        torch.cuda.empty_cache()

    act_range = {}
    mpt_qkv_para = {}
    # smoother for inputs of attn.out_proj and ffn.down_proj
    mpt_smoother = {}
    if args.smoothquant is not None or args.calibrate_kv_cache:
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false")
        dataset = load_dataset("ccdv/cnn_dailymail",
                               '3.0.0',
                               cache_dir=args.dataset_cache_dir)
        act_range = capture_activation_range(model, tokenizer, dataset)
        if args.smoothquant is not None:
            smooth_mpt_model(model, act_range, args.smoothquant, mpt_qkv_para,
                             mpt_smoother)

    hf_config = vars(model.config)
    config = configparser.ConfigParser()
    config['gpt'] = {}
    try:
        config['gpt']['model_name'] = 'mpt' if hf_config[
            '_name_or_path'] == '' else hf_config['_name_or_path']
        config['gpt']['n_head'] = str(hf_config['n_heads'])
        n_embd = hf_config['d_model']
        config['gpt']['n_embd'] = str(n_embd)
        # config['gpt']['size_per_head'] = str(n_embd // hf_config['n_heads'])
        config['gpt']['n_inner'] = str(n_embd * hf_config['expansion_ratio'])
        config['gpt']['n_positions'] = str(hf_config['max_seq_len'])
        config['gpt']['n_layer'] = str(hf_config['n_layers'])
        config['gpt']['vocab_size'] = str(hf_config['vocab_size'])
        config['gpt']['bos_token_id'] = str(
            hf_config['bos_token_id']
        ) if hf_config['bos_token_id'] != None else str(tokenizer.bos_token_id)
        config['gpt']['eos_token_id'] = str(
            hf_config['eos_token_id']
        ) if hf_config['eos_token_id'] != None else str(tokenizer.eos_token_id)
        config['gpt'][
            'storage_dtype'] = args.storage_type  # == 'fp32' else 'float16'
        config['gpt']['tensor_parallelism'] = str(args.tensor_parallelism)
        config['gpt']['activation_function'] = str("gelu")
        config['gpt']['calibrate_kv_cache'] = 'False'
        config['gpt']['smoothquant'] = str(args.smoothquant)
        # nn.LayerNorm default eps is 1e-5
        config['gpt']['layer_norm_epsilon'] = str(1e-5)
        if hf_config['attn_config']['alibi']:
            config['gpt']['position_embedding_type'] = str("alibi")
            # config['gpt']['has_positional_encoding'] = str(False)
            # config['gpt']['use_attention_linear_bias'] = str(True)
        if hf_config['attn_config']['clip_qkv'] and not args.force:
            raise RuntimeError(
                'clip_qkv is enabled for this MPT model. This may not work as expected in FT. Use --force to force a conversion.'
            )
        if hf_config['attn_config']['qk_ln'] and not args.force:
            raise RuntimeError(
                'qk_ln is enabled for this MPT model. This may not work as expected in FT. Use --force to force a conversion.'
            )
        if 'kv_n_heads' in hf_config['attn_config']:
            config['gpt']['n_kv_head'] = str(
                hf_config['attn_config']['kv_n_heads'])

        if 'no_bias' in hf_config and hf_config['no_bias']:
            config['gpt']['bias'] = str(False)

        with open(os.path.join(saved_dir, 'config.ini'), 'w') as configfile:
            config.write(configfile)
    except:
        print(f'Failed to save the config in config.ini.')
        raise

    multi_query_mode = True if 'kv_n_heads' in hf_config[
        'attn_config'] else False

    global_trt_llm_weights = [
        "model.wpe",
        "model.wte",
        "model.final_layernorm.weight",
        "model.final_layernorm.bias",
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
        trt_llm_name = mpt_to_trt_llm_name(name)

        if args.convert_model_on_cpu:
            param = param.cpu()

        if name.replace(".weight", "") in mpt_smoother.keys():
            smoother = mpt_smoother[name.replace(".weight", "")]
            starmap_args.append(
                (0, saved_dir, infer_tp,
                 f"{trt_llm_name}.smoother".replace(".weight", ""),
                 smoother.to(torch.float32), torch.float32, None, {
                     "int8_outputs": int8_outputs
                 }))

        param = transpose_weights(trt_llm_name, param)

        if trt_llm_name in global_trt_llm_weights:
            torch_to_numpy(param.to(torch_data_type).cpu()).tofile(
                saved_dir / f"{trt_llm_name}.bin")
            if 'final_layernorm.weight' in trt_llm_name:
                write_zero_bias('final_layernorm.weight', saved_dir,
                                param.shape[-1], torch_data_type)
        else:
            local_dim = model.transformer.blocks[0].attn.d_model \
                if multi_query_mode else None
            starmap_args.append(
                (0, saved_dir, infer_tp, trt_llm_name,
                 param.to(torch_data_type).detach().cpu(), torch_data_type,
                 act_range.get(name.replace(".weight", "")), {
                     "int8_outputs": int8_outputs,
                     "no_bias": hf_config["no_bias"],
                     "multi_query_mode": multi_query_mode,
                     "local_dim": local_dim
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
    hf_mpt_converter(args)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    run_conversion(ProgArgs.parse())
