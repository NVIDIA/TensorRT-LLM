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
Convert huggingface Meta OPT model. Use https://huggingface.co/facebook/opt-125m as demo.
'''

import argparse
import configparser
import multiprocessing
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM  # transformers-4.20.0.dev0

from tensorrt_llm.logger import logger


def save_val(val, dir, key, tp_num=None):
    path = str(dir / ("model." + key))
    if tp_num is not None:
        path += "." + str(tp_num)
    path += ".bin"

    val.tofile(path)


def save_split(split_vals, dir, key, i, factor):
    for j, val in enumerate(split_vals):
        save_val(val, dir, key, i * factor + j)


def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"


def quantize(mat, act_range):
    # qkv proj weight quantization
    if mat.ndim == 3 and mat.shape[1] == 3:
        # get max_q, max_k, max_v
        mat_max = np.abs(mat).clip(1e-8, None).max(axis=(0, 2))[None, :, None]
    else:
        mat_max = np.abs(mat).clip(1e-8, None).max()

    act_scale_in = 127. / np.array(act_range["input"])
    weight_scales = 127. / mat_max
    act_scale_post = 127. / np.array(act_range["output"])

    mat_quant = (mat * weight_scales).round().astype(np.int8)
    return mat_quant, weight_scales, act_scale_in, act_scale_post


def split_and_convert_process(i, saved_dir, factor, key, args, val, old_name,
                              dtype):
    logger.debug(f"split_and_convert_process {key}")
    old_name.rpartition(".")[0]

    if "input_layernorm.weight" in key or "input_layernorm.bias" in key or \
        "attention.dense.bias" in key or "post_attention_layernorm.weight" in key or \
        "post_attention_layernorm.bias" in key or "mlp.dense_4h_to_h.bias" in key or \
        "final_layernorm.weight" in key or "final_layernorm.bias" in key:

        # shared weights, only need to convert the weights of rank 0
        if i == 0:
            save_val(val, saved_dir, key)

    elif "attention.dense.weight" in key or "mlp.dense_4h_to_h.weight" in key:
        save_split(np.split(val, factor, axis=0), saved_dir, key, i, factor)

    elif "mlp.dense_h_to_4h.weight" in key or "mlp.dense_h_to_4h.bias" in key:
        save_split(np.split(val, factor, axis=-1), saved_dir, key, i, factor)

    elif "attention.query_key_value.bias" in key:
        local_dim = val.shape[-1] // 3

        val = val.reshape(3, local_dim)
        save_split(np.split(val, factor, axis=-1), saved_dir, key, i, factor)

    elif "attention.query_key_value.weight" in key:
        hidden_dim = val.shape[0]
        local_dim = val.shape[-1] // 3

        val = val.reshape(hidden_dim, 3, local_dim)
        save_split(np.split(val, factor, axis=-1), saved_dir, key, i, factor)

    else:
        logger.error("[ERROR] Key '{}' not handled".format(key))


@torch.no_grad()
def split_and_convert(args):
    saved_dir = Path(args.saved_dir) / f"{args.infer_gpu_num}-gpu"

    if not saved_dir.exists():
        saved_dir.mkdir(parents=True)

    t_gpu_num = args.trained_gpu_num
    i_gpu_num = args.infer_gpu_num
    assert (i_gpu_num % t_gpu_num == 0)

    factor = (int)(i_gpu_num / t_gpu_num)

    # load position_embedding from rank 0
    model = AutoModelForCausalLM.from_pretrained(args.in_file,
                                                 device_map="auto")
    hf_config = vars(model.config)

    num_layers = hf_config["num_hidden_layers"]

    # NOTE: save parameters to config files (loaded by triton backends)
    config = configparser.ConfigParser()
    config["gpt"] = {}
    try:
        config["gpt"]["model_name"] = "opt" if hf_config[
            "_name_or_path"] == '' else hf_config["_name_or_path"]
        config["gpt"]["n_head"] = str(hf_config["num_attention_heads"])
        n_embd = hf_config["hidden_size"]
        config["gpt"]["size_per_head"] = str(n_embd //
                                             hf_config["num_attention_heads"])
        config["gpt"]["inter_size"] = str(hf_config["ffn_dim"])
        config['gpt']['n_positions'] = str(hf_config['max_position_embeddings'])
        config['gpt']['n_embd'] = str(hf_config['hidden_size'])
        config["gpt"]["n_layer"] = str(hf_config["num_hidden_layers"])
        config["gpt"]["layernorm_eps"] = "1e-5"
        config["gpt"]["norm_type"] = "layernorm"
        config["gpt"]["norm_position_type"] = "pre" if hf_config[
            "do_layer_norm_before"] else "post"
        config["gpt"]["activation_type"] = "Relu"
        config["gpt"]["vocab_size"] = str(hf_config["vocab_size"])
        config["gpt"]["start_id"] = str(hf_config["bos_token_id"])
        config["gpt"]["end_id"] = str(hf_config["eos_token_id"])
        config['gpt']['weight_data_type'] = args.weight_data_type

        for key in vars(args):
            config["gpt"][key] = f"{vars(args)[key]}"
        for k, v in vars(model.config).items():
            config["gpt"][k] = f"{v}"
        with open(str(saved_dir / "config.ini"), 'w') as configfile:
            config.write(configfile)
    except Exception as e:
        logger.error(f"Fail to save the config in config.ini due to error {e}.")

    np_weight_data_type = get_weight_data_type(args.weight_data_type)

    huggingface_model_name_pattern = [
        "self_attn_layer_norm.bias",
        "self_attn_layer_norm.weight",
        "self_attn.qkv_proj.bias",
        "self_attn.qkv_proj.weight",
        "self_attn.out_proj.bias",
        "self_attn.out_proj.weight",
        "final_layer_norm.bias",
        "final_layer_norm.weight",
        "fc1.bias",
        "fc1.weight",
        "fc2.bias",
        "fc2.weight",
    ]

    ft_model_name_pattern = [
        "input_layernorm.bias",
        "input_layernorm.weight",
        "attention.query_key_value.bias",
        "attention.query_key_value.weight",
        "attention.dense.bias",
        "attention.dense.weight",
        "post_attention_layernorm.bias",
        "post_attention_layernorm.weight",
        "mlp.dense_h_to_4h.bias",
        "mlp.dense_h_to_4h.weight",
        "mlp.dense_4h_to_h.bias",
        "mlp.dense_4h_to_h.weight",
    ]

    model_named_parameters_iter = model.named_parameters()
    model_named_parameters = dict()
    for name, param in model_named_parameters_iter:
        if "embed" in name:
            model_named_parameters[name] = param
        elif "project_in" in name:
            model_named_parameters[name] = param.permute(1, 0)
        elif "project_out" in name:
            model_named_parameters[name] = param
        else:
            model_named_parameters[name] = param.permute(1, 0) if len(
                param.shape) == 2 else param

    for l in range(num_layers):
        prefix = f'model.decoder.layers.{l}.self_attn.'
        q_weight = model_named_parameters[prefix +
                                          'q_proj.weight'].detach().cpu()
        k_weight = model_named_parameters[prefix +
                                          'k_proj.weight'].detach().cpu()
        v_weight = model_named_parameters[prefix +
                                          'v_proj.weight'].detach().cpu()
        q_bias = model_named_parameters[prefix + 'q_proj.bias'].detach().cpu()
        k_bias = model_named_parameters[prefix + 'k_proj.bias'].detach().cpu()
        v_bias = model_named_parameters[prefix + 'v_proj.bias'].detach().cpu()
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=-1)
        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=-1)
        model_named_parameters[prefix + 'qkv_proj.weight'] = qkv_weight
        model_named_parameters[prefix + 'qkv_proj.bias'] = qkv_bias

    pool = multiprocessing.Pool(args.processes)
    padding_offset = 2
    promises = []
    for name, param in model_named_parameters.items():
        if name == 'model.decoder.embed_positions.weight':
            param[padding_offset:, ...].detach().cpu().numpy().astype(
                np_weight_data_type).tofile(saved_dir / "model.wpe.bin")
        elif name == 'model.decoder.embed_tokens.weight':
            if 'model.decoder.project_in.weight' in model_named_parameters.keys(
            ):
                project_in = model_named_parameters[
                    'model.decoder.project_in.weight']
                project_out = model_named_parameters[
                    'model.decoder.project_out.weight']
                torch.matmul(param, project_in).detach().cpu().numpy().astype(
                    np_weight_data_type).tofile(saved_dir / "model.wte.bin")
                torch.matmul(param, project_out).detach().cpu().numpy().astype(
                    np_weight_data_type).tofile(saved_dir /
                                                "model.lm_head.weight.bin")
            else:
                param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
                    saved_dir / "model.wte.bin")
                param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
                    saved_dir / "model.lm_head.weight.bin")
        elif name == 'model.decoder.final_layer_norm.weight':
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
                saved_dir / "model.final_layernorm.weight.bin")
        elif name == 'model.decoder.final_layer_norm.bias':
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(
                saved_dir / "model.final_layernorm.bias.bin")
        elif "project_in" in name or "project_out" in name:
            continue
        else:
            starmap_args = []
            for i in range(len(huggingface_model_name_pattern)):
                if huggingface_model_name_pattern[i] in name:
                    new_name = name.replace(
                        "model.decoder.layers.",
                        "layers.").replace(huggingface_model_name_pattern[i],
                                           ft_model_name_pattern[i])

                    starmap_args.append(
                        (0, saved_dir, factor, new_name, args,
                         param.detach().cpu().numpy().astype(
                             np_weight_data_type), name, np_weight_data_type))
            if args.processes == 1:
                for star_args in starmap_args:
                    split_and_convert_process(*star_args)
            else:
                promises.append(
                    pool.starmap_async(split_and_convert_process, starmap_args))

    # check if async calls have raised exceptions
    for promise in promises:
        promise.get()

    pool.close()
    pool.join()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-saved_dir',
                        '-o',
                        type=str,
                        help='file name of output file',
                        required=True)
    parser.add_argument('-in_file',
                        '-i',
                        type=str,
                        help='file name of input checkpoint file',
                        required=True)
    parser.add_argument('-trained_gpu_num',
                        '-t_g',
                        type=int,
                        help='How many gpus for inference',
                        default=1)
    parser.add_argument('-infer_gpu_num',
                        '-i_g',
                        type=int,
                        help='How many gpus for inference',
                        required=True)
    parser.add_argument(
        "-processes",
        "-p",
        type=int,
        help="How many processes to spawn for conversion (default: 4)",
        default=4)
    parser.add_argument("-weight_data_type",
                        type=str,
                        default="fp32",
                        choices=["fp32", "fp16"])
    parser.add_argument('--log_level', type=str, default='info')

    args = parser.parse_args()
    logger.set_level(args.log_level)

    logger.info("\n=============== Argument ===============")
    for key in vars(args):
        logger.info(f"{key}: {vars(args)[key]}")
    logger.info("========================================")

    start_time = datetime.now()
    split_and_convert(args)
    stop_time = datetime.now()
    run_time = (stop_time - start_time)
    logger.info(f"[INFO] Spend {run_time} (h:m:s) to convert the model")
