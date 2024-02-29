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
import logging
import multiprocessing
import os
from datetime import datetime
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import torch  # pytype: disable=import-error
from transformers import T5ForConditionalGeneration

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy

LOGGER = logging.getLogger(__name__)

extra_configs = {
    "structure": {
        "t5_with_bias": "false",
        "use_gated_activation": "false",
        "position_embedding_type": "relative",
        'model_type': 't5'
    }
}


def fuse_qkv(model, factor, saved_dir):

    def get_attn_module(component, block, layer, attn_type):
        m = getattr(model, component)
        m = m.block[int(block)].layer[int(layer)]
        m = getattr(m, attn_type)
        return m

    for name, param in model.named_parameters():
        if 'Attention.q' in name:
            q = param
            component, _, block_idx, _, layer_idx, attn_type, *_ = name.split(
                '.')
            attn_mdl = get_attn_module(component, block_idx, layer_idx,
                                       attn_type)
            shape = q.shape  # (d_out, d_in)
            qkv = torch.cat([q, attn_mdl.k.weight, attn_mdl.v.weight],
                            dim=0).reshape([3, shape[0],
                                            shape[1]])  # (3, d_out, d_in)
            qkv = torch_to_numpy(qkv)
            # embed_dim --> hidden_dim qkv projection weights, [3, hidden_dim, embed_dim] split dim=1
            # ColumnLinear projection weights W=[3*d_out/TP, d_in], split dim=0 or dim=1 in [3, d_out/TP, d_in]
            split_dim = 1
            split_vals = np.split(qkv, factor, axis=split_dim)
            for j in range(factor):
                saved_path = saved_dir / f"{component}.block.{block_idx}.layer.{layer_idx}.{attn_type}.qkv.weight.{j}.bin"
                split_vals[j].tofile(saved_path.as_posix())


def split_and_convert_process(key, val, factor, saved_dir):
    # The split_factor indicates the number of ranks to implement
    # distributed GEMMs. For Tensor Parallelism, each rank/GPU works
    # on split_hidden_dim // split_factor channels.

    saved_key = key
    LOGGER.debug(f"key: {key}, val.shape: {val.shape}")

    if "shared.weight" in key or "layer_norm.weight" in key:
        # embedding table / layernorm weight, no split
        saved_path = saved_dir / f"{saved_key}.bin"
        val.tofile(saved_path.as_posix())

    elif "relative_attention_bias" in key:
        # relative attention table, transpose [num_buckets, num_heads] -> [num_heads, num_buckets]
        # and split on num_heads // split_factor dim
        split_dim = 0
        val = np.ascontiguousarray(val.transpose(1, 0))
        if val.shape[0] % factor != 0:
            LOGGER.error(
                f"[ERROR] Relative attention table, number of heads {val.shape[0]} is not divisible by TP size {factor}!"
            )  # assert doesn't work
        split_vals = np.split(val, factor, axis=split_dim)
        for j in range(factor):
            saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    elif ("SelfAttention.o.weight" in key or "EncDecAttention.o.weight" in key
          or "DenseReluDense.wo.weight" in key):
        # RowLinear projection weight W=[d_out, d_in/TP], split dim=-1
        split_dim = -1
        split_vals = np.split(val, factor, axis=split_dim)
        for j in range(factor):
            saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    elif ("lm_head.weight" in key or "DenseReluDense.wi.weight" in key
          or "DenseReluDense.wi_0.weight" in key
          or "DenseReluDense.wi_1.weight" in key):
        # ColumnLinear projection weights W=[d_out/TP, d_in], split dim=0
        split_dim = 0
        if "DenseReluDense.wi_0.weight" in key:
            saved_key = key.replace("wi_0", "wi")
        elif "DenseReluDense.wi_1.weight" in key:
            saved_key = key.replace("wi_1", "wi2")
        split_vals = np.split(val, factor, axis=split_dim)
        for j in range(factor):
            saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    elif (("encoder" in key and
           ("SelfAttention.q.weight" in key or "SelfAttention.k.weight" in key
            or "SelfAttention.v.weight" in key)) or
          ("decoder" in key and
           ("SelfAttention.q.weight" in key or "SelfAttention.k.weight" in key
            or "SelfAttention.v.weight" in key or "EncDecAttention.q.weight"
            in key or "EncDecAttention.k.weight" in key
            or "EncDecAttention.v.weight" in key))):
        # weight needs to be fused, handled by fuse_qkv()
        pass

    elif "encoder.embed_tokens.weight" in key or "decoder.embed_tokens.weight" in key:
        LOGGER.warning(f"Not save {key}, using shared.weight directly.")

    else:
        LOGGER.warning(f"cannot find key '{key}' with shape {val.shape}")


def convert_checkpoint(args):
    saved_dir = Path(args.output_dir) / f"tp{args.inference_tensor_para_size}"
    saved_dir.mkdir(parents=True, exist_ok=True)

    t5_model = T5ForConditionalGeneration.from_pretrained(args.input_dir)
    t5_model = t5_model.to(str_dtype_to_torch(args.weight_data_type))

    config = configparser.ConfigParser()
    extra_configs["structure"]["use_gated_activation"] = str(
        t5_model.encoder.config.is_gated_act)

    config["encoder"] = {}
    for key, val in t5_model.encoder.config.to_dict().items():
        config["encoder"][key] = f"{val}"
    config["encoder"]["weight_data_type"] = args.weight_data_type

    # manually set q_scaling to offset attention scaling's effect.
    # TODO: modify kernels to control whether to disable attention scaling
    def get_offset_q_scaling(config) -> str:
        d_model = config.d_model
        num_heads = config.num_heads
        head_size = d_model / num_heads
        scaling = 1 / head_size**.5
        return str(scaling)

    config["encoder"]["q_scaling"] = get_offset_q_scaling(
        t5_model.encoder.config)

    config["decoder"] = {}
    for key, val in t5_model.decoder.config.to_dict().items():
        config["decoder"][key] = f"{val}"
    config["decoder"]["weight_data_type"] = args.weight_data_type

    config["decoder"]["q_scaling"] = get_offset_q_scaling(
        t5_model.decoder.config)

    for key, val in extra_configs.items():
        config[key] = {}
        for val_key, val_val in val.items():
            config[key][val_key] = val_val
    with open((saved_dir / f"config.ini").as_posix(), 'w') as configfile:
        config.write(configfile)

    i_gpu_num = args.inference_tensor_para_size

    pool = multiprocessing.Pool(args.processes)
    pool.starmap_async(split_and_convert_process,
                       [(name, torch_to_numpy(param), i_gpu_num, saved_dir)
                        for name, param in t5_model.state_dict().items()])

    pool.close()
    pool.join()

    fuse_qkv(t5_model, i_gpu_num, saved_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input_dir",
                        "-i",
                        type=str,
                        help="Path to the framework checkpoint file",
                        required=True)
    parser.add_argument("--output_dir",
                        "-o",
                        type=str,
                        help="Path to the converted TRT-LLM model weight file",
                        required=True)
    parser.add_argument("--inference_tensor_para_size",
                        "-i_g",
                        type=int,
                        help="How many gpus for inference",
                        required=True)
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        help="How many processes to spawn for conversion (default: 4)",
        default=4)
    parser.add_argument("--weight_data_type",
                        type=str,
                        default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Provide verbose messages")
    args = parser.parse_args()
    log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format=log_format)
    LOGGER.info("\n=============== Argument ===============")
    for key in vars(args):
        LOGGER.info(f"{key}: {vars(args)[key]}")
    LOGGER.info("========================================")

    start_time = datetime.now()
    convert_checkpoint(args)
    stop_time = datetime.now()
    run_time = (stop_time - start_time)
    LOGGER.info("Spend {} (h:m:s) to convert the model".format(run_time))
