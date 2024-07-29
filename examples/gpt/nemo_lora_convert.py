#! /usr/bin/env python3
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
import datetime
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from tensorrt_llm._utils import str_dtype_to_torch, to_json_file, torch_to_numpy
from tensorrt_llm.lora_manager import LoraManager, get_all_nemo_lora_weights
from tensorrt_llm.models.gpt.convert import cpu_map_location, unpack_nemo_ckpt

log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
logging.basicConfig(format=log_format)
LOGGER = logging.getLogger(__name__)


def get_lora_keys(layer_idx):
    in_key = f'model.language_model.encoder.layers.{layer_idx}.self_attention.adapter_layer.lora_kqv_adapter.linear_in.weight'
    out_key = f'model.language_model.encoder.layers.{layer_idx}.self_attention.adapter_layer.lora_kqv_adapter.linear_out.weight'
    return in_key, out_key


def save_val(val, dir, key, tp_num=None, write_npy=False):
    ext = "npy" if write_npy else "bin"
    suffix = ext if tp_num is None else f"{tp_num}.{ext}"
    if write_npy:
        np.save(dir / f"model.{key}.{suffix}", val)
    else:
        val.tofile(dir / f"model.{key}.{suffix}")


def lora_convert(out_dir, lora_config, lora_weights, customization_id,
                 precision):
    saved_dir = Path(out_dir)
    saved_dir.mkdir(parents=True, exist_ok=True)
    num_layers = int(lora_config["num_layers"])
    config = {"lora_config": {"lora_kqv_adapter": {}}}
    config['lora_config']['precision'] = precision
    layer_weights = get_all_nemo_lora_weights(lora_weights)
    for layer_idx in range(num_layers):
        linear_in_weight = layer_weights[layer_idx]['in']
        linear_out_weight = layer_weights[layer_idx]['out']
        config["lora_config"]["lora_kqv_adapter"]["0"] = {
            "key": f"{customization_id}",
            "low_rank": f"{linear_in_weight.shape[0]}",
        }

        # do something else here.  just choose some key instead of basing it on the nemo key
        in_key, out_key = get_lora_keys(layer_idx)

        save_val(
            torch_to_numpy(
                linear_in_weight.transpose(
                    1, 0).contiguous().to(dtype=str_dtype_to_torch(precision))),
            saved_dir,
            in_key.replace("lora_kqv_adapter", f"lora_kqv_adapter.{0}"))
        save_val(
            torch_to_numpy(
                linear_out_weight.transpose(
                    1, 0).contiguous().to(dtype=str_dtype_to_torch(precision))),
            saved_dir,
            out_key.replace("lora_kqv_adapter", f"lora_kqv_adapter.{0}"))

        to_json_file(config, saved_dir / "lora_weights.json")


def lora_convert_cpp_runtime(out_dir,
                             lora_config,
                             lora_weights,
                             precision='float16'):
    saved_dir = Path(out_dir)
    saved_dir.mkdir(parents=True, exist_ok=True)
    num_layers = int(lora_config["num_layers"])
    weights = []
    weight_config = []
    layer_weights = get_all_nemo_lora_weights(lora_weights)
    for layer_idx in range(num_layers):
        in_weights = layer_weights[layer_idx]['in']
        out_weights = layer_weights[layer_idx]['out']
        LOGGER.debug(f"layer {layer_idx} in_weights: {in_weights.shape}")
        LOGGER.debug(f"layer {layer_idx} out_weights: {out_weights.shape}")
        in_out_weights = []
        adapter_size = 0
        for w, inout in ((in_weights, "in"), (out_weights, "out")):
            assert len(w.shape) == 2
            # assume that the hidden dim is the larger of the 2
            dim0 = w.shape[0]
            dim1 = w.shape[1]
            adapter_size = min(dim0, dim1)
            # in_weights should have shape [adaper_size, hidden]
            if dim1 < dim0 and inout == "in":
                adapter_size = dim1
                w = w.transpose(1, 0)
            # out_weights should have shape [hidden, adapter_size]
            elif dim0 < dim1 and inout == "out":
                adapter_size = dim0
                w = w.transpose(1, 0)

            w = w.contiguous().flatten().to(dtype=str_dtype_to_torch(precision))
            in_out_weights.append(w)
        in_out_weights = torch.concatenate(in_out_weights).flatten().numpy()
        weights.append(in_out_weights)
        weight_config.append(
            np.array([
                LoraManager.LORA_MODULE_IDS["attn_qkv"], layer_idx, adapter_size
            ],
                     dtype=np.int32))
    all_weights = np.expand_dims(np.stack(weights), 0)
    all_configs = np.expand_dims(np.stack(weight_config), 0)

    save_val(all_weights,
             saved_dir,
             "lora_weights",
             tp_num=None,
             write_npy=True)
    save_val(all_configs, saved_dir, "lora_config", tp_num=None, write_npy=True)


def main(args):
    start_time = datetime.datetime.now()
    with tempfile.TemporaryDirectory() as prompt_out_dir:
        prompt_out_dir = Path(prompt_out_dir)
        unpack_nemo_ckpt(args.in_file, prompt_out_dir)
        LOGGER.info("Spent %s (h:m:s) to unpack NeMo prompt archive",
                    datetime.datetime.now() - start_time)

        model_weights_ckpt = "model_weights.ckpt"
        with open(prompt_out_dir / "model_config.yaml") as f:
            prompt_config = yaml.full_load(f)
        LOGGER.debug(prompt_config)

        start_time = datetime.datetime.now()
        weight_path = prompt_out_dir / model_weights_ckpt

        prompt_weights = torch.load(
            weight_path,
            map_location=cpu_map_location,
        )
    if args.write_cpp_runtime_tensors:
        lora_convert_cpp_runtime(args.out_dir,
                                 prompt_config,
                                 prompt_weights,
                                 precision=args.storage_type)
    else:
        lora_convert(args.out_dir,
                     prompt_config,
                     prompt_weights,
                     args.customization_id,
                     precision=args.storage_type)

    LOGGER.info("Spent %s (h:m:s) to convert the prompt model",
                datetime.datetime.now() - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out-dir',
        '-o',
        type=Path,
        help='path to output embedding table file in the .npy format',
        required=True)
    parser.add_argument('--in-file',
                        '-i',
                        type=Path,
                        help='path to input prompt-tuning checkpoint file',
                        required=True)
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Provide verbose messages")
    parser.add_argument("--customization-id", type=str, default="lora")
    parser.add_argument("--write-cpp-runtime-tensors",
                        action="store_true",
                        default=False)
    parser.add_argument("--storage-type",
                        type=str,
                        default="float16",
                        choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    LOGGER.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    main(args)
